import torch.nn as nn
import random
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
import jieba
# from snownlp import SnowNLP
# import pycantonese
from nlpcda import Randomword, Homophone, Similarword, RandomDeleteChar
import re

class ConfigTextCNN(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集

        try:
            self.class_list = [x.strip() for x in open(
                dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        except:
            self.class_list = [x.strip() for x in open(
                dataset + '/data/label.txt', encoding='utf-8').readlines()]

        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.log_path = dataset + '/log/' + self.model_name + '/' + time.strftime('%m.%d-%H.%M', time.localtime())
        self.save_path = f'{self.log_path}/' + self.model_name + '.ckpt'  # 模型训练结果


        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding).astype('float32'))\
            if embedding != 'random' else None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        # 体验不佳场景最优
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 200                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 1000                                         # epoch数
        self.batch_size = 16                                          # mini-batch大小
        self.pad_size = 512                                         # 每句话处理成的长度(短填长切)
        self.learning_rate = 0.0001# 1e-3                                       # 学习率
        # self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3)                                   # 卷积核尺寸
        self.num_filters = 32                                         # 卷积核数量(channels数)


## char-word-mixup

class Highway(nn.Module):
    """
    Input shape=(batch_size,dim,dim)
    Output shape=(batch_size,dim,dim)
    """

    def __init__(self, layer_num, dim=300):
        super(Highway, self).__init__()
        self.layer_num = layer_num
        self.linear = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(self.layer_num)])
        self.gate = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(self.layer_num)])

    def forward(self, x):
        for i in range(self.layer_num):
            gate = torch.sigmoid(self.gate[i](x))
            nonlinear = F.relu(self.linear[i](x))
            x = gate * nonlinear + (1 - gate) * x
        return x



class MixUpEmbedding(nn.Module):
    """
    word and char embedding
    Input shape: word_emb=(batch_size,sentence_length,emb_size) char_emb=(batch_size,sentence_length,word_length,emb_size)
    Output shape: y= (batch_size,sentence_length,word_emb_size+char_emb_size)
    """

    def __init__(self, highway_layers, word_dim, char_dim):
        super(MixUpEmbedding, self).__init__()
        self.highway = Highway(highway_layers, word_dim + char_dim)

    def forward(self, word_emb, char_emb):
        char_emb, _ = torch.max(char_emb, 2)

        emb = torch.cat([word_emb, char_emb], dim=2)
        emb = self.highway(emb)

        return emb


class TextCNNHighway(nn.Module):
    def __init__(self, word_dim, char_dim, n_filters, filter_sizes, output_dim,
                 dropout, word_emb, char_emb, highway_layers):
        super().__init__()

        self.char_embedding = nn.Embedding.from_pretrained(char_emb, freeze=False)
        self.word_embedding = nn.Embedding.from_pretrained(
            word_emb, freeze=False)

        self.text_embedding = Embedding(highway_layers, word_dim, char_dim)

        self.convs = Conv1d(word_dim + char_dim, n_filters, filter_sizes)

        self.fc = Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text_word, text_char):
        text_word, _ = text_word

        word_emb = self.word_embedding(text_word)
        char_emb = self.char_embedding(text_char)

        char_emb = char_emb.permute(1, 0, 2, 3)

        text_emb = self.text_embedding(word_emb, char_emb)
        # text_emb: [sent len, batch size, emb dim]

        text_emb = text_emb.permute(1, 2, 0)
        # text_emb: [batch size, emb dim, sent len]

        conved = self.convs(text_emb)

        # conv_n = [batch size, n_filters, sent len - filter_sizes[n] - 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.text_embedding = MixUpEmbedding(highway_layers=2, word_dim=config.embed, char_dim=config.embed)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed*2)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

        self.backup = {}

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        word_x, char_x, seq = x
        word_emb = self.embedding(word_x)
        char_emb = self.embedding(char_x)
        out = self.text_embedding(word_emb, char_emb)

        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        sample_feature = out

        out = self.dropout(out)
        out = self.fc(out)
        return out, sample_feature

    def attack(self, epsilon=1., emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]

        self.backup = {}



def clean_data(desstr,restr=''):
    # 过滤制表空格
    desstr = re.sub('\u200d','',desstr)

    # 过滤左右空格
    desstr = re.sub('\\s{1,}|\t','', desstr)


    # # 过滤转人工前
    # tmp = re.search('轉人工|人工|真人', desstr)
    # if tmp:
    #     span = tmp.span()[-1]
    #     desstr = desstr[span:]

    #过滤表情
    try:
        co = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')
    except re.error:
        co = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')

    desstr = co.sub(restr, desstr)

    # 过滤订单号:
    desstr = re.sub('訂單.{0,4}\d{11}','', desstr)

    # 过滤单个数字:
    desstr = re.sub('\d[,.。，]', '', desstr)

    # 过滤 
    desstr = re.sub(' ','', desstr)

    # 过滤邮箱
    desstr = re.sub('[^\u4e00-\u9fa5]*@.*com','', desstr)

    # 过滤年月日
    desstr = re.sub('[0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日[0-9]{1,2}:[0-9]{1,2}','', desstr)
    desstr = re.sub('[0-9]+/[0-9]+/[0-9]+','', desstr)
    desstr = re.sub('[0-9]+/[0-9]+.{0,4}[0-9]+/[0-9]+','',desstr)

    # 重复标点符号过滤：
    desstr = re.sub('[!,.。，！?？]+',',', desstr)
    # desstr = re.sub('[!,.。，！]+',',', desstr)
    # desstr = re.sub('[？?],','?',desstr)

    # 过滤无用符号：
    desstr = re.sub('[{}$/’：:-]+','',desstr)

    # 过滤开头的符号
    desstr = re.sub('^,','', desstr)

    return desstr


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_dataset(config, use_word):
    # 分词的方式，是按字分还是按词分
    if use_word:
        # tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        with open(r'F:\工作与学习文件\Github_project\NLP-practice\classical_text_classification'
                  r'\data_im\data\vocab_freq.txt','r', encoding='utf-8-sig') as f:
            jieba.load_userdict(f)
        tokenizer = lambda x : list(jieba.cut(x))
        print('Use word for spliting..')

    tokenizer_char = lambda x: [y for y in x]  # char-level

    # 判断是否有词表，有词表就直接加载，没有就根据自己的分词规则进行分词且创建词表
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        raise ValueError('no vocab path')
    print(f"Vocab size: {len(vocab)}")


    def load_dataset(path, pad_size=300, add=False):  # 创建word2idx后的数据集
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                try:
                    content, label = lin.split('\t')
                except:
                    continue

                rawcontext = content
                content = clean_data(content)
                content = re.sub('當前諮詢.*details', '', content)
                content = re.sub('[a-zA-Z]+','',content)

                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        # token = token[:pad_size]
                        token = token[-pad_size:]
                        seq_len = pad_size

                # char to idx
                chars_line = [] # seq_length, word_length
                max_char_length = 20
                for tos in token:
                    if tos != '<PAD>':
                        chars_tmp = tokenizer_char(tos)
                        chars_tmp = [vocab.get(c, vocab.get(UNK)) for c in chars_tmp]
                        if len(chars_tmp) < max_char_length:
                            chars_tmp.extend([vocab.get(PAD, vocab.get(UNK))]*(max_char_length-len(chars_tmp)))
                        else:
                            chars_tmp = chars_tmp[:max_char_length]
                        chars_line.append(chars_tmp)
                    else:
                        chars_line.append([vocab.get(PAD, vocab.get(UNK))]*max_char_length)

                # word to idx
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                assert len(chars_line) == len(words_line)
                contents.append((words_line, chars_line, int(label), seq_len, rawcontext))


        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path, config.pad_size, add=False)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    print(f'trian:{len(train)}|dev:{len(dev)}|test:{len(test)}')

    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, rdrop=False):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.rdrop = rdrop

    def _to_tensor(self, datas):

        if not self.rdrop:
            x_word = torch.LongTensor([_[0] for _ in datas]).to(self.device)
            x_char = torch.LongTensor([_[1] for _ in datas]).to(self.device)
            y = torch.LongTensor([_[2] for _ in datas]).to(self.device)
            contexts  = [_[4] for _ in datas]
            # pad前的长度(超过pad_size的设为pad_size)
            seq_len = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        else:
            ## used rdrop for train
            x_word = torch.LongTensor(sum([[_[0], _[0]] for _ in datas], [])).to(self.device)
            x_char = torch.LongTensor(sum([[_[1], _[1]] for _ in datas], [])).to(self.device)
            y = torch.LongTensor(sum([[_[2], _[2]] for _ in datas], [])).to(self.device)
            contexts  = sum([[_[4], _[4]] for _ in datas], [])
            # pad前的长度(超过pad_size的设为pad_size)
            seq_len = torch.LongTensor(sum([[_[3], _[3]] for _ in datas], [])).to(self.device)

        return (x_word, x_char, seq_len), y, contexts

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            # self.shuffle_data(self.batches)
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

    def shuffle_data(self, data):
        index = list(range(len(data)))
        random.shuffle(index)
        ran_data = [data[_] for _ in index]
        self.batches = ran_data


def build_iterator(dataset, config, rdrop=False):
    iter = DatasetIterater(dataset, config.batch_size, config.device, rdrop=rdrop)
    return iter



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
import logging
import os
from Contract_loss import SupConLoss

ce = torch.nn.CrossEntropyLoss()
kld = torch.nn.KLDivLoss(reduction="none")
suploss = SupConLoss(temperature=0.1, scale_by_temperature=True)

def Rdroploss(y_pred, y_true, alpha=4):
    """配合R-Drop的交叉熵损失"""

    loss1 = ce(y_pred, y_true)
    loss2 = kld(torch.log_softmax(y_pred[::2], dim=1), y_pred[1::2].softmax(dim=-1)) + \
            kld(torch.log_softmax(y_pred[1::2], dim=1), y_pred[::2].softmax(dim=-1))

    return loss1 + torch.mean(loss2) / 4 * alpha




def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w", encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass



def train(config, model, train_iter, dev_iter, test_iter):

    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    dev_best_acc = - float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    ## 记录训练过程日志
    writer = SummaryWriter(log_dir=config.log_path)
    logger = get_logger(os.path.join(config.log_path, 'logs-remark.txt'))
    config.logger = logger
    config.logger.info(f'====== Using rdrop = {config.rdrop} ======')

    for epoch in range(config.num_epochs):
        config.logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels, _) in enumerate(train_iter):

            outputs, sample_feature = model(trains)
            if config.rdrop == False:
                loss = F.cross_entropy(outputs, labels)
            else:
                loss = Rdroploss(outputs, labels)

            # loss_sup = suploss(sample_feature, labels=labels)
            # loss = loss + loss_sup

            loss.backward()

            # 对抗训练
            model.attack(0.3)
            loss_adv =  Rdroploss(model(trains)[0], labels)
            loss_adv.backward(retain_graph=True)
            model.restore()

            # 梯度更新
            optimizer.step()
            model.zero_grad()

            if total_batch % 5 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                config.logger.info(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1

            if total_batch - last_improve > config.require_improvement:
                config.logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if flag:
            torch.save(model.state_dict(), config.save_path.replace('.ckpt', f'-acc{dev_best_acc}.ckpt'))
            break

    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    config.logger.info(msg.format(test_loss, test_acc))
    config.logger.info("Precision, Recall and F1-Score...")
    config.logger.info(test_report)
    config.logger.info("Confusion Matrix...")
    config.logger.info(test_confusion)
    time_dif = get_time_dif(start_time)
    config.logger.info(f"Time usage:{time_dif}")


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    all_contexts = []
    with torch.no_grad():
        for texts, labels, context in data_iter:
            outputs,_ = model(texts)
            loss = F.cross_entropy(outputs, labels)

            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            all_contexts.append(context)


    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4, labels=range(len(config.class_list)))
        confusion = metrics.confusion_matrix(labels_all, predict_all)

        all_cons = []
        import pandas as pd
        da = pd.DataFrame(columns=['text', 'gold', 'predict'])
        for a in all_contexts:
            all_cons.extend(a)
        da.text = all_cons
        da.gold = labels_all
        da.predict = predict_all
        da = da[da['gold']!=da['predict']]
        da.to_excel(f'./{config.log_path}/badcase.xlsx')

        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)




if __name__ == '__main__':

    rdrop = True  # 训练集是否使用rdrop
    dataset = 'data_im'  # 数据集
    embedding = 'embedding_wiki_zhe_yue.npy'

    config = ConfigTextCNN(dataset, embedding)
    config.rdrop = rdrop

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    vocab, train_data, dev_data, test_data = build_dataset(config, use_word=True)
    train_iter = build_iterator(train_data, config, rdrop=rdrop)
    dev_iter = build_iterator(dev_data, config, rdrop=False)
    test_iter = build_iterator(test_data, config, rdrop=False)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = Model(config).to(config.device)
    init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)

