# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import jieba
# from snownlp import SnowNLP
import pycantonese

MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, use_word):
    # 分词的方式，是按字分还是按词分
    if use_word:
        # tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        # with open(r'D:\Users\xbye\Desktop\Chinese-Text-Classification-Pytorch-master\data_im_体验不佳\data\vocab.pkl','rb') as f:
        #     jieba.load_userdict(pkl.load(f))
        # tokenizer = lambda x : list(jieba.cut(x))  # jieba分词
        tokenizer = lambda x: pycantonese.segment(x)  # 使用pycantonese
        print('Use word for spliting..')
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    # 判断是否有词表，有词表就直接加载，没有就根据自己的分词规则进行分词且创建词表
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")


    def load_dataset(path, pad_size=300):  # 创建word2idx后的数据集
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

                # # # 去除英文文本
                # import re
                # content = re.sub('[a-zA-Z]+','',content)
                # content = re.sub('[.。,，!！？?]+', '', content)

                # # 繁转简中
                # s_jianti = SnowNLP(content)
                # content = s_jianti.han

                words_line = []
                token = tokenizer(content)

                # # 当使用pycantonese对应的词不在时，将其按字符划分
                # def filters(words):
                #     def is_number(s):
                #         try:
                #             float(s)
                #             return True
                #         except ValueError:
                #             pass
                #         try:
                #             import unicodedata
                #             unicodedata.numeric(s)
                #             return True
                #         except (TypeError, ValueError):
                #             pass
                #         return False
                #
                #     def contain_zh(word):
                #
                #         zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
                #         match = zh_pattern.search(word)
                #
                #         return match
                #
                #     new_tokens = []
                #     for w in words:
                #         if w not in vocab:
                #             if not is_number(w) and contain_zh(w):
                #                 for char in w:
                #                     new_tokens.append(char)
                #         else:
                #             new_tokens.append(w)
                #
                #     return new_tokens
                # token = filters(token)
                # # 当使用pycantonese对应的词不在时，将其按字符划分

                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size

                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len, content))

        return contents  # [([...], 0), ([...], 1), ...]

    train = load_dataset(config.train_path, config.pad_size)
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
            x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
            y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
            contexts  = [_[3] for _ in datas]
            # pad前的长度(超过pad_size的设为pad_size)
            seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        else:
            ## used rdrop for train
            x = torch.LongTensor(sum([[_[0], _[0]] for _ in datas], [])).to(self.device)
            y = torch.LongTensor(sum([[_[1], _[1]] for _ in datas], [])).to(self.device)
            contexts  = sum([[_[3], _[3]] for _ in datas], [])
            # pad前的长度(超过pad_size的设为pad_size)
            seq_len = torch.LongTensor(sum([[_[2], _[2]] for _ in datas], [])).to(self.device)

        return (x, seq_len), y, contexts

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
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



def build_iterator(dataset, config, rdrop=False):
    iter = DatasetIterater(dataset, config.batch_size, config.device, rdrop=rdrop)
    return iter



def build_dataset_predict(config, ues_word):
    # 分词的方式，是按字分还是按词分
    if ues_word:
        # tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
        tokenizer = lambda x : list(jieba.cut(x))
    else:
        tokenizer = lambda x: [y for y in x]  # char-level

    # 判断是否有词表，有词表就直接加载，没有就根据自己的分词规则进行分词且创建词表
    try:
        vocab = pkl.load(open(config.vocab_path, 'rb'))
        print(f"Vocab size: {len(vocab)}")
    except Exception as e:
        print('error in vocab')


    def load_dataset(path, pad_size=300):  # 创建word2idx后的数据集
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                try:
                    content = lin
                except:
                    continue

                import re
                content = re.sub('[a-zA-Z]+','',content)
                content = re.sub('[.。,，!！？?]+', '', content)
                # content = re.sub('\d+', '', content)

                words_line = []
                token = tokenizer(content)

                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, seq_len, content))

        return contents

    predict = load_dataset(config.data_path, config.pad_size)

    print(f'predict:{len(predict)}')

    return vocab, predict


class DatasetIterater_predict(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        contexts  = [_[2] for _ in datas]
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        return (x, seq_len), contexts

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
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



def build_iterator_predict(dataset, config):
    iter = DatasetIterater_predict(dataset, config.batch_size, config.device)
    return iter




def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))



if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
