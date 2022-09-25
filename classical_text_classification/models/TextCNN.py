# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict_全客人/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''
# class Model(nn.Module):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         if config.embedding_pretrained is not None:
#             self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
#         else:
#             self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
#         self.dropout = nn.Dropout(config.dropout)
#         self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
#
#         self.backup = {}
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x
#
#     def forward(self, x):
#         out = self.embedding(x[0])
#         out = out.unsqueeze(1)
#         out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
#         out = self.dropout(out)
#         out = self.fc(out)
#         return out
#
#     def attack(self, epsilon=1., emb_name='embedding'):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 self.backup[name] = param.data.clone()
#                 norm = torch.norm(param.grad)
#                 if norm != 0 and not torch.isnan(norm):
#                     r_at = epsilon * param.grad / norm
#                     param.data.add_(r_at)
#
#     def restore(self, emb_name='embedding'):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#
#         self.backup = {}


# class Model(nn.Module):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         if config.embedding_pretrained is not None:
#             self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
#         else:
#             self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
#         self.dropout = nn.Dropout(config.dropout)
#         self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
#
#         self.backup = {}
#
#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x
#
#     def forward(self, x):
#         out = self.embedding(x[0])
#         out = out.unsqueeze(1)
#         out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
#         sample_feature = out
#
#         out = self.dropout(out)
#         out = self.fc(out)
#         return out, sample_feature
#
#     def attack(self, epsilon=1., emb_name='embedding'):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 self.backup[name] = param.data.clone()
#                 norm = torch.norm(param.grad)
#                 if norm != 0 and not torch.isnan(norm):
#                     r_at = epsilon * param.grad / norm
#                     param.data.add_(r_at)
#
#     def restore(self, emb_name='embedding'):
#         # emb_name这个参数要换成你模型中embedding的参数名
#         for name, param in self.named_parameters():
#             if param.requires_grad and emb_name in name:
#                 assert name in self.backup
#                 param.data = self.backup[name]
#
#         self.backup = {}


## char-word-mixup

class Highway(nn.Module):
    """
    Input shape=(batch_size,dim,dim)
    Output shape=(batch_size,dim,dim)
    """

    def __init__(self, layer_num, dim=600):
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
        return



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
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

        self.backup = {}

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
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