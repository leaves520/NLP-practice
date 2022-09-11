import torch
import numpy as np
import time

class ConfigDPCNN(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'DPCNN'
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

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 50                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 1000                                            # epoch数
        self.batch_size = 16                                           # mini-batch大小
        self.pad_size = 512                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-4                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.num_filters = 64                                          # 卷积核数量(channels数)



class ConfigTextRNN_Att(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN_Att'
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

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 50                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 1000                                            # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.pad_size = 512                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.hidden_size2 = 64




class ConfigTextRCNN(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRCNN'
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
        self.save_path = dataset + '/saved_dict_全客人/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        # self.embedding_pretrained = torch.tensor(
        #     np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
        #     if embedding != 'random' else None                                       # 预训练词向量

        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding).astype('float32'))\
            if embedding != 'random' else None
        # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 1.0                                              # 随机失活
        self.require_improvement = 20                               # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 100                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 200                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 1                                             # lstm层数



class ConfigTextRNN(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict_全客人/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding).astype('float32')) if embedding != 'random' else None
        # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 50                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 200                                            # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数



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
        self.require_improvement = 100                                # 若超过1000batch效果还没提升，则提前结束训练
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
        self.num_filters = 64                                         # 卷积核数量(channels数)



        # # test
        # self.dropout = 0.5                                              # 随机失活
        # self.require_improvement = 50                                # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_classes = len(self.class_list)                         # 类别数
        # self.n_vocab = 0                                                # 词表大小，在运行时赋值
        # self.num_epochs = 1000                                          # epoch数
        # self.batch_size = 16                                          # mini-batch大小
        # self.pad_size = 300                                           # 每句话处理成的长度(短填长切)
        # self.learning_rate = 0.0001# 1e-3                                       # 学习率
        # self.embed = self.embedding_pretrained.size(1)\
        #     if self.embedding_pretrained is not None else 300           # 字向量维度
        # self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        # self.num_filters = 64                                         # 卷积核数量(channels数)


class ConfigFastText(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'FastText'
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

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 50                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 1000                                            # epoch数
        self.batch_size = 16                                           # mini-batch大小
        self.pad_size = 300                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 0.0001                                      # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.hidden_size = 64                                         # 隐藏层大小
        self.n_gram_vocab = 250499                                      # ngram 词表大小


class ConfigHan(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'HAN'
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
        self.require_improvement = 50                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 1000                                         # epoch数
        self.batch_size = 16                                          # mini-batch大小
        self.pad_size = 300                                          # 每句话处理成的长度(短填长切)
        self.learning_rate = 0.0001# 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.hidden_size = 64   # 隐层大小