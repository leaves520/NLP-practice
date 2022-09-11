# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import predict
from importlib import import_module

from utils import get_time_dif, build_dataset_predict, build_iterator_predict


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.data_path = dataset + '/data/predict.txt'                                # 训练集
        self.save_path = dataset + '/saved_dict_全客人/' + self.model_name + '_0.82.ckpt'
        try:
            self.class_list = [x.strip() for x in open(
                dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        except:
            self.class_list = [x.strip() for x in open(
                dataset + '/data/label.txt', encoding='utf-8').readlines()]

        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表

        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding).astype('float32'))\
            if embedding != 'random' else None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        # 体验不佳场景最优
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 50                                # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 1000                                          # epoch数
        self.batch_size = 16                                          # mini-batch大小
        self.pad_size = 512                                            # 每句话处理成的长度(短填长切)
        self.learning_rate = 0.0001# 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3)                                   # 卷积核尺寸
        self.num_filters = 64                                         # 卷积核数量(channels数)



if __name__ == '__main__':

    dataset = 'data_im_体验不佳'  # 数据集
    model_name = 'TextCNN'
    embedding = 'embedding_wiki_zhe_yue.npy'

    x = import_module('models.' + model_name)
    config = Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    vocab, predict_data = build_dataset_predict(config, ues_word=False)
    predict_iter = build_iterator_predict(predict_data, config)


    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    print(model.parameters)
    result = predict(config, model, predict_iter)
    print(result)