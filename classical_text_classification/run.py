# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from Config import *

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, default='TextCNN', help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':

    rdrop = True # 训练集是否使用rdrop
    dataset = 'data_im_未入住取消'  # 数据集
    embedding = 'embedding_wiki_zhe_yue.npy'
    # embedding = 'embedding_wiki_simple.npy'
    model_name = args.model


    if args.embedding == 'random':
        embedding = 'random'

    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)

    if model_name == 'TextCNN':
        config = ConfigTextCNN(dataset, embedding)
    elif model_name == 'TextRCNN':
        config = ConfigTextRCNN(dataset, embedding)
    elif model_name == 'TextRNN_Att':
        config = ConfigTextRNN_Att(dataset, embedding)
    elif model_name == 'TextRNN':
        config = ConfigTextRNN(dataset, embedding)
    elif model_name == 'FastText':
        config = ConfigFastText(dataset, embedding)
    elif model_name == 'DPCNN':
        config = ConfigDPCNN(dataset, embedding)
    elif model_name == 'HAN':
        config = ConfigHan(dataset, embedding)
    else:
        raise ValueError(f'not support model {model_name}')

    config.rdrop = rdrop

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config, rdrop=rdrop)
    dev_iter = build_iterator(dev_data, config, rdrop=False)
    test_iter = build_iterator(test_data, config, rdrop=False)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
