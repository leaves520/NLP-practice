# coding: UTF-8

import os
import torch
import time

class Config(object):
    def __init__(self, data_dir):
        assert os.path.exists(data_dir)
        self.train_file = os.path.join(data_dir, "train.txt")
        self.dev_file = os.path.join(data_dir, "dev.txt")
        self.test_file = os.path.join(data_dir, 'test.txt')
        self.label_file = os.path.join(data_dir, "label.txt")
        assert os.path.isfile(self.train_file)
        assert os.path.isfile(self.dev_file)
        assert os.path.isfile(self.label_file)

        self.data_path = data_dir
        self.saved_model_dir = os.path.join(data_dir, 'logs', "model{}".format(time.strftime('%m.%d-%H.%M', time.localtime())))
        # self.saved_model = os.path.join(self.saved_model_dir, "bert_model.pth")
        self.saved_model = os.path.join(self.saved_model_dir, "bert_model.pth")

        if not os.path.exists(self.saved_model_dir):
            os.makedirs(self.saved_model_dir)

        self.label_list = [label.strip() for label in open(self.label_file, "r", encoding="UTF-8").readlines()]
        self.num_labels = len(self.label_list)

        self.num_epochs = 3
        self.log_batch = 4
        self.batch_size = 16
        self.max_seq_len = 512
        self.require_improvement = 1000

        self.warmup_steps = 0
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.learning_rate = 5e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Config_multi(object):
    def __init__(self, data_dir):
        assert os.path.exists(data_dir)
        self.train_file = os.path.join(data_dir, "train.txt")
        self.dev_file = os.path.join(data_dir, "dev.txt")
        self.test_file = os.path.join(data_dir, 'test.txt')
        self.label_file = os.path.join(data_dir, "multi_task/label_one.txt")
        self.label_file = os.path.join(data_dir, 'multi_task/label_two.txt')
        assert os.path.isfile(self.train_file)
        assert os.path.isfile(self.dev_file)
        assert os.path.isfile(self.label_file)

        self.data_path = data_dir
        self.saved_model_dir = os.path.join(data_dir, 'logs', "model{}".format(time.strftime('%m.%d-%H.%M', time.localtime())))
        self.saved_model = os.path.join(self.saved_model_dir, "bert_model.pth")
        if not os.path.exists(self.saved_model_dir):
            os.makedirs(self.saved_model_dir)

        self.label_list = [label.strip() for label in open(self.label_file, "r", encoding="UTF-8").readlines()]
        self.num_labels = len(self.label_list)

        self.num_epochs = 3
        self.log_batch = 4
        self.batch_size = 16
        self.max_seq_len = 512
        self.require_improvement = 1000

        self.warmup_steps = 0
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.learning_rate = 5e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")