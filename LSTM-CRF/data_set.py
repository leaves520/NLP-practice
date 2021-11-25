from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence  # pad, 针对不同长度的句子进行padding
import gensim.downloader
import numpy as np
import torch
import json

label_dict = {"<START>": 0, "P": 1, "M": 2, "MP": 3, "O": 4, "<END>": 5}
id2label = {v: k for k, v in label_dict.items()}  # 建立输出label的token -> 字符label的表示
special_token_list = ["<PADDING>", "<UNKNOWN>"]  # 0: padding, 1: unknown


# options
class ModelEmbedding:

    def __init__(self, model_name='glove-wiki-gigaword-50'):
        # 使用gensim训练好的词汇向量矩阵
        self.w2v_object = gensim.downloader.load(model_name)
        self.word2vector = self.w2v_object.vectors
        self.input_word_list = special_token_list + self.w2v_object.index_to_key  # 输入的总词数(记得加上新增的词汇)

        self.embedding = np.concatenate((np.random.rand(2, self.word2vector.shape[-1]), self.word2vector))
        self.embedding_size = self.word2vector.shape[-1]
        self.vocab_size = len(self.input_word_list)
        self.n_class = len(label_dict)
        self._get_word2id()

    def _get_word2id(self):
        # 建立字符 -> token映射
        self.word2id = dict(zip(self.input_word_list, list(range(len(self.input_word_list)))))


class MyDataSet(Dataset):

    def __init__(self, json_path, word2id, test=False):
        file = open(json_path, 'r')
        if test:
            self.raw_data = json.load(file)[:100]  # 5000 for demo testing
        else:
            self.raw_data = json.load(file)

        # 将训练数据x和label_y从字符转化为数字token去表示
        self.data_x = [[word2id[w] if w in word2id else word2id["<UNKNOWN>"] for w in item[0]] for item in
                       self.raw_data]
        self.data_y = [[label_dict[label] for label in item[1]] for item in self.raw_data]

    def __getitem__(self, idx):
        return torch.Tensor(self.data_x[idx]), torch.Tensor(self.data_y[idx])

    def __len__(self):
        return len(self.raw_data)


def collate_fn(batch):  # 将每个batch的数据组织并返回相应的可训练格式
    """
    :param batch: (batch_num, ([sentence_len, word_embedding], [sentence_len]))
    :return:
    """
    x_list = [x[0] for x in batch]
    y_list = [x[1] for x in batch]
    lengths = [len(item[0]) for item in batch]

    x_list = pad_sequence(x_list, padding_value=0)  # 针对不同长度的训练数据进行<PAD>填充
    y_list = pad_sequence(y_list, padding_value=-1)

    return x_list.transpose(0, 1), y_list.transpose(0, 1), lengths


if __name__ == '__main__':
    # download the pretrained-model
    # import gensim.downloader
    # glove_vectors = gensim.downloader.load('glove-wiki-gigaword-50')
    # print(glove_vectors)

    # test mydataset
    t = MyDataSet(json_path='./labeled_data.json', word2id=ModelEmbedding().word2id, test=True)
    print(t[0])
