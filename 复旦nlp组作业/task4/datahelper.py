import os
import pickle # pickle方式进行保存，打开文件需要wb,rb
import collections
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Vocabulary(object):
    def __init__(self, words_list, label_list, chars_list, Padding=True, Unkonwn=True, POS=True):
        if Padding:
            words_list = ['<PAD>'] + words_list
            chars_list = ['<PAD>'] + chars_list
        if Unkonwn:
            words_list = words_list + ['UNK']
            chars_list = chars_list + ['UNK']
            self.unk_idx_word = len(words_list) - 1
            self.unk_idx_char = len(chars_list) - 1
        if POS:
            label_list = ['<S>'] + label_list + ['<E>']

        self.c2idx = {c:i for i, c in enumerate(chars_list)}
        self.idx2c = {i:c for i, c in enumerate(chars_list)}
        self.w2idx = {w:i for i, w in enumerate(words_list)}
        self.idx2w = {i:w for i, w in enumerate(words_list)}
        self.la2idx = {la:i for i,la in enumerate(label_list)}
        self.idx2la = {i:la for i,la in enumerate(label_list)}

    def char2idx(self, char):
        idx = self.c2idx.get(char, self.unk_idx_char)
        assert self.unk_idx_char is not None, ValueError('char is not in vocab. But also no UNK. ')
        return idx

    def idx2char(self, idx):
        assert 0 <= idx < len(self.idx2c), IndexError()
        return self.idx2c[idx]


    def word2idx(self, word):
        idx = self.w2idx.get(word, self.unk_idx_word)
        assert self.unk_idx_word is not None, ValueError('Word is not in vocab. But also no UNK. ')
        return idx

    def idx2word(self, idx):
        assert 0 <= idx < len(self.idx2w), IndexError()
        return self.idx2w[idx]

    def label2idx(self, label):
        return self.la2idx[label]

    def idx2label(self, idx):
        assert 0 <= idx < len(self.la2idx), IndexError()
        return self.idx2la[idx]

    @property
    def size(self):
        return {'vocab_word':len(self.w2idx), 'vocab_char':len(self.c2idx), 'label': len(self.la2idx)}



class CoNLL_DataSet:
    def __init__(self, path='../dataset/Conll_2003', sample=0.1):
        self.MaxCharLen = 0
        self.RawDatset = self.read(path)
        self.trn_x, self.trn_y, self.val_x, self.val_y, \
            self.test_x, self.test_y, self.vocabulary = self.process(sample=sample)


    def read(self, path):
        # read data from raw file.
        if os.path.isfile(path+'/read_data.pkl'):
            print('Read_data is already Finish. (Saved in {})'.format(path+'/read_data.pkl'))
            with open(path+'/read_data.pkl', mode='rb') as f:
                return pickle.load(f)

        def read_txt(data_path):
            data_x, data_y = [], []
            with open(data_path,encoding='utf8',mode='r') as file:
                cur_words, cur_ners = [], []
                flag = False
                for line in file:
                    line = line.strip()
                    if '-DOCSTART' in line:
                        continue

                    if line == '' and flag:
                        flag = False
                        assert len(cur_ners) == len(cur_words)
                        data_x.append(cur_words)
                        data_y.append(cur_ners)
                        cur_words, cur_ners = [], []
                    elif line != '':
                        word, _, _, ner_label = line.split(' ')
                        cur_words.append(word.lower())
                        cur_ners.append(ner_label)
                        flag = True

            return data_x, data_y


        dataset = collections.defaultdict(dict)
        for type in ['train','valid','test']:
            file_path = path + f'/{type}.txt'
            self.__check_file(file_path)
            x, y = read_txt(file_path)
            dataset[type]['x'] = x
            dataset[type]['y'] = y

        with open(path+'/read_data.pkl', mode='wb') as f:
            pickle.dump(dataset, f)

        print('Read data Finish. (It will be saved in “read_data.pkl”)')

        return dataset



    def process(self, sample):
        # bulid the vocabulary.
        train_x, train_y = self.RawDatset['train']['x'], self.RawDatset['train']['y']
        val_x, val_y = self.RawDatset['valid']['x'], self.RawDatset['valid']['y']
        test_x, test_y = self.RawDatset['test']['x'], self.RawDatset['test']['y']

        data = train_x + val_x + test_x
        y = train_y + val_y + test_y

        if sample != 1:
            cutoff = int(len(train_x)*sample)
            train_x, train_y = train_x[:cutoff], train_y[:cutoff]
            print('Sample some Training data ({}|{}) for testing code.'.format(sample, cutoff))

        chars = set()
        words = set()
        labels = set()

        for i in range(len(data)):
            words = words | set(data[i])
            labels = labels | set(y[i])
            chars = chars | set(''.join(data[i]))
            cur_max_char_lens = max(list(map(len, data[i])))
            self.MaxCharLen = max(self.MaxCharLen, cur_max_char_lens)

        words = list(words)
        labels = list(labels)
        chars = list(chars)
        vocabulary = Vocabulary(words_list=words, label_list=labels, chars_list=chars, Padding=True, Unkonwn=True)


        return train_x, train_y, val_x, val_y, test_x, test_y, vocabulary



    def CreateTorchDataset(self):
        class PytorchDataset(Dataset):
            def __init__(self, x, y, vocabulary):
                self.vocabulary = vocabulary
                self.word_token, self.char_token, self.y_token, self.lengths = self.convert(x, y)

            def __len__(self):
                return len(self.word_token)

            def __getitem__(self, item):  # item: should be integers and slice objects
                return self.word_token[item], self.char_token[item], self.y_token[item], self.lengths[item]

            def convert(self, x, y):
                w2idx = [[self.vocabulary.word2idx(w) for w in s] for s in x]
                la2idx = [[self.vocabulary.label2idx(la) for la in s] for s in y]
                length = [len(s) for s in x]

                char2idx = []
                for s in x:
                    seq_char = [[self.vocabulary.c2idx[c] for c in w] for w in s]
                    char2idx.append(seq_char)

                assert len(w2idx) == len(la2idx) == len(char2idx) == len(length)
                return w2idx, char2idx, la2idx, length


        return PytorchDataset(self.trn_x, self.trn_y, self.vocabulary), \
               PytorchDataset(self.val_x, self.val_y, self.vocabulary), PytorchDataset(self.test_x, self.test_y, self.vocabulary)



    def CreateDataLoader(self, data, batch_size=32,shuffle=False):

        char_max_len = self.MaxCharLen
        class collate_fn:
            # __call__ 使得类'实例对象'可以像调用普通函数那样，以“对象名()”的形式使用。
            def __call__(self, examples):
                word_x, char_x, ys, lengths = self.unpack(examples)
                Mask, sentence_max_len = self.GetMaskMartix(lengths)
                pad_word_x = self.SentencePadding(word_x, sentence_max_len)
                pad_char_x = self.CharPadding(char_x, sentence_max_len)
                ys = self.LabelPadding(ys, sentence_max_len)
                return torch.LongTensor(pad_word_x), torch.LongTensor(pad_char_x), \
                       torch.LongTensor(ys), torch.LongTensor(Mask)


            def GetMaskMartix(self, lengths):
                N = len(lengths)
                max_lens = max(lengths)
                Mask = np.zeros(shape=(N, max_lens))
                for i,l in enumerate(lengths):
                    Mask[i,:l] = np.array([1]*l)
                return Mask, max_lens

            def CharPadding(self, Xs, max_len):
                pad_x = np.zeros(shape=(len(Xs), max_len, char_max_len))
                for batch_i in range(len(Xs)):
                    for word_i, word in enumerate(Xs[batch_i]):
                        pad_x[batch_i, word_i, :len(word)] = word

                return pad_x

            def SentencePadding(self, Xs, max_len):
                pad_x = np.zeros(shape=(len(Xs),max_len))
                for i in range(len(Xs)):
                    data = Xs[i]
                    pad_x[i,:len(data)] = data

                return pad_x

            def LabelPadding(self, ys, max_len):
                N = len(ys)
                pad_y = np.full(shape=(N,max_len),fill_value=-1)
                for i in range(N):
                    la = ys[i]
                    pad_y[i,:len(la)] = la

                return pad_y


            def unpack(self, examples):
                word_x, char_x, ys, lengths = [], [], [], []
                for i in range(len(examples)):
                    word_x.append(examples[i][0])
                    char_x.append(examples[i][1])
                    ys.append(examples[i][2])
                    lengths.append(examples[i][3])


                return word_x, char_x, ys, lengths



        return DataLoader(dataset=data, batch_size=batch_size, drop_last=False, collate_fn=collate_fn(),
                            shuffle=shuffle, num_workers=0)


    def __check_file(self, path):
        assert os.path.isfile(path), IOError(f'{path} is not exists.')
        return True



if __name__ == '__main__':
    data = CoNLL_DataSet(sample=0.01)
    print(len(data.RawDatset['train']['x']), len(data.RawDatset['train']['y']))
    print(data.vocabulary.size)
    trn, val, test = data.CreateTorchDataset()
    print(trn[:1])
    trn_loader = data.CreateDataLoader(trn,batch_size=4)
    print(next(iter(trn_loader)))
