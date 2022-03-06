import collections
import re
import pandas as pd
import numpy as np
import torch
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

class MyDataset:
    def __init__(self, path='../dataset/snli_1.0/snli_1.0_train.txt' ,sample=0.01, word_cutoff=5000):
        self.sen1, self.sen2, self.label = self.read_data(path, sample)
        self.sen1_token, self.sen2_token, self.labels, self.word2idx, self.label2idx = \
            self.process(self.sen1, self.sen2, self.label, word_cutoff)

    def read_data(self, path, sample):
        data = pd.read_csv(path, sep='\t').sample(frac=sample)
        sen1 = data['sentence1'].values
        sen2 = data['sentence2'].values
        label = data['gold_label'].values

        return sen1, sen2, label


    def process(self, sen1, sen2, label, word_cutoff):
        N = len(label)
        words_frequency = collections.defaultdict(int) # 保存token以及出现的频率

        stemmer = WordNetLemmatizer()
        stopword = set(stopwords.words("english"))

        wrongs_idx = set()
        def pre_sen(sentence, i):
            try:
                sentence = sentence.lower() #字母小写化
            except:
                print('some wrongs in sentence.lower(). Discarded')
                wrongs_idx.add(i)
                return []

            sentence = re.sub(r"[^a-zA-Z]",r" ", sentence) # 去除标点符号
            words = word_tokenize(sentence) # 分词

            words_final = []
            for w in words:
                if w not in stopword:
                    w = stemmer.lemmatize(w)
                    words_final.append(w)
                    words_frequency[w] += 1

            return words_final

        sen1_token = []
        sen2_token = []
        for i in range(N):
            sen1_token.append(pre_sen(sen1[i],i))
            sen2_token.append(pre_sen(sen2[i],i))

        words_common = [w[0] for w in sorted(words_frequency.items(),key=lambda d: d[1],reverse=True)][:word_cutoff]# 根据词表中单词出现频率进行截断
        words_common = ['<PAD>'] + words_common
        word2idx = {w:i for i,w in enumerate(words_common)}
        label2idx = {la:i for i, la in enumerate(set(label))}

        x1s = []
        x2s = []
        ys = []
        for i in range(N):
            if i not in wrongs_idx:
                s1 = [word2idx[w_token] for w_token in sen1_token[i] if w_token in word2idx]
                s2 = [word2idx[w_token] for w_token in sen2_token[i] if w_token in word2idx]

                # Discard the empty sentence, otherwise it will get Nan Values in training process.
                if len(s1) == 0 or len(s2) == 0:
                    print('No words in sentence. Discarded')
                    continue

                x1s.append(s1)
                x2s.append(s2)
                ys.append(label2idx[label[i]])


        assert len(x1s) == len(x2s) == len(ys)
        return x1s, x2s, ys, word2idx, label2idx



    def get_dataloader(self, data, batch_size, type):
        assert type in ['train','val','test']
        shuffle = (type == 'train')

        class collate_fn:
            # 使得类'实例对象'可以像调用普通函数那样，以“对象名()”的形式使用。
            def __call__(self, examples):
                p, h, p_len, h_len, label = self.__unpack(examples)

                N = len(p)
                max_len_p = max(p_len)
                max_len_h = max(h_len)

                mask_p = np.array([ [0]*pl + [1]*(max_len_p-pl) for pl in p_len])
                mask_h = np.array([ [0]*hl + [1]*(max_len_h-hl) for hl in h_len])

                p_pad = np.zeros(shape=(N, max_len_p))
                h_pad = np.zeros(shape=(N, max_len_h))
                for idx,(p_i, h_i) in enumerate(zip(p, h)):
                    p_pad[idx, :len(p_i)] = p_i
                    h_pad[idx, :len(h_i)] = h_i

                assert p_pad.shape == mask_p.shape and mask_h.shape == h_pad.shape

                return torch.LongTensor(p_pad), torch.LongTensor(h_pad), torch.tensor(label), \
                       torch.LongTensor(mask_p), torch.LongTensor(mask_h)


            def __unpack(self, examples):
                p = []
                h = []
                p_len = []
                h_len = []
                label = []
                for pi,hi,y in examples:
                    p.append(pi)
                    h.append(hi)
                    p_len.append(len(pi))
                    h_len.append(len(hi))
                    label.append(y)

                return p, h, p_len, h_len, label


        return DataLoader(dataset=data, collate_fn=collate_fn(),batch_size=batch_size,shuffle=shuffle,
                            drop_last=False, num_workers=0)


    def create_pydataset(self, splitrate=(0.7,0.1,0.2)):
        class pydataset(Dataset):
            def __init__(self, data, y):
                self.data = data
                self.y = y

            def __len__(self):
                return len(self.y)

            def __getitem__(self, item):
                return self.data[item][0], self.data[item][1], self.y[item]

        X = list(zip(self.sen1_token, self.sen2_token))
        data_x, test_x, data_y, test_y = train_test_split(X, self.labels, test_size=splitrate[2])
        trn_x, val_x, trn_y, val_y = train_test_split(data_x, data_y, test_size=len(test_y)//2)

        return pydataset(trn_x, trn_y), pydataset(val_x, val_y), pydataset(test_x, test_y)


if __name__ == '__main__':
    dataset = MyDataset(sample=0.01)
    x,y,z = dataset.create_pydataset()
    dataloader = dataset.get_dataloader(data=x, batch_size=4, type='val')
    print(next(iter(dataloader)))