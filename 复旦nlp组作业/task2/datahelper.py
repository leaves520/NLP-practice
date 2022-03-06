import collections
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import jieba
import random
import warnings
warnings.filterwarnings("ignore")

class Dataset:
    def __init__(self, data_root='../dataset/', sample=0.1, word_cutoff=5000):
        self.sens_token, self.labels, self.word2idx, self.idx2word, self.label2idx \
            = self.read_data(data_root + 'cnews.train.txt', sample=sample, word_cutoff=word_cutoff)


    def read_data(self, path, sample, word_cutoff=5000):
        stopwords = set()
        vocab = set()
        classes = []
        sens_token = []
        vocab_frequency = collections.defaultdict(int)
        with open('../dataset/cn_stopwords.txt', encoding='utf8') as f1:
            stops = f1.readlines()
            for w in stops:
                w = w.strip()
                stopwords.add(w)
            stopwords.add(' ')
            stopwords.add('\xa0')

        with open(path, encoding='utf8') as f2:
            sentences = f2.readlines()
            print('Total numbers of data: {}'.format(len(sentences)))

            if sample != 1:
                random.shuffle(sentences)
                sentences = sentences[: int(len(sentences) * sample)]
                print(f'Sampling {sample * 100}% data(total:{len(sentences)}) for saving time and testing')

            for sen in sentences:
                label, text = sen.strip().split('\t')
                classes.append(label)
                sen_token = []
                for word in jieba.cut(text):
                    if word not in stopwords:
                        sen_token.append(word)
                        vocab_frequency[word] += 1
                sens_token.append(sen_token)
                vocab = vocab | set(sen_token)

        vocab_most = sorted(vocab_frequency.items(),key=lambda x: x[1],reverse=True)[:word_cutoff]
        vocab = ['<PAD>'] + [k for k, _ in vocab_most]  # '<PAD>' for batch padding
        word2index = {word: idx for idx, word in enumerate(vocab)}  # Vocabulary, word2index
        index2word = {idx: word for idx, word in enumerate(vocab)}
        label_vocab = {la: idx for idx, la in enumerate(set(classes))}  # class_strs -> Integer

        assert len(sens_token) == len(classes)
        return sens_token, classes, word2index, index2word, label_vocab


    def splitdata(self, frac=(0.7, 0.1, 0.2)):
        X, y = self.sens_token, self.labels
        X_data, X_test, y_data, y_test = train_test_split(X, y, test_size=frac[-1])
        X_trn, X_val, y_trn, y_val = train_test_split(X_data, y_data, test_size=len(y_test)//2)

        return X_trn, y_trn, X_val, y_val, X_test, y_test , self.word2idx, self.idx2word, self.label2idx


    def batch_iter(self, x, y, batch_size=64, max_length=300):
        assert len(x) == len(y)
        x = np.array(x)
        y = np.array(y)
        N = len(x)
        indices = np.random.permutation(np.arange(N))
        x_shuffle = x[indices]
        y_shuffle = y[indices]
        for i in range(N//batch_size):
            start_id = i*batch_size
            if i < N // batch_size - 1:
                end_id = (i+1)*batch_size
            else:
                end_id = N

            yield self.__convert(x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], max_length)


    def __convert(self, X, y, max_length):
        y_new = []  # In pytorch case, we don't need to transformer the Integer label to one-hot Vector
        for label in y:
            y_new.append(self.label2idx[label])

        X_new = []
        for x in X:
            x_vec = [self.word2idx[word] for word in x if word in self.word2idx]
            if len(x_vec) > max_length:  # max_length for sentences cutoff
                x_vec = x_vec[:max_length]
            else:
                x_vec.extend([self.word2idx['<PAD>']] * (max_length - len(x_vec)))

            X_new.append(x_vec)

        return torch.LongTensor(X_new), torch.LongTensor(np.array(y_new))




if __name__ == '__main__':
    dataset = Dataset(sample=0.01)
    X_trn, y_trn, X_val, y_val, X_test, y_test , word2idx, idx2word = dataset.splitdata()
    print(X_trn[0])
    print(next(iter(dataset.batch_iter(X_trn,y_trn,batch_size=4)))[0])
    print(next(iter(dataset.batch_iter(X_trn,y_trn,batch_size=4)))[0].shape)
    print(next(iter(dataset.batch_iter(X_trn,y_trn,batch_size=4)))[1].shape)




