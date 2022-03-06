import random
import numpy as np
import collections
import re
import jieba

class DataSet:
    def __init__(self, path='../dataset/ChineseTextClassifition/cnews.train.txt', types='Bag-of-word',sample=0.1):
        self.token_sens, self.labels, self.vocab, self.labels_vocab = self.read_data(path, sample)
        self.data = self.process(types)

    def read_data(self, path, sample):
        stopwords = set()
        vocab = set()
        classes = []
        sens_token = []
        with open('../dataset/cn_stopwords.txt', encoding='utf8') as f1:
            stopword = f1.readlines()
            for w in stopword:
                w = w.strip()
                stopwords.add(w)

        with open(path, encoding='utf8') as f2:
            sentences = f2.readlines()
            print('Total numbers of data: {}'.format(len(sentences)))

            if sample != 1:
                random.shuffle(sentences)
                sentences = sentences[: int(len(sentences)*sample)]
                print(f'Sampling {sample*100}% data(total:{len(sentences)}) for saving time and testing')

            for sen in sentences:
                label, text = sen.strip().split('\t')
                classes.append(label)
                sen_token = [word for word in jieba.cut(text) if word not in stopwords]
                sens_token.append(sen_token)
                vocab = vocab | set(sen_token)


        vocab = { word: idx for idx, word in enumerate(vocab)} # Vocabulary, word2index
        class_types = {la: idx for idx,la in enumerate(set(classes))}  # class_strs -> Integer
        classes = [class_types[cla] for cla in classes] # class_strs -> Integer

        assert len(sens_token) == len(classes)
        return sens_token, classes, vocab, class_types


    def process(self, types):
        assert types in ['Bag-of-word','3-gram']
        print('Preparing data with {}.'.format(types))
        X_tokens = self.token_sens  # token_list for sentences
        y = self.labels

        # bag of word: tf-idf
        if types == 'Bag-of-word':
            X = self.__bow_process(sentences=X_tokens)

        # N-gram: default 3-gram
        else:
            raise ImportError('3-gram is not available.')


        # construct one-hot vector for label
        label_nums = len(self.labels_vocab)
        y_vec = []
        for cur in y:
            vec = [0 for _ in range(label_nums)]
            vec[cur] = 1
            y_vec.append(vec)
        return list(zip(np.array(X), np.array(y_vec)))


    def SplitData(self, frac=(0.7,0.1,0.2)):
        print("Split Rate: {}".format(frac))
        random.shuffle(self.data)

        n = len(self.data)
        trn_offset, val_offset = int(n*frac[0]), int(n*frac[1])

        trn_x, trn_y = self.__package_data(self.data[:trn_offset])
        val_x, val_y = self.__package_data(self.data[trn_offset: trn_offset+val_offset])
        test_x, text_y = self.__package_data(self.data[trn_offset+val_offset:])

        print('trn_shape: {}'.format(trn_x.shape))
        print('val_shape: {}'.format(val_x.shape))
        print('test_shape: {}'.format(test_x.shape))

        return trn_x, trn_y, val_x, val_y, test_x, text_y


    def batch_iter(self, x, y, batch_size=64):
        data_len = len(x)
        indices = np.random.permutation(np.arange(data_len))  # shuffle the data
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(data_len//batch_size):
            start_id = i * batch_size
            if i < data_len // batch_size - 1:
                end_id = (i+1) * batch_size
            else:
                end_id = data_len

            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


    def __bow_process(self, sentences):

        total_sentences = len(sentences) # for idf compute
        words2sens = collections.defaultdict(int)  # for idf compute
        for sen in sentences:
            words = set(sen)
            for w in words:
                words2sens[w] += 1

        Vec_X = []
        # sentence2vector: tf-idf
        for i, sen in enumerate(sentences):
            words2nums = collections.Counter(sen)
            length_sen = len(sen)
            vec_sens = [0 for _ in range(len(self.vocab))]
            for w, c in words2nums.items():
                idx = self.vocab[w]

                # tf * idf:
                tf_idf = (c / length_sen) * np.log(total_sentences/(words2sens[w] + 1))
                vec_sens[idx] = tf_idf

            Vec_X.append(vec_sens)

        return Vec_X


    def __ngram_process(self, sentences, gram=3):
        # construct the N-gram Vocabulary
        TotalWords = set()
        for sen in sentences:
            sen = re.sub(pattern='[,.]',repl='',string=sen)
            words = sen.split(' ')
            gram3_word = set()
            for i in range(len(words)-gram):
                gram3_word.add(''.join(words[i:i+gram]))

            if len(words) < gram:
                gram3_word.add(''.join(words[:]))
            TotalWords = TotalWords | gram3_word
        TotalWords = { w:idx for idx, w in enumerate(TotalWords)}
        print(TotalWords)
        Vec_X = []
        for sen in sentences:
            sen = re.sub(pattern='[,.]', repl='', string=sen)
            words = sen.split(' ')
            vec = [0 for _ in range(len(TotalWords))]
            gram3_word = collections.defaultdict(int)
            for i in range(len(words)-gram):
                gram3_word[''.join(words[i:i+gram])] += 1

            for w, c in gram3_word.items():
                w_idx = TotalWords[w]
                vec[w_idx] = c

            Vec_X.append(vec)

        return Vec_X


    def __package_data(self, data):
        X = []
        y = []
        for x, label in data:
            X.append(x)
            y.append(label)

        return np.array(X), np.array(y)



if __name__ == '__main__':
    # test code
    data = DataSet(sample=0.01)
    trn_x, trn_y, val_x, val_y, test_x, text_y = data.SplitData()
    t_x, t_y = next(iter(data.batch_iter(trn_x,trn_y,batch_size=4)))
    print(t_x,t_y)
    print(t_x.shape)