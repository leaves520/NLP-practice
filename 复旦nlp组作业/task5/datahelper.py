import numpy as np
from sklearn.model_selection import train_test_split
import re
from torch.utils.data import Dataset, DataLoader
import torch



class Vocabulary(object):
    def __init__(self, words_list, Padding=True, Unkonwn=True, start_end=True):
        if Padding:
            words_list = ['<PAD>'] + words_list + ['<START>', '<END>']
        if Unkonwn:
            words_list = words_list + ['<UNK>']
            self.unk_idx_word = len(words_list) - 1

        self.w2idx = {w:i for i, w in enumerate(words_list)}
        self.idx2w = {i:w for i, w in enumerate(words_list)}


    def word2idx(self, word):
        idx = self.w2idx.get(word, self.unk_idx_word)
        assert self.unk_idx_word is not None, ValueError('Word is not in vocab. But also no UNK. ')
        return idx

    def idx2word(self, idx):
        assert 0 <= idx < len(self.idx2w), IndexError()
        return self.idx2w[idx]


    @property
    def property(self):
        return {'vocab_size':len(self.w2idx), '<PAD>':self.word2idx('<PAD>'),
                '<START>': self.word2idx('<START>'),'<END>': self.word2idx('<END>'),'<UNK>': self.word2idx('<UNK>')}



class PoetryTang(object):
    def __init__(self, path='../dataset/ChineseTextGeneration/poetryFromTang.txt', sample=0.2, SplitRate=(0.8,0.1,0.1)):
        self.RawData = self.read(path)
        self.train, self.val, self.test, self.vocabulary = self.process(self.RawData, sample, SplitRate)


    def read(self, path):
        data = []
        with open(path, mode='r', encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                data.append(line)

        return data


    def process(self, data, sample, SplitRate):

        data_clean = []
        words = set()
        for sen in data:
            sen = re.sub(pattern='[a-zA-Z]',repl='', string=sen)
            tmp = []
            for w in sen:
                if w not in ['，', '。']:
                    words.add(w)
                    tmp.append(w)
            data_clean.append(tmp)

        print('Total Numbers: {}'.format(len(data_clean)))

        if sample != 1:
            cutoff = int(len(data_clean) * sample)
            data_clean = data_clean[:cutoff]
            print('Sample some Training data ({}|{}) for testing code.'.format(sample, cutoff))

        data_x, test = train_test_split(data_clean, test_size=SplitRate[-1])
        trn, val = train_test_split(data_x, test_size=len(test))

        words = list(words)
        vocabulary = Vocabulary(words_list=words)

        return trn, val, test, vocabulary



    def CreateTorchDataset(self):
        class PytorchDataset(Dataset):
            def __init__(self, x, vocabulary):
                self.vocabulary = vocabulary
                self.x_token, self.lengths = self.convert(x)

            def __len__(self):
                return len(self.x_token)

            def __getitem__(self, item):  # item: should be integers and slice objects
                return self.x_token[item], self.lengths[item]

            def convert(self, x):
                Xs, lengths = [], []
                for i in range(len(x)):
                    x_token = [ self.vocabulary.word2idx(w) for w in x[i]]
                    x_token = [self.vocabulary.word2idx('<START>')] + x_token + [self.vocabulary.word2idx('<END>')]
                    lengths.append(len(x_token))
                    Xs.append(x_token)

                return Xs, lengths

        return PytorchDataset(self.train, self.vocabulary), \
               PytorchDataset(self.val, self.vocabulary), PytorchDataset(self.test,self.vocabulary)


    def CreateDataLoader(self, data, batch_size=32,shuffle=False):

        padding_idx = self.vocabulary.word2idx('<PAD>')
        class collate_fn:
            def __call__(self, examples):
                x, y, lengths = self.unpack(examples)
                Mask, sentence_max_len = self.GetMaskMartix(lengths)
                x = self.pad_sequences(x, maxlen=sentence_max_len, value=padding_idx,
                                       padding='post', truncating='post')
                y = self.pad_sequences(y, maxlen=sentence_max_len, value=padding_idx,
                                       padding='post', truncating='post')
                return torch.LongTensor(x), torch.LongTensor(y), torch.LongTensor(Mask)


            def GetMaskMartix(self, lengths):
                N = len(lengths)
                max_lens = max(lengths)
                Mask = np.zeros(shape=(N, max_lens))
                for i,l in enumerate(lengths):
                    Mask[i,:l] = np.array([1]*l)
                return Mask, max_lens



            def pad_sequences(self, sequences,maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
                """
                code from keras
                Pads each sequence to the same length (length of the longest sequence).
                If maxlen is provided, any sequence longer
                than maxlen is truncated to maxlen.
                Truncation happens off either the beginning (default) or
                the end of the sequence.
                Supports post-padding and pre-padding (default).
                Arguments:
                    sequences: list of lists where each element is a sequence
                    maxlen: int, maximum length
                    dtype: type to cast the resulting sequence.
                    padding: 'pre' or 'post', pad either before or after each sequence.
                    truncating: 'pre' or 'post', remove values from sequences larger than
                        maxlen either in the beginning or in the end of the sequence
                    value: float, value to pad the sequences to the desired value.
                Returns:
                    x: numpy array with dimensions (number_of_sequences, maxlen)
                Raises:
                    ValueError: in case of invalid values for `truncating` or `padding`,
                        or in case of invalid shape for a `sequences` entry.
                """
                if not hasattr(sequences, '__len__'):
                    raise ValueError('`sequences` must be iterable.')
                lengths = []
                for x in sequences:
                    if not hasattr(x, '__len__'):
                        raise ValueError('`sequences` must be a list of iterables. '
                                         'Found non-iterable: ' + str(x))
                    lengths.append(len(x))

                num_samples = len(sequences)
                if maxlen is None:
                    maxlen = np.max(lengths)

                # take the sample shape from the first non empty sequence
                # checking for consistency in the main loop below.
                sample_shape = tuple()
                for s in sequences:
                    if len(s) > 0:  # pylint: disable=g-explicit-length-test
                        sample_shape = np.asarray(s).shape[1:]
                        break

                x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
                for idx, s in enumerate(sequences):
                    if not len(s):  # pylint: disable=g-explicit-length-test
                        continue  # empty list/array was found
                    if truncating == 'pre':
                        trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
                    elif truncating == 'post':
                        trunc = s[:maxlen]
                    else:
                        raise ValueError('Truncating type "%s" not understood' % truncating)

                    # check `trunc` has expected shape
                    trunc = np.asarray(trunc, dtype=dtype)
                    if trunc.shape[1:] != sample_shape:
                        raise ValueError(
                            'Shape of sample %s of sequence at position %s is different from '
                            'expected shape %s'
                            % (trunc.shape[1:], idx, sample_shape))

                    if padding == 'post':
                        x[idx, :len(trunc)] = trunc
                    elif padding == 'pre':
                        x[idx, -len(trunc):] = trunc
                    else:
                        raise ValueError('Padding type "%s" not understood' % padding)

                return x



            def unpack(self, examples):
                '''
                Note: input x is the Right-Shift of the y(target)
                      that is input = x[:-1], begin with start_idx, no end_idx
                              target = x[1:] begin with first truth word, the last word is end_idx
                '''
                x, y, lengths = [], [], []
                for i in range(len(examples)):
                    x.append(examples[i][0][:-1])
                    y.append(examples[i][0][1:])
                    # -1: the raw_x is begin with start_idex and end with end_idx， But remove some one both in x and y.
                    lengths.append(examples[i][1]-1)

                return x, y, lengths



        return DataLoader(dataset=data, batch_size=batch_size, drop_last=False, collate_fn=collate_fn(),
                            shuffle=shuffle, num_workers=0)




if __name__ == '__main__':
    # test code
    # import random
    # t = [[random.randint(1,10) for _ in range(l)] for l in range(4,0,-1)]
    # print(t)
    # pad_t = pad_sequences(t, maxlen=None, padding='post', truncating='post',)
    # print(pad_t)

    data = PoetryTang()
    print(data.vocabulary.property)
    print(data.vocabulary.w2idx.keys())
    trn, val, test = data.CreateTorchDataset()
    print(trn[:2])
    trn_loader = data.CreateDataLoader(trn, batch_size=2, shuffle=False)
    print(next(iter(trn_loader)))

    print(torch.pow(torch.tensor(4),1/2))
