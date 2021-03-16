import torch
import numpy as np
import jieba
from torch.utils.data import DataLoader,Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_src_target(file_path):
    with open(file_path,'r',encoding='utf8') as file:
        raw_data = file.readlines()

    texts = [jieba.lcut(line[4:]) for line in raw_data]  # 对原文中句话进行中文分词
    src = texts[0::2]  # 偶数行为问句
    target = texts[1::2] # 奇数行为答句
    return texts,src,target


class Vocabolary:
    def __init__(self,file_path):
        self.texts,self.src,self.target = get_src_target(file_path)
        self.word2index,self.index2word = self.get_dict()

    def get_dict(self): # 简单统计单词，不作过滤低频单词的操作
        word_set = set()
        word_set.add('UNK')
        for text in self.texts:
            for word in text:
                word_set.add(word)

        word2index = {'PAD':0,'START':1} # PAD,用于对不定长batch进行填充，后面不常与计算loss
        index2word = {0:'PAD',1:'START'}
        word2index.update({w:i+2 for i,w in enumerate(word_set)})
        index2word.update({i+2:w for i,w in enumerate(word_set)})

        return word2index,index2word

    @property
    def vocab_size_(self):
        return len(self.word2index)

    def index(self,word):
        unk_index = self.word2index['UNK']
        return self.word2index.get(word,unk_index)

    def word(self,index):
        try:
            return self.index2word[index]
        except:
            print('index is out of dict_size!')


class MyDataset(Dataset):
    def __init__(self,vocab):
        super(MyDataset, self).__init__()
        self.vocab = vocab
        self.src = vocab.src  # input
        self.target = vocab.target # label

        self.src_index,self.target_index = self.get_word2index()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, item):
        # print(self.src[item],self.target[item])
        return {'x':self.src_index[item],'y':self.target_index[item]}


    def get_word2index(self):
        src,tgt = [],[]
        for sentence in self.src:
            src.append([self.vocab.index(word) for word in sentence])
        for sentence in self.target:
            tgt.append([self.vocab.index(word) for word in sentence])
        return src,tgt


class Collator:
    def __call__(self,examples):
        # examples : list[dict]
        self.src = self.unpack_(examples,'x')
        self.tgt = self.unpack_(examples,'y')
        src_maxlen = max([len(s) for s in self.src])
        tgt_maxlen = max([len(s) for s in self.tgt])

        # padding, 'PAD' : 0 index
        self.src_pad = np.zeros(shape=(len(examples),src_maxlen),dtype=np.int32)  # batch_size,src_max_seqlen
        self.tgt_pad = np.zeros(shape=(len(examples),tgt_maxlen),dtype=np.int32)  # batch_size,tgt_max_seqlen

        for index in range(len(examples)):
            self.src_pad[index,:len(self.src[index])] = self.src[index]
            self.tgt_pad[index,:len(self.tgt[index])] = self.tgt[index]

        return torch.LongTensor(self.src_pad).to(device),torch.LongTensor(self.tgt_pad).to(device)


    def unpack_(self, examples, name):
        return [e[name] for e in examples]


# for test
if __name__ == '__main__':
    file_path = './subtitle.txt'
    dataset = MyDataset(vocab=Vocabolary(file_path))
    dataloader = DataLoader(dataset,batch_size=4,shuffle=True,drop_last=False,collate_fn=Collator())
    dataloader.__iter__().next()
    print()

