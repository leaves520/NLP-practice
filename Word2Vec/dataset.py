import numpy as np
import torch
from collections import Counter
from torch.utils.data import DataLoader,Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
Word_Frequency = 3  # 设定语料库中单词出现的阈值,大于阈值则保留
Max_DictSize = 10000  # 设定保留的最大字典大小, 不包括UNK
Nums_Neg = 5  # 负采样的样本个数
Windows_Size = 3 # 滑窗大小: 以中心词为Centre, 左右所包括的单词数量。如：win_size=1, i love you, love为中心，左边所包括单词为i,右边为you

class Vocabolary():
    def __init__(self, text):
        # load and reprocess dataset 'text8.test.txt'
        self.text = text
        self.word2index, self.index2word, self.word_count = self.process()

        # 将单词出现频次转为频率
        total_count = np.array(self.word_count,dtype=np.float32).sum() # 所有单词出现的总频次
        self.word_fre = np.array(self.word_count,dtype=np.float32) / total_count # 单词的频率等于单词的出现频次除以所有单词出现的总频次
        self.word_fre = self.word_fre ** (3.0/4.0)  # 论文中频率转为原来的0.75次方


    def process(self):
        word_counts = dict(Counter(self.text))
        word_counts = [(key, value) for key, value in word_counts.items() if value > Word_Frequency] # 去除低频词
        word_counts = word_counts[:Max_DictSize-1]  # 只保留Max_DictSize-1字典大小的单词及其出现的频次

        word2index = {w: i for i, (w, _) in enumerate(word_counts)}
        index2word = {i: w for i, (w, _) in enumerate(word_counts)}
        word_count = [count for _,count in word_counts]

        word2index.update({"UNK":Max_DictSize})
        index2word.update({Max_DictSize:"UNK"})
        return word2index, index2word, word_count

    @property
    def vocab_size_(self):
        return len(self.word2index)

    def index(self,word):
        assert isinstance(word,str)
        return self.word2index.get(word,Max_DictSize-1) # 根据单词返回在字典中的index，不存在则返回UNK所在的index

    def word(self,index):
        assert isinstance(index,int) and index >= 0 and index <=Max_DictSize
        return self.index2word[index]


def get_Vocab_text(file_path,test = False):
    with open(file_path, 'r') as file:
        text = file.read()

    # for debug
    if test:
        Word_Frequency = 0
        text = text[:20]

    text = text.lower().split()  # 转换小写并按空格分割单词, 文本转为用words list表示
    # nltk.download('stopwords')
    # from nltk.corpus import stopwords  # 加载停用词，如 the, this, am,
    # text = [word for word in text if word not in stopwords.words("english")] # 对文本去除停用词,比较耗时,
    vocab = Vocabolary(text)

    # 将文本的words list -> 文本的index list
    index_text = [vocab.index(word) for word in text]
    return index_text,vocab


class MyDataset(Dataset):
    def __init__(self,train_text,vocab):
        super(MyDataset, self).__init__()

        self.text_encoded = train_text
        self.word_freq = torch.Tensor(vocab.word_fre)  # 用于后续负采样，频次越大，被抽中的作为负样本概率越大(如某些语气词出现多次，可以作为负样本). 一个pos_targets就采样Nums_Neg个负样本, 正负样本比例1:Nums_Neg

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        input = [self.text_encoded[idx]]
        pos_targets = self.get_target(idx)
        neg_targets = torch.multinomial(self.word_freq,num_samples=len(pos_targets) * Nums_Neg ,replacement = True) # torch.multinomial，input要抽样对象(表示权重)，n_samples抽样次数，replacement是否有放回抽样

        # 保证负采样不出现在pos_targets中:
        while len(set(pos_targets) & set(neg_targets)) > 0:
            neg_targets = torch.multinomial(self.word_freq,num_samples=len(pos_targets) * Nums_Neg ,replacement = True)

        input,pos_targets,neg_targets = torch.LongTensor(input),torch.LongTensor(pos_targets),torch.LongTensor(neg_targets)
        assert len(pos_targets) == len(neg_targets) // Nums_Neg  # 正负样本比例1:Nums_Neg
        return input.to(device),pos_targets.to(device),neg_targets.to(device)


    def get_target(self,idx): # idx,中心词的下标,获取正训练样本
        pos_index = list(range(idx-Windows_Size,idx)) + list(range(idx+1,idx+1+Windows_Size)) # 获取左右窗口大小Windows_Size的单词
        pos_index = [ i % len(self.text_encoded) for i in pos_index] # 防止如起点，终点等出现数组越界问题
        return [self.text_encoded[pi] for pi in pos_index]



# for debug
if __name__ == '__main__':
    index_text,vocab = get_Vocab_text('./text8.test.txt',test=True)
    test = MyDataset(index_text,vocab)
    print(test[0])





