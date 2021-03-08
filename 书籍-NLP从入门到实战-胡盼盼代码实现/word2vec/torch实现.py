import numpy as np
import torch
from torch import nn, optim
import random
from collections import Counter
import matplotlib.pyplot as plt

text = "i like dog i like cat i like animal dog cat animal apple cat dog like dog fish milk like dog" \
       "cat eyes like i apple apple i hate apple i movie book music like cat dog hate cat dog like"

EMBEDDING_DIM = 2
PRINT_EVERY = 100
EPOCHS = 1000
BATCH_SIZE = 5
N_SAMPLES = 3
WINDOW_SIZE = 5
FREQ = 0
DELETE_WORDS = False

# 文本预处理
def preprocess(text,FREQ):
    text.lower()
    words = text.split()
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > FREQ]
    return trimmed_words

words = preprocess(text,FREQ)

vocab = set(words) # 对文本中存在的单词去重,set会自动排序
vocab2int = {w:c for c,w in enumerate(vocab)}  # 建立单词词典映射
int2vocab = {c:w for c,w in enumerate(vocab)}  # 建立单词词典映射
int_words = [vocab2int[word] for word in words]  # 将text文本转为字典中的index去表示

# 负采样,对整个语料库来操作的，并不是只针对某个文本或者文档, 下面是指语料库只有一个text的例子
int_word_count = Counter(int_words)
total_count = len(int_word_count)
word_freqs = {w: c / total_count for w,c in int_word_count.items() }

# 单词分布(用频次来表示分布)
# 方法1、负采样，抽样了
unigram_dist = np.array(list(word_freqs.values())) # 表示每一个单词以及这个单词在字典中的频次
noise_dist = torch.from_numpy(unigram_dist ** (0.75) / np.sum(unigram_dist**(0.75)))

# 方法2、高频词进行抽样处理,这里抽样了就是不要了的意思
if DELETE_WORDS:
    t = 1e-5
    prob_drop = {w:1-np.sqrt(t/word_freqs[w]) for w in int_word_count}
    train_words = [w for w in int_words if random.random() < (1-prob_drop[w])]
else:
    train_words = int_words


# 获取输入单词的窗口内的词(函数)
def get_target(words,idx,WINDOW_SIZE):
    target_window = np.random.randint(1,WINDOW_SIZE+1)
    start_point = idx - target_window if idx - target_window > 0 else 0
    end_point = idx + target_window if idx + target_window < len(words) else len(words) - 1
    # 中间的单词是输入，左右两边窗口里的单词是对应的label
    targets = set(words[start_point:idx] + words[idx:end_point+1])
    return list(targets)

# get batch处理，构建迭代器:
def get_batch(words,BATCH_SIZE,WINDOW_SIZE):
    n_batches = len(words) // BATCH_SIZE
    words = words[:n_batches * BATCH_SIZE]
    for idx in range(0,len(words),BATCH_SIZE):
        batch_x,batch_y = [],[]
        batch = words[idx : idx + BATCH_SIZE] # 一个batch里的所有单词进行遍历
        for j in range(len(batch)):
            x = batch[j]
            y = get_target(batch,j,WINDOW_SIZE)
            batch_x.extend([x] * len(y))
            batch_y.extend(y)

        yield batch_x,batch_y



# 基于负采样的skip-gram word2vec 模型
class SkipGramNeg(nn.Module):
    def __init__(self,n_vocab,n_embed,noise_dist):
        super(SkipGramNeg, self).__init__()
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist

        # 定义词向量层
        self.in_embed = nn.Embedding(n_vocab,n_embed) # 这是模型最后输出的字典, 每一次更新一个单词的参数,其实就是更新该单词的嵌入表示
        self.out_embed = nn.Embedding(n_vocab,n_embed)
        # 参数初始化
        self.in_embed.weight.data.uniform_(-1,1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self,input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors

    def forward_output(self,output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors

    def forward_noise(self,size,N_SAMPLES):
        noise_dist = self.noise_dist
        # 频次越高越容易被选作负样本,输出的是单词的下标（或者说值在noise_dist中的下标）
        # torch.multinomial，input要抽样对象，n_samples抽样次数，replacement是否有放回抽样
        noise_words = torch.multinomial(noise_dist,size*N_SAMPLES, replacement = True)
        # 好像noise_vectors不用管是哪个具体的单词，主要用于更新in_embed就行
        noise_vectors = self.out_embed(noise_words).view(size,N_SAMPLES,self.n_embed) # batch_size, num_word, features
        return noise_vectors

# 定义损失函数,同样要继承nn.Module,这里的loss是不用softmax，对每一pair，不论正样本还是负样本，都计算二分类的loss
class NegativeSampingLoss(nn.Module):
    def __init__(self):
        super(NegativeSampingLoss, self).__init__()
    def forward(self,input_vectors,output_vectors,noise_vectors):
        BATCH_SIZE, embed_size = input_vectors.shape
        input_vectors = input_vectors.view(BATCH_SIZE,embed_size,1)
        output_vectors = output_vectors.view(BATCH_SIZE,1,embed_size)
        # 目标词损失,torch.bmm是仅允许3维张量严格相乘，不允许广播
        out_loss = torch.bmm(output_vectors,input_vectors).sigmoid().log()
        out_loss =out_loss.squeeze() # 去掉dim = 1 的维度
        noise_loss = torch.bmm(noise_vectors.neg(),input_vectors).sigmoid().log()
        # 先去掉dim = 1的维度，然后再对N_samples的loss进行求和sum，keep_dim = False
        noise_loss = noise_loss.squeeze().sum(1)

        return  - (out_loss + noise_loss).mean()

if __name__ == '__main__':
    model = SkipGramNeg(len(vocab2int),EMBEDDING_DIM,noise_dist = noise_dist)
    critertion = NegativeSampingLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.005)
    steps = 0
    for e in range(EPOCHS):
        for input_words,target_words in get_batch(train_words,BATCH_SIZE,WINDOW_SIZE):
            steps += 1
            # input 和 targets都是word的索引，nn.embedding能够根据这个索引获取相应的词向量
            input,targets = torch.LongTensor(input_words),torch.LongTensor(target_words)
            input_vectors = model.forward_input(input)
            output_vectors = model.forward_output(targets)
            size, _ = input_vectors.shape
            noise_vectors = model.forward_noise(size,N_SAMPLES)

            # 计算损失
            loss = critertion(input_vectors,output_vectors,noise_vectors)
            # 打印损失：
            if steps % PRINT_EVERY == 0:
                print("loss:",loss)

            optimizer.zero_grad() # 上一次梯度零清
            loss.backward()  # 计算梯度
            optimizer.step() # 优化

    # 因为 dim = 2 ，所以可以直接在二维平面上展示
    for i,w in int2vocab.items():
        vectors = model.state_dict()["in_embed.weight"]
        x,y = float(vectors[i][0]),float(vectors[i][1])
        plt.scatter(x,y)
        plt.annotate(w,xy=(x,y),xytext=(5,2),textcoords="offset points",ha="right",va="bottom")

    plt.show()
