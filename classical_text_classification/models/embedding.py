import numpy as np
import torch
import torch.nn as nn



class WordEmbedding(nn.Module):
    def __init__(self, path, available_vocab):
        super(WordEmbedding, self).__init__()

        print('Use the ChineseTextClassifition pretrain-embedding from https://github.com/Embedding/Chinese-Word-Vectors')
        with open(path,encoding='utf8') as f:
            for i, text in enumerate(f):
                text = text.strip()
                if i == 0:
                    self.feature_size = int(text.split(' ')[-1])
                    pretrain_weight = np.random.random(size=(len(available_vocab),self.feature_size))
                else:
                    text_split = text.split(' ')
                    word = text_split[0]
                    idx = available_vocab.get(word, -1)
                    if idx != -1:
                        pretrain_weight[idx,:] = np.array(text_split[1:],dtype=np.float32)

        self.WordEmbed =nn.Embedding(num_embeddings=len(available_vocab),embedding_dim=self.feature_size,padding_idx=0)
        self.WordEmbed.weight = nn.Parameter(torch.tensor(pretrain_weight))

    @property
    def feature_dim(self):
        return self.feature_size

    def forward(self, x):
        return self.WordEmbed(x)




if __name__ == '__main__':
    # test code
    # model_cnn = TextCNN(vocab_size=30, feature_size=10, max_length=5, tasks=2)
    # x = torch.randint(low=0,high=29,size=(3,5))
    # print(x)
    # print(x.shape)
    # print(model_cnn(x))


    # model_rnn = TextRNN(vocab_size=30,feature_size=10, tasks=2)
    # x = torch.randint(low=0,high=30,size=(3,5))
    # print(x)
    # print(x.shape)
    # print(model_rnn(x))

    availabel_dict = {'<PAD>':0,'姚明':1,'政府':2}
    path = '../dataset/pretrain_cn_sgns.wiki.char'
    word_embed = WordEmbedding(path=path,available_vocab=availabel_dict)
    print(word_embed.feature_size)
    x = torch.randint(low=0,high=3,size=(4,5))
    print(x)
    print(x.shape)
    print(word_embed(x).shape)

