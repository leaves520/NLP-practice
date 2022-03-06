import numpy as np
import torch
import torch.nn as nn

class TextCNN(nn.Module):
    '''
    Ref: https://www.cnblogs.com/ModifyRong/p/11319301.html
         https://arxiv.org/abs/1408.5882
    '''

    def __init__(self, vocab_size, feature_size, max_length, tasks, filter_windows=(3,4,5), feature_maps=100, word_embedding=None):
        super(TextCNN, self).__init__()

        self.filter_nums = len(filter_windows)

        if word_embedding is None:
            self.wordembed = nn.Embedding(num_embeddings=vocab_size,embedding_dim=feature_size,padding_idx=0)
        else:
            assert isinstance(word_embedding,nn.Module)
            self.wordembed = word_embedding

        self.ConvModule = nn.ModuleList()
        self.MaxPools = nn.ModuleList()
        for idx, filter_size in enumerate(filter_windows):
            # out_channels is also called the FeatureMaps.
            self.ConvModule.append(module=nn.Conv2d(in_channels=1,out_channels=feature_maps,
                                                        kernel_size=(filter_size,feature_size), stride=(1,1)))
            self.MaxPools.append(module=nn.MaxPool2d(kernel_size=(max_length-filter_size+1,1)))  # (n - f + 2p) / stride +1

        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        # we use max-pooling to get one-scalar for each featureMap.
        # Actually, we will got (filter_nums*feature_maps) scalar after Convolution.
        self.linear_project = nn.Linear(self.filter_nums*feature_maps, tasks)

    def forward(self, x):
        # x : batch_size * max_len
        x_embed = self.wordembed(x) # batch_size * max_len * feature_size
        x_embed = x_embed.unsqueeze(1) # batch_size * channel * max_len * feature_size

        feature_maps = []
        for i in range(self.filter_nums):
            conv = self.ConvModule[i]
            max_pool = self.MaxPools[i]

            fea_map = max_pool(self.dropout(self.relu(conv(x_embed)))).squeeze()
            feature_maps.append(fea_map)

        concate_x = torch.cat(feature_maps, dim=-1)

        return self.linear_project(concate_x)



class TextRNN(nn.Module):
    '''
    Dropout: Dropout having the desired impact on training with a slightly slower trend in convergence and in this case a lower final accuracy.
    The model could probably use a few more epochs of training and may achieve a higher skill
    '''

    def __init__(self, vocab_size, feature_size, tasks,  word_embedding=None):
        super(TextRNN, self).__init__()

        if word_embedding is None:
            self.wordembed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=feature_size,padding_idx=0)
        else:
            assert isinstance(word_embedding, nn.Module)
            self.wordembed = word_embedding

        self.relu = nn.ReLU()
        self.dropout =  nn.Dropout(0.2)
        self.rnn = nn.LSTM(input_size=feature_size, hidden_size=feature_size, num_layers=2,
                           batch_first=True)  # batch_first = True, input_shape: batch_size * sen_len * feature_size

        self.linear_project = nn.Linear(feature_size, tasks)

    def forward(self, x):
        x_embed = self.wordembed(x)  # batch_size * sen_len * feature_size
        x_embed = self.dropout(self.relu(x_embed))

        output, (h0, c0) = self.rnn(x_embed)
        last_word_embed = output[:,-1,:].squeeze(1)  # batch_size * feature_size

        return self.linear_project(last_word_embed)



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


