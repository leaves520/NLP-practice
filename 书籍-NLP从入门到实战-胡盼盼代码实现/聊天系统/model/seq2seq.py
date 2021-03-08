from torch import nn
import torch
import numpy as np

class Seq2Seq(nn.Module):
    def __init__(self, Deaultconfig, dict_len, encoder_input, decoder_input):
        # 这里处理不好，把训练数据写在初始化那边，没有用minibatch训练
        super(Seq2Seq, self).__init__()
        self.opts = Deaultconfig
        self.dict_len = dict_len
        _ , encoder_input_size = encoder_input.shape
        _ , decoder_input_size = decoder_input.shape
        self.encoder_input,self.decoder_input= torch.LongTensor(encoder_input), torch.LongTensor(decoder_input)

        # 编码器
        self.embed_encoder = nn.Embedding(dict_len,self.opts.w2v_size,)
        self.encoder = nn.LSTM(input_size=self.opts.w2v_size,hidden_size=self.opts.hidden_dim,batch_first=True) # batch_first = True，则input为(batch,seq,input_size)

        # 解码器
        self.embed_decoder = nn.Embedding(dict_len,self.opts.w2v_size)
        self.decoder = nn.LSTM(input_size=self.opts.w2v_size,hidden_size=self.opts.hidden_dim,batch_first=True)

        # 全连接层，维度为词典的长度，相当于多分类问题，类别大小为词语的个数
        self.nn_layer = nn.Linear(self.opts.hidden_dim,dict_len) # 对decoder lstm的输出再接全连接层作分类任务
        self.acitivation = nn.Softmax(dim=-1)

    def forward(self):
        output,(h,c) = self.encoder(self.embed_encoder(self.encoder_input)) # h\c: num_layers,batch,hidden_dim
        decoder_out, (_,_) = self.decoder(self.embed_decoder(self.decoder_input),(h,c)) # batch,seq_len,hidden_dim
        predict = self.acitivation(self.nn_layer(decoder_out)) # batch,seq_len,dict_len,  posibility of the words in dictionary

        return predict


class Seq2Seq_inference(nn.Module):
    def __init__(self, Deaultconfig, dict_len):
        super(Seq2Seq_inference, self).__init__()
        self.opts = Deaultconfig
        self.dict_len = dict_len

        # 编码器
        self.embed_encoder = nn.Embedding(dict_len,self.opts.w2v_size,)
        self.encoder = nn.LSTM(input_size=self.opts.w2v_size,hidden_size=self.opts.hidden_dim,batch_first=True) # batch_first = True，则input为(batch,seq,input_size),少这样设置

        # 解码器
        self.embed_decoder = nn.Embedding(dict_len,self.opts.w2v_size)
        self.decoder = nn.LSTM(input_size=self.opts.w2v_size,hidden_size=self.opts.hidden_dim,batch_first=True)

        # 全连接层，维度为词典的长度，相当于多分类问题，类别大小为词语的个数
        self.nn_layer = nn.Linear(self.opts.hidden_dim,dict_len) # 对decoder lstm的输出再接全连接层作分类任务
        self.acitivation = nn.Softmax(dim=-1)

    # 利用训练好的模型，对于输入的任意一句话，预测出相应的输出。n_steps=80,表示预测句子最长为80，且遇到\n则终止
    def forward(self,source,dict_word2index,dict_index2word):
        text = self.seq2index(source,dict_word2index) # 将中文句子变为index list来表示，后面用嵌入矩阵进行索引
        text = self.embed_encoder(text).unsqueeze(0)
        _,(h,c) = self.encoder(text) # h\c: num_layers,batch,hidden_dim 即获取encoder后的状态state h|c

        # decoder的初始输入字符为'\t'
        target_seq = torch.zeros((1,1))
        target_seq[0,0] = dict_word2index['\t']
        target_seq = target_seq.long()
        output = ''

        # 通过encoder得到的state作为decoder的初始状态输入，
        # 解码过程中，每次利用上次预测的词作为输入来预测下一次的词，直到预测出终止符'\n'
        for i in range(self.opts.n_steps):
            # 每一次输出单词以及隐状态
            target_seq = self.embed_decoder(target_seq)
            y,(h,c) = self.decoder(target_seq,(h,c))
            word_index = self.acitivation(self.nn_layer(y)).argmax(-1)[0][0] # 获取最大可能的词
            word = dict_index2word[word_index.item()] # tensor.item() 获取tensor的值
            if word =='\n':
                break
            output = output + '' + word

            # 更新下一步要输入的词，使用这一步的输出词
            target_seq = torch.zeros((1,1))
            target_seq[0,0] = dict_word2index[word]
            target_seq = target_seq.long()

        return output


    @staticmethod
    def seq2index(text,dict_word_index):
        '''
        :param text: 中文语句
        :param dict_word_index: 词与标号对应的词典
        :return:  转为index代表的语句
        '''
        # 实际是用于将句子转为index list所表示
        # 将输入文字转为字典中对应的标号，找不到就标记为0
        return torch.LongTensor([dict_word_index.get(word,0) for word in text])



# 损失函数
class Critertion(nn.Module):
    def __init__(self):
        super(Critertion, self).__init__()
        self.loss = nn.CrossEntropyLoss() # input,`(N, C)` where `C = number of classes`; output, (N)` where each value is true label

    def forward(self,predict,decoder_output):
        decoder_output = torch.LongTensor(decoder_output)
        loss = 0.0
        for i in range(predict.shape[0]):
            loss += self.loss(predict[i],decoder_output[i].argmax(-1))

        return loss






