import random
import torch.nn as nn
import torch
from dataset import *


'''
对于seq2seq with attention结构的解释，可具体参考大神的知乎文章：https://zhuanlan.zhihu.com/p/36131103
'''

# 利用decoder上一步的隐状态st-1, 与encoder所有时刻的output计算注意力分数。
# 然后加权求和得decoder这一时刻需要用到的context上下文向量(c)
class Attention(nn.Module):
    def __init__(self,enc_hid_size,dec_hid_size,engry_hid_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(enc_hid_size * 2 + dec_hid_size,engry_hid_size,bias=False)  # 将s0和enc_output concat后送进FC作project得到E
        self.tanh= nn.Tanh()
        self.v = nn.Linear(engry_hid_size,1,bias=False)
        self.softmax = nn.Softmax(-1)

    def forward(self,enc_output,s): # enc_output是最后一层所有时刻的hidden_state,s是最后一层最后时刻的hidden_state,两者组合计算每一个src时刻的attention score
        # enc_out: [seq_len,batch,enc_hid_dim * num_directions] s0: [1,batch,dec_hid_dim]
        enc_len = enc_output.shape[0]
        s = s.repeat(enc_len,1,1) # [seq_len,batch,dec_hid_dim], 为了对每一个时刻都进行concat,并行参与后面的attention计算.
        engry = self.tanh(self.attn(torch.cat([enc_output,s],dim=-1))) # engry : [seq_len,batch,enc_hid_dim*2 + dec_hid_dim] -> [seq_len,batch,engry_hid_size]
        attention = self.v(engry).squeeze(-1).transpose(0,1) # [seq_len,batch,1] -> [seq_len,batch] -> [batch,seq_len]
        return self.softmax(attention) # [batch,seq_len]


# 采用单层双向的GRU
# s0使用的是最后层最后时刻T的输出(正反向拼接向量)来表示
class Encoder(nn.Module):
    def __init__(self,vocab_size,embed_size,enc_hid_dim,dec_hid_dim,dropout=0.1):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.enc_embed = nn.Embedding(vocab_size, embed_size)
        self.BiGRU = nn.GRU(input_size=embed_size,hidden_size=enc_hid_dim,num_layers=1,bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim*2,dec_hid_dim)  # 为了后续 S_0 能作为decoder的初始隐藏状态输入，需要hidden的size转为dec_hid的size
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        # x: batch,seq_len -> x_embed: seq_len,batch,embed_size
        x_embed = self.enc_embed(x).transpose(0,1)
        # output: [seq_len,batch,enc_hid_dim * num_directions], 最后层每一个时刻的hidden-state
        # hidden: [num_layers * num_directions,batch,enc_hid_dim], 每一层最后时刻的hidden-state
        output,hidden = self.BiGRU(x_embed)
        s0 = self.dropout(self.fc(output[-1:,:,:])) # s0: [1,batch,enc_hidden * num_directions] -> s0: [1,batch,dec_hid_dim]
        return output, s0 # [seq_len,batch,enc_hid_dim * num_directions] 和 [1,batch,dec_hid_dim]


# 采用单向单层GRU
# Decoder接受三个输入, enc_out, st-1, embed(yt),由于GRU只能接受两个输入, enc_out和st-1计算上一个时刻的context ct-1
# 将ct-1与embed(yt)进行拼接操作,与隐状态st-1作为GRU的输入,得到这一时刻,同时也是下一时刻的隐状态st和输出h_ouput
class Decoder(nn.Module):
    def __init__(self,vocab_size,embed_size,enc_hid_dim,dec_hid_dim):
        super(Decoder, self).__init__()

        self.dec_embed = nn.Embedding(vocab_size,embed_size)
        self.attention = Attention(enc_hid_dim,dec_hid_dim,engry_hid_size=dec_hid_dim)
        self.GRU = nn.GRU(input_size=(enc_hid_dim * 2) + embed_size,hidden_size=dec_hid_dim,num_layers=1,bidirectional=False)
        # fc_out接受三个输入进行拼接, embed(yt),st,ct-1
        self.fc_out = nn.Linear(dec_hid_dim + embed_size + enc_hid_dim*2,vocab_size)
        self.softmax = nn.Softmax(-1)

    def forward(self,input,enc_out,s):
        '''
        :param input: 要decoder的输入序列 [batch,1]  # 每一次只输入一个时刻的，不能多个时刻一起算，因为后面的context受到这时刻的输出s影响
        :param s: decoder上一时刻的隐藏状态  [1,batch,dec_hid_dim]
        :param enc_out: encoder的所有时刻的最后层hidden_state: [seq_len,batch,enc_hid_dim * 2]
        '''
        input = input.unsqueeze(-1)
        dec_embedding = self.dec_embed(input).transpose(0,1) # 1,batch,embed_size   t
        attn_score = self.attention(enc_out,s).unsqueeze(1) # [batch,seq_len] -> [batch,1,seq_len]
        # 利用attention_score对enc_out的加权求和计算, bmm必须都是三维张量进行矩阵乘法，0维要一致，1和2维得满足矩阵乘法的要求.
        context = torch.bmm(attn_score,enc_out.transpose(0,1)).transpose(0,1) # [1, batch, enc_hid_dim * 2]  t-1

        # gru_input = dec_embedding + context (concat), 隐藏状态s0 = s, 两个都是batch_size = False(1维度)
        gru_input = torch.cat([dec_embedding,context],dim=-1)

        # dec_output_t: [len= 1,batch,dec_hid_dim*1],st: [num_layers*nums_directions=1,batch,dec_hid_dim]
        dec_output, s = self.GRU(gru_input,s)

        # for predict and classify
        # 作下一个单词预测,fc层需要接收这一时刻的s(单层也为dec_output),embed(y),上一时刻的c.
        fc_input = torch.cat([dec_embedding,dec_output,context],dim=-1).squeeze(0) # [batch, embed_size + enc_hid_dim*2 + dec_hid_dim]
        pred = self.fc_out(fc_input)

        return self.softmax(pred),s # [batch,vocab_size]、[num_layers*num_directions,batch,dec_hid_dim]




class Seq2Seq_attention(nn.Module):
    def __init__(self,vocab_size,embed_size,enc_hid_dim,dec_hid_dim,teacher_forcing_ratio=0.5):
        super(Seq2Seq_attention, self).__init__()

        self.vocab_size =vocab_size
        self.encoder = Encoder(vocab_size,embed_size,enc_hid_dim,dec_hid_dim)
        self.decoder = Decoder(vocab_size,embed_size,enc_hid_dim,dec_hid_dim)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self,src,target_in):
        # src:[batch_size,src_len],
        # target_in:[batch_size,tgt_len]  target_in是target的右移序列

        # tensor to store decoder outputs
        # target_in.shape[0]:batch_size  target_in.shape[1]: target_sequence_len
        out_puts = torch.zeros(size=(target_in.shape[0],target_in.shape[1],self.vocab_size))

        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output,s = self.encoder(src)

        dec_input = target_in[:,0]  # target_in是target的右移序列，即<START>作为t1
        for t in range(1,target_in.shape[1]):  # 每一次只输入一个时刻的，不能多个时刻一起算，因为后面的context受到这时刻的输出s影响
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_output, s = self.decoder(dec_input,enc_output,s) # [batch,vocab_size]、[decoder's num_layers*num_directions = 1,batch,dec_hid_dim]

            # place predictions in a tensor holding predictions for each token
            out_puts[:,t-1,:] = dec_output

            # decide if we are going to use teacher forcing or not,有一定概率下一刻的输入取这一刻的输出(不一定与target一致)，增强输出的丰富性
            teacher_force = random.random() < self.teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)

            # if teacher forcing, use actual next token as next input (actual target sequence)
            # if not, use predicted token
            dec_input = target_in[:,t] if teacher_force else top1 # 取下一个时刻的输入单词

        dec_output, s = self.decoder(dec_input, enc_output,s)
        out_puts[:, t - 1, :] = dec_output

        return out_puts  # [batch_size, target_seq_len, vocab_size]


