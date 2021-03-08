import torch.nn as nn
import torch
import numpy as np
import math

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_q = d_k = d_v = 64  # Embedding Size of K(=Q), V
n_layers = 6  # number of Encoder and Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention


class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout=0.1,max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self,x):
        '''
        :param x: [seq_len,batch_size,d_model]
        '''
        x = x + self.pe[:x.size(0),:]
        return self.dropout(x)


def get_attn_pad_mask(seq_q,seq_k):  # 用于mask Q * K^T矩阵
    '''
    :param seq_q: [batch_size,seq_len]
    :param seq_k: [batch_size,seq_len]
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size,len_q = seq_q.shape
    batch_size,len_k = seq_k.shape
    # eq(0) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size,1,len_k] 返回一个大小和seq_k一样的bool tensor,再加上一个维度

    return pad_attn_mask.expand(batch_size,len_q,len_k) # batch_size,len_q,len_k
    # expand len_q, 后续用于mask Q * K^T 是表示词与词之间的attention score矩阵. 理论上对行维度的也要mask掉, 但是实际影响不大


def get_attn_subsequence_mask(seq):  # only decoder used
    '''
    :param seq: [batch_size,tgt_len]
    '''

    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), # torch.triu 生成上三角矩阵
                      diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, tgt_len, tgt_len], batch中每一个句子都是相同的mask矩阵,所以expand,都是t时刻只能利用t-1及之前的信息，不能利用t+1后的信息
    return mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(-1)

    def forward(self,Q,K,V,attention_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        score = torch.matmul(Q, K.transpose(-1,-2))/ np.sqrt(d_k)  # batch_size,nums_head,len_q,len_k, 表示词与词之间的attention score矩阵
        score.masked_fill_(attention_mask,-1e9) # attten_mask is bool tensor. 类似np.where,true的位置就填负无穷,softmax后值为0,不参与后面的加权求和
        attn_score = self.softmax(score)
        context = torch.matmul(attn_score,V) # batch_size,nums_head,seq_len,d_v
        return context,attn_score


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model,d_q * n_heads, bias=False)  # 这里多头矩阵都用同一个线性变换得到
        self.W_K = nn.Linear(d_model,d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model,d_v * n_heads, bias=False)
        self.fc = nn.Linear(d_v * n_heads,d_model,bias=False)
        self.scaled_dot = ScaledDotProductAttention()
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self,input_Q,input_K,input_V,attention_mask):
        '''
        :param input_Q: batch_size, seq_len, d_model
        :param input_K:
        :param input_V:
        :param attention_mask: batch_size,seq_len,seq_len
        '''
        residual,batch_size = input_Q, input_Q.shape[0] # 残差使用的是input_Q
        Q = self.W_Q(input_Q).view(batch_size,n_heads,-1,d_q) # Q: [batch_size, n_heads, len_q, d_q]
        K = self.W_K(input_K).view(batch_size,n_heads,-1,d_k) # K: [batch_size, n_heads, len_q, d_k]
        V = self.W_K(input_V).view(batch_size,n_heads,-1,d_v) # V: [batch_size, n_heads, len_q, d_v]

        atten_mask = attention_mask.unsqueeze(1).repeat(1,n_heads,1,1) # 扩展到多头矩阵

        context, attn = self.scaled_dot(Q,K,V,atten_mask) # context: [batch_size,nums_head,seq_len,d_v]
        context = context.transpose(1,2).reshape(batch_size,-1,n_heads * d_v) # concat操作,[batch_size,seq_len,,d_v]

        output = self.fc(context)
        return self.layernorm(output + residual) , attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model,d_ff,bias=False),
            nn.ReLU(),
            nn.Linear(d_ff,d_model,bias=False)
        )
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self,inputs):
        # inputs: [batch_size,seq_len,d_model]
        residual = inputs
        output = self.fc(inputs)
        return self.layerNorm(output+residual) # [batch_size,seq_len,d_model]


class EncoderLayer(nn.Module):  # 拼接组件，得到完整的Encoder layer
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self,enc_inputs,enc_self_atten_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        enc_outputs,attn = self.enc_self_attn(enc_inputs,enc_inputs,enc_inputs,enc_self_atten_mask)
        enc_outputs = self.pos_ffn(enc_outputs) # batch_size,src_len,d_model

        return enc_outputs,attn


class Encoder(nn.Module):
    def __init__(self,vocab_size):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size,d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self,enc_inputs):
        # batch_size,src_len
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_atten_mask = get_attn_pad_mask(enc_inputs,enc_inputs) # [batch_size,src_len,src_len]. 根据每一个句子的dict_index(int)序列得到相应的mask矩阵,用于对Q*K^T的mask
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs,enc_self_attn = layer(enc_outputs,enc_self_atten_mask)  # 每一层都是用同样的mask矩阵
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module): # 调用两次多头注意力
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_atten = MultiHeadAttention() # 输入进行一次自注意力，获取相应的dec_outputs
        self.dec_enc_atten = MultiHeadAttention() # 和encoder的输出(作为K,V生成的元素), dec_outputs(作为Q的生成元素)再调用自注意力
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self,dec_inputs,enc_outputs,dec_self_attn_mask,dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        dec_outputs,dec_self_attn = self.dec_self_atten(dec_inputs,dec_inputs,dec_inputs,dec_self_attn_mask)
        dec_outputs,dec_enc_attn = self.dec_enc_atten(dec_outputs,enc_outputs,enc_outputs,dec_enc_attn_mask) # 对应用于(Q,K,V)生成的元素
        dec_outputs = self.pos_ffn(dec_outputs) #[batch_size,tgt_len,d_model]
        return dec_outputs,dec_self_attn,dec_enc_attn


class Decoder(nn.Module):
    def __init__(self,vocab_size):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self,dec_inputs,enc_inputs,enc_outputs):
        '''
        :param dec_inputs: [batch_size,tgt_len] - index_sequence for target_inputs
        :param enc_inputs: [batch_size,src_len] - index_sequence for enc_inputs
        :param enc_outputs: [batch_size,src_len,d_model] - outputs for the last layers of Encoder
        '''
        dec_outputs = self.tgt_emb(dec_inputs) # batch_size,tgt_len,d_model
        dec_outputs = self.pos_emb(dec_outputs.transpose(0,1)).transpose(0,1) # add_position embedding

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs,dec_inputs) # index_sequence to get pad mask matrix
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs) # index_sequence to get sequence_mask
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),0) # pad and sequence mask
        # torch.gt(a, value) 将 a 中各个位置上的元素和 value 比较，若大于 value，则该位置取 1，否则取 0

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs,enc_inputs) # batch_size,tgt_len,src_len
        # 因为decoder每个词的解码要用到encoder中每个词的embedding信息. dec_enc_attn_mask,用于mask掉enc_input与target_input的pad

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model],
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len],
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self,vocab_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(vocab_size)
        self.decoder = Decoder(vocab_size)
        self.projection = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs): # encoder-decoder框架，接收两个输入.
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns




