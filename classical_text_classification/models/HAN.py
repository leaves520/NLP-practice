import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.max_word_num = 15  # 15 句子所含最大词数
        self.max_sents_num = 20  # 60 文
        self.hidden_size = config.hidden_size


        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)



        self.dropout0 = torch.nn.Dropout(config.dropout)

        self.bi_rnn1 = torch.nn.GRU(config.embed, config.hidden_size, bidirectional=True, batch_first=True, dropout=0.2)
        self.word_attn = torch.nn.Linear(config.hidden_size * 2, self.hidden_size)
        self.word_ctx = torch.nn.Linear(config.hidden_size, 1, bias=False)

        self.bi_rnn2 = torch.nn.GRU(2 * config.hidden_size, config.hidden_size, bidirectional=True, batch_first=True, dropout=0.2)
        self.sent_attn = torch.nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.sent_ctx = torch.nn.Linear(config.hidden_size, 1, bias=False)

        self.dropout = torch.nn.Dropout(config.dropout)
        self.out = torch.nn.Linear(config.hidden_size * 2, config.num_classes)


    def forward(self, inputs, hidden1=None, hidden2=None):  # [b, 60, 15]
        batch_size = inputs[0].shape[0]
        x = inputs[0].view(batch_size, self.max_sents_num, self.max_word_num) # 将每个文本切分为多个句子的形式

        embedded = self.dropout0(self.embedding(x))  # =>[b, 60, 15, 100]

        word_inputs = embedded.view(-1, embedded.size()[-2], embedded.size()[-1])  # =>[b*60, 15, embedding_dim]
        # word_inputs = self.layernorm1(word_inputs)
        self.bi_rnn1.flatten_parameters()
        """
        为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)。
        类似我们调用tensor.contiguous
        """
        word_encoder_output, hidden1 = self.bi_rnn1(word_inputs,
                                                    hidden1)  # =>[b*60,15,2*hidden_size], [b*60,2,hidden_size]
        word_attn = self.word_attn(word_encoder_output).tanh()  # =>[b*60,15,hidden_size]
        word_attn_energy = self.word_ctx(word_attn)  # =>[b*60,15,1]
        word_attn_weights = F.softmax(word_attn_energy, dim=1).transpose(1, 2)  # =>[b*60,15,1]=>[b*60,1,15]
        word_att_level_output = torch.bmm(word_attn_weights, word_encoder_output)  # =>[b*60,1,2*hidden_size]

        sent_inputs = word_att_level_output.squeeze(1).view(-1, self.max_sents_num, 2 * self.hidden_size)  # =>[b*60,2*hidden_size]=>[b,60,2*hidden_size]

        self.bi_rnn2.flatten_parameters()
        sent_encoder_output, hidden2 = self.bi_rnn2(sent_inputs, hidden2)  # =>[b,60,2*hidden_size], [b,2,hidden_size]
        sent_attn = self.sent_attn(sent_encoder_output).tanh()  # =>[b,60,hidden_size]
        sent_attn_energy = self.sent_ctx(sent_attn)  # =>[b,60,1]
        sent_attn_weights = F.softmax(sent_attn_energy, dim=1).transpose(1, 2)  # =>[b,60,1]=>[b,1,60]
        sent_att_level_output = torch.bmm(sent_attn_weights, sent_encoder_output)  # =>[b,1,2*hidden_size]

        # logits = self.out(self.dropout(self.layernorm2(sent_att_level_output.squeeze(1))))  # =>[b,2*hidden_size]=>[b,num_classes]
        logits = self.out(self.dropout(sent_att_level_output.squeeze(1)))  # =>[b,2*hidden_size]=>[b,num_classes]
        return logits  # [b,num_classes]