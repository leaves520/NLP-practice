import torch.nn as nn
import torch

class RnnGeneration(nn.Module):
    def __init__(self, word_size, dim, hidden_dim, pad_idx, num_layers=2, dropout=0.2, wordembedding=None):
        super(RnnGeneration, self).__init__()

        if wordembedding is None:
            self.wordembed = nn.Embedding(word_size, dim, padding_idx=pad_idx)
        else:
            assert isinstance(wordembedding, nn.Module)
            self.wordembed = wordembedding

        self.LSTM = nn.LSTM(input_size=dim, hidden_size=hidden_dim, num_layers=2,batch_first=True)
        self.project = nn.Linear(hidden_dim, word_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embed_x = self.wordembed(x)
        output, (h0, c0) = self.LSTM(embed_x) # output: batch_size * seq_len * hidden_dim
        output = self.dropout(output)

        y = self.project(output) # batch_size * seq_len * word_size
        return y

