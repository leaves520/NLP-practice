import torch
import torch.nn as nn

Embedding_Size = 200

class Word2Vec(nn.Module):
    def __init__(self,vocab_size,embed_size):
        super(Word2Vec, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(vocab_size,embed_size)  # 训练出来的是作为中心词的embedding, 是最后模型训练结束后我们需要的
        self.context_embed = nn.Embedding(vocab_size,embed_size) # 训练出来的是作为上下文词的embedding

    def forward(self,input,pos_target,neg_target):

        batch_size = input.shape[0]
        input_embedding = self.in_embed(input) # batch_size, embedding_size
        pos_embedding = self.context_embed(pos_target) # batch_size,window_size * 2, embedding_size
        neg_embedding = self.context_embed(neg_target) # batch_size,window_size * 2 * k, embedding_size

        input_embedding = input_embedding.view(batch_size,1,Embedding_Size)
        pos_embedding = pos_embedding.view(batch_size,Embedding_Size,-1)
        neg_embedding = neg_embedding.view(batch_size,Embedding_Size,-1)

        pos_loss = torch.bmm(input_embedding,pos_embedding).squeeze(1) # batch_size,window_size * 2
        neg_loss = torch.bmm(input_embedding,-neg_embedding).squeeze(1) # batch_size,window_size * 2 * k

        pos_loss = torch.sigmoid(pos_loss).log().sum(1)
        neg_loss = torch.sigmoid(neg_loss).log().sum(1)

        loss = pos_loss + neg_loss

        return -loss   # 极大化对数似然函数等于极小化负对数似然损失


    @property
    def input_embedding_(self):
        return self.in_embed.weight.detach().numpy()

