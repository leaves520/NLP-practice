from torch.utils.data import DataLoader
from model import LstmCrf, get_mask
from data_set import MyDataSet, ModelEmbedding, collate_fn
import torch

w2v_name = 'glove-wiki-gigaword-50'
data_path = 'labeled_data.json'

model_embedding = ModelEmbedding(model_name=w2v_name)
model = LstmCrf(input_dim=model_embedding.embedding_size, hidden_dim=100, n_class=model_embedding.n_class,
                vocab_size=model_embedding.vocab_size, gensim_embedding=model_embedding.embedding)
word2id = model_embedding.word2id

optim = torch.optim.Adam(model.parameters(), lr=1e-3)  # 优化器
my_dataset = MyDataSet(data_path, word2id, test=True)
print('The size of training data: {}'.format(len(my_dataset)))

train_size = int(0.8 * len(my_dataset))
test_size = len(my_dataset) - train_size
trn_data, test_data = torch.utils.data.random_split(my_dataset, [train_size, test_size])

trn_dataloader = DataLoader(trn_data, batch_size=10, collate_fn=collate_fn)  # 可迭代对象
test_dataloader = DataLoader(test_data, batch_size=4, collate_fn=collate_fn)

print('Training Start!')
for e in range(100):
    loss = 0
    for x, y, lengths in trn_dataloader:
        x = x.long()
        y = y.long()
        mask = get_mask(lengths)
        emission = model.forward(input_data=x, input_len=lengths)
        loss = model.get_loss(emission=emission, labels=y, mask=mask)

        optim.zero_grad()
        loss.backward()
        optim.step()
    print('epochs: {} -> {}'.format(e, loss.item()))
