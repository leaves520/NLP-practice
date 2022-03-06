import torch
from datahelper import Dataset
from torch_model import TextCNN, WordEmbedding
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score


if __name__ == '__main__':
    dataset = Dataset(sample=0.05,word_cutoff=3000)
    X_trn, y_trn, X_val, y_val, X_test, y_test, word2idx, idx2word, label2idx = dataset.splitdata()

    Use_pretrain = True
    epochs = 150
    lr = 1e-3
    max_length = 200 # the largest length of sentence
    batch_size = 32

    if not Use_pretrain:
        model = TextCNN(vocab_size=len(word2idx), feature_size=100, max_length=max_length, tasks=len(label2idx)
                        ,feature_maps=100, filter_windows=(3,4,5))
    else:
        WordEmbedding = WordEmbedding(path='../dataset/pretrain_cn_sgns.wiki.char', available_vocab=word2idx)
        model = TextCNN(vocab_size=len(word2idx), feature_size=WordEmbedding.feature_size, max_length=max_length,
                        tasks=len(label2idx), feature_maps=100, filter_windows=(3,4,5))

    criterion = nn.CrossEntropyLoss()
    optimer = Adam(model.parameters(), lr=lr)

    for e in range(epochs):
        model.train()
        batch_trns = dataset.batch_iter(X_trn,y_trn,batch_size=batch_size,max_length=max_length)
        for idx,(x,y) in enumerate(batch_trns):
            pre_proba = torch.softmax(model(x),dim=-1)
            loss = criterion(pre_proba, y)
            optimer.zero_grad()
            loss.backward()
            optimer.step()

            if idx % 20 == 0:
                print(f'epochs {e} | loss: {loss}.')

    print('Training Finished.')
    print('-------test------')
    batch_test = dataset.batch_iter(X_test, y_test, batch_size=batch_size, max_length=max_length)
    with torch.no_grad():
        model.eval()
        y_true = []
        y_pre = []
        for x,y in batch_test:
            y_true.extend(y.numpy())
            pre_proba = model(x)
            pre_label = torch.argmax(pre_proba, dim=-1).numpy()
            y_pre.extend(pre_label)

        acc = accuracy_score(y_pre, y_true)
        print('Model in test, acc: {}'.format(acc))
