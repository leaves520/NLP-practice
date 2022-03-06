from datahelper import MyDataset
from model import ESIM, CEloss
import torch.optim as optim
import torch
from sklearn.metrics import accuracy_score


if __name__ == '__main__':

    dataset = MyDataset(sample=0.05,word_cutoff=300)
    trn, val, test = dataset.create_pydataset()
    print('====DataPrepare Finshed====')
    print("trn val test: ",len(trn),len(val),len(test))
    dataloader_trn = dataset.get_dataloader(data=trn, batch_size=32, type='train')
    dataloader_val = dataset.get_dataloader(data=val, batch_size=32, type='val')
    dataloader_test = dataset.get_dataloader(data=test, batch_size=32, type='test')

    vocab_size = len(dataset.word2idx)
    task_nums = len(dataset.label2idx)
    model = ESIM(vocab_size=vocab_size, hidden_dim=100, task_nums=task_nums, dropout=0.5, OOV_size=0, word_embedding=None)
    optimer = optim.Adam(params=model.parameters(),lr=0.0004)
    criterion = CEloss()

    EPOCHS = 150
    for e in range(EPOCHS):
        model.train()

        for idx, (p, h, y, p_mask, h_mask) in enumerate(dataloader_trn):
            pre_proba = model(p, h, p_mask, h_mask)
            loss = criterion(pre_proba, y)
            optimer.zero_grad()
            loss.backward()
            optimer.step()

            if idx % 20 == 0:
                print(f'epochs {e} | loss: {loss}.')

    print('Training Finished.')
    print('-------test------')
    with torch.no_grad():
        model.eval()
        y_true = []
        y_pre = []
        for idx, (p, h, y, p_mask, h_mask) in enumerate(dataloader_test):
            pre_proba = model(p, h, p_mask, h_mask)
            y_true.extend(y.numpy())
            pre_label = torch.argmax(pre_proba, dim=-1).numpy()
            y_pre.extend(pre_label)

        acc = accuracy_score(y_pre, y_true)
        print('Model in test, acc: {}'.format(acc))


