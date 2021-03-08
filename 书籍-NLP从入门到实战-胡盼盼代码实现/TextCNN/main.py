import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score
import time


class MRdataset(Dataset):  # 创建pytorch dataset, 主要要重写 __getitem__ 和 __len__
    def __init__(self, data):
        from ast import literal_eval  # 由于pandas保存list后会转为str，再次读取时要转list
        data_label = data
        self.data = [literal_eval(sen) for sen in data_label['word_index'].to_list()]
        self.label = data_label['label'].to_list()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''get datapoint with index'''
        data = torch.LongTensor(self.data[idx])
        label = torch.tensor(self.label[idx])

        return data, label


class TextCNN(nn.Module):
    def __init__(self, dict_len, hidden_dim, tasks):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(dict_len, hidden_dim)

        self.cov1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(3, hidden_dim), stride=1) # out_channels:其实是卷积核的个数
        self.cov2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(4, hidden_dim), stride=1)
        self.cov3 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(5, hidden_dim), stride=1)

        self.pool1 = nn.MaxPool2d((10 - 3 + 1, 1), stride=1)  # 卷积后大小： (n - f) / stride + 1
        self.pool2 = nn.MaxPool2d((10 - 4 + 1, 1), stride=1)
        self.pool3 = nn.MaxPool2d((10 - 5 + 1, 1), stride=1)

        self.dropout = nn.Dropout(0.1)
        self.linear_project = nn.Linear(9,2)
        self.activate = nn.Softmax(dim=-1)

    def forward(self,x):
        x = self.embedding(x)
        x = x.unsqueeze(1) # batch, channel, seq_len, hidden_dim , 多加一个channal维度

        x1 = self.pool1(self.cov1(x)).squeeze()
        x2 = self.pool2(self.cov2(x)).squeeze()
        x3 = self.pool3(self.cov3(x)).squeeze()

        x = torch.cat([x1,x2,x3],dim=1)
        x = self.dropout(x)
        y = self.linear_project(x)

        return self.activate(y)


if __name__ == '__main__':
    # 读字典
    with open('data/word_dict','rb') as f:
        dict_word = pickle.load(f)
    dict_len = len(dict_word)

    # 读数据并打乱数据
    data = pd.read_csv('data/train_test.csv')
    data = data.sample(frac=1).reset_index(drop=True) # 重置index并弃掉原来的index
    train_data = data.sample(frac=0.8) # 划分训练集
    test_data = data[~data.index.isin(train_data.index)] # 取出测试集，关键在isin

    Train = MRdataset(train_data)
    Test = MRdataset(test_data)
    dataloader_train = DataLoader(Train,batch_size=64,shuffle=True,drop_last=False)
    dataloader_test = DataLoader(Test, batch_size=64, shuffle=False, drop_last=False)

    model = TextCNN(dict_len,hidden_dim=64,tasks=2)
    crtien = nn.CrossEntropyLoss()
    optimer = optim.Adam(model.parameters(),lr=1e-3)
    scheduler = lr_scheduler.ExponentialLR(optimer, gamma=0.8)


    log = open('log/log_{}.txt'.format(time.strftime('%y%m%d%H')), 'w')  # 记录训练日志
    log.write('epoch step loss\n')

    EPOCHS = 100
    for e in range(EPOCHS):
        model.train()
        for index,(x,y) in enumerate(dataloader_train):
            pre_proba = model(x)
            loss = crtien(pre_proba,y)
            optimer.zero_grad()
            loss.backward()
            optimer.step()
            if index % 20 == 0:
                print('epochs_{}, loss: {}'.format(e+1,loss))
                data = str(e + 1) + ' ' + str(index + 1) + ' ' + str(loss.item()) + '\n'
                log.write(data)
                log.flush()

        scheduler.step()
        torch.save(model.state_dict(),'cur_model.pt')

        model.eval()
        y_true = []
        y_pre = []
        y_pos_proba = []
        for id, (x, y) in enumerate(dataloader_test):
            y_true.extend(y.numpy())
            pre_proba = model(x)
            y_pre.extend(torch.argmax(pre_proba,dim=-1).numpy())
            y_pos_proba.extend(pre_proba[:,1].detach().numpy())

        acc = accuracy_score(y_true,y_pre)
        auc = roc_auc_score(y_true,y_pos_proba)

        print('---------------------test--------------------')
        print("\tacc:{}, auc:{}".format(acc,auc))



