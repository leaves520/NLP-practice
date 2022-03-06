import collections
import warnings
import torch
import torch.optim as optim
import torch.nn.functional as F
from datahelper import PoetryTang
from model import RnnGeneration
import numpy as np
warnings.filterwarnings("ignore")

EPOCHS = 100
batch_size = 10
lr = 1e-3

dataset = PoetryTang(sample=1)
trn, val, test = dataset.CreateTorchDataset()
print(dataset.vocabulary.property)
VocabSetting = dataset.vocabulary.property

trn_dataloader, val_dataloader, test_dataloader = \
    dataset.CreateDataLoader(trn, shuffle=True, batch_size=batch_size), \
    dataset.CreateDataLoader(val, batch_size=4), dataset.CreateDataLoader(test, batch_size=4)

model = RnnGeneration(word_size=VocabSetting['vocab_size'], dim=50, hidden_dim=50, pad_idx=VocabSetting['<PAD>'],
                      num_layers=2, dropout=0.2, wordembedding=None)
optimer = optim.Adam(model.parameters(),lr=lr)
criterion = torch.nn.CrossEntropyLoss(reduction='none')
# nn.CrossEntropyLoss: pred: N*C (possibility, that is after softmax). label: N


def PPL(pred_possibility, y, mask):
    '''
    Ref: 《[神经网络与深度学习](https://nndl.github.io/)》 第15章，第4节—评价方法)
    '''
    pred_possibility = pred_possibility.view(-1, dataset.vocabulary.property['vocab_size'])
    y = y.flatten()
    mask = mask.flatten()

    end_sum = 0
    ppl = 1.
    for pos_i, y_i, mask_i in zip(pred_possibility, y, mask):
        # Note: remove the possibility of <END> tag and <PADDING> tag
        if mask_i == 1 and y_i != dataset.vocabulary.property['<END>']:
            ppl *= pos_i[y_i]
        if y_i == dataset.vocabulary.property['<END>']:
            end_sum += 1


    T = mask.sum() - end_sum
    ppl = 1 / torch.pow(ppl, 1/T)

    return ppl.item()


def BELU(pre_possibility, y, mask, gram):
    bacth_size = len(pre_possibility)
    end_idx = dataset.vocabulary.property['<END>']
    belu_bacth = 0.
    for pre_pos, y_i, mask_i in zip(pre_possibility, y, mask):
        pre = torch.argmax(pre_pos, dim=-1)
        predict = []
        label = []
        for j in range(len(mask_i)):
            if mask_i[j] == 1 and y_i[j] != end_idx:
                predict.append(str(pre[j].item()))
                label.append(str(y_i[j].item()))

        belu_i = 0
        pred_n_gram = collections.defaultdict(int)
        label_n_gram = collections.defaultdict(int)
        for n in range(1, gram + 1):
            for i in range(len(predict)-n+1):
                pred_n_gram[','.join(predict[i:i+n])] += 1
            w_total_pred = sum(pred_n_gram.values())

            for i in range(len(label)-n+1):
                label_n_gram[','.join(label[i:i+n])] += 1

            total = 0
            for k, v in pred_n_gram.items():
                total += min(v, label_n_gram[k])

            belu_i += 1/gram * np.log(total/w_total_pred)

        belu_bacth += np.exp(belu_i)


    return belu_bacth / bacth_size



def ROUGE(pre_possibility, y, mask, gram):
    bacth_size = len(pre_possibility)
    end_idx = dataset.vocabulary.property['<END>']
    rouge_bacth = 0.
    for pre_pos, y_i, mask_i in zip(pre_possibility, y, mask):
        pre = torch.argmax(pre_pos, dim=-1)
        predict = []
        label = []
        for j in range(len(mask_i)):
            if mask_i[j] == 1 and y_i[j] != end_idx:
                predict.append(str(pre[j].item()))
                label.append(str(y_i[j].item()))

        rouge_i = 0
        pred_n_gram = collections.defaultdict(int)
        label_n_gram = collections.defaultdict(int)
        for n in range(1, gram + 1):
            for i in range(len(predict) - n + 1):
                pred_n_gram[','.join(predict[i:i + n])] += 1

            for i in range(len(label) - n + 1):
                label_n_gram[','.join(label[i:i + n])] += 1
            w_total_label = sum(label_n_gram.values())

            total = 0
            for k, v in label_n_gram.items():
                total += min(v, pred_n_gram[k])

            rouge_i += 1 / gram * np.log(total / w_total_label)

        rouge_bacth += np.exp(rouge_i)

    return rouge_bacth / bacth_size








for e in range(EPOCHS):
    model.train()
    for i, (x, y, mask) in enumerate(trn_dataloader):
        # print(x,y)
        pred_logtis = model(x)
        pred_possibility = F.softmax(pred_logtis, dim=-1).view(-1, dataset.vocabulary.property['vocab_size'])
        y = y.flatten()
        mask = mask.flatten()
        loss = torch.sum(criterion(pred_possibility, y)*mask) / mask.sum()  # the pos of padding not be include in loss

        optimer.zero_grad()
        loss.backward()
        optimer.step()

        # gradient clip
        # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)

        if i % 10 == 0:
            print(f'epochs:{e}| iter:{i} -> loss: {loss.item()}')

    if e % 10 == 0:
        print('======Evaluation======')
        model.eval()
        ppl = 0.
        belu = 0.0
        rouge = 0.0
        with torch.no_grad():
            for i, (x, y, mask) in enumerate(val_dataloader):
                pred_possibility = F.softmax(model(x),dim=-1)
                ppl += PPL(pred_possibility, y, mask)
                belu += BELU(pred_possibility, y, mask, 3)

            N = len(val_dataloader)
            print('epochs{} ==> PPL:{} | BELU:{} | ROUGE:{}'.format(e, ppl / N, belu / N, rouge / N))