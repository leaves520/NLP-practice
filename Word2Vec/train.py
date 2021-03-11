from model import *
from dataset import *
from torch.utils.data import DataLoader
import os,time


EPOCHS = 200
index_text, vocab = get_Vocab_text('./text8.test.txt', test=False)
dataloader = DataLoader(MyDataset(index_text, vocab),drop_last=False,shuffle=True,batch_size=32)
model = Word2Vec(vocab_size=vocab.vocab_size_,embed_size=Embedding_Size)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)


# 保存训练记录到日志文件
if not os.path.exists('log/'):
    os.makedirs('log/')
log = open('log/train_info_{}.txt'.format(time.strftime('%y-%m%d-%H%M')), 'w')
log.write('epoch step loss\n')

# 训练
for e in range(EPOCHS):
    for i,(intput,pos_target,neg_target) in enumerate(dataloader):
        loss = model(intput,pos_target,neg_target).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 30 ==0:
            print('epoch', e, 'iteration', i, loss.item())
            date = " ".join(['epoch',str(e),'iteration',str(i),'loss',str(loss.item())])
            log.write(date + '\n')
            log.flush()

# 保存训练好的input_embedding层参数，以后可作为预训练模型
embedding_weights = model.input_embedding_
torch.save(model.state_dict(), "embedding-{}.th".format(Embedding_Size))