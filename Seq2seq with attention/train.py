from model import *
from dataset import *
import torch.optim as optim
import torch

EPOCHS = 300
embed_size = 100
enc_hid_dim = dec_hid_dim = 200

file_path = './subtitle.txt'
vocab = Vocabolary(file_path)
dataloader = DataLoader(dataset=MyDataset(vocab),drop_last=False,shuffle=True,batch_size=16,collate_fn=Collator())

model = Seq2Seq_attention(vocab_size=vocab.vocab_size_,embed_size=embed_size,enc_hid_dim=enc_hid_dim,dec_hid_dim=dec_hid_dim)
model.to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0).to(device) # 忽略<PAD> :0 的loss
optimizer = optim.Adam(model.parameters(),lr=1e-3)


for e in range(EPOCHS):
    for index,(src,tgt) in enumerate(dataloader):
        # src,tgt : [batch_size,src or tgt length]
        tgt_right = torch.zeros_like(tgt)  # 获取右移序列作为输入, i like you -> START i like
        tgt_right[:,0] = torch.full((1,tgt.shape[0]),vocab.index('START'))
        tgt_right[:,1:] = tgt[:,0:-1]

        pred = model(src,tgt_right.to(device))

        # CrossEntropy输入：pred为二维tensor (N-样本数, C-类别数), true_label为一维的tensor,表示属于哪个类别 (N-样本数). 如[0,7,8,1,...]
        # 因此要将pred,tgt展平成loss计算的格式, (展平会按原来batch的顺序进行拼接的)
        pred = pred.reshape(-1,pred.shape[-1]) # [batch_size,tgt_len,vocab_size] -> [batch_size*tgt_len,vocab_size]
        tgt = tgt.reshape(-1)  # [batch_size,tgt_len] -> [batch_size * tgt_len]
        loss = criterion(pred,tgt)

        # if index % 30 == 0:
        print(f'epochs:{e} steps:{index} loss:{loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(),'best.pt')

