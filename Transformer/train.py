from model import Transformer
from dataset import dataloader,vocabulary
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Transformer(vocab_size=vocabulary.dict_size_).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters())


for epoch in range(30):
    for enc_inputs,dec_inputs,dec_outputs in dataloader:
        enc_inputs,dec_inputs,dec_outputs = enc_inputs.to(device),dec_inputs.to(device),dec_outputs.to(device)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs,dec_outputs.view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()