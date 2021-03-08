import pickle
from data.data_parse import DataParse
from model.seq2seq import Seq2Seq,Critertion,Seq2Seq_inference
from config import DefaultConfig
import torch.optim as optim
import torch

def train():
    data = DataParse(DefaultConfig)
    input_,output_ = data.data_read()
    dict_len, encoder_input_data, decoder_input_data, decoder_target_data = data.data_parse(input_,output_)

    # 三大关键东西：模型、损失函数、优化器，这些都封装成类来处理
    model = Seq2Seq(DefaultConfig,dict_len,encoder_input_data,decoder_input_data) # 模型
    crtien = Critertion() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # 优化器

    best = float('inf')
    p = 0
    patience = 50
    for e in range(1000000):
        loss = crtien(model(),decoder_target_data)
        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # update
        print('epochs{}：training loss: {}'.format(e + 1, loss))

        if loss < best:
            best = loss
            p = 0
            torch.save(model.state_dict(),'model/best_current.pt')  # 保存模型结构参数，字典的形式
        else:
            p += 1

        if p > patience:
            print("early stop!")
            break



def test():
    with open(DefaultConfig.dict_path,'rb') as f:
        dict_word2index = pickle.load(f)  # 二进制的pickle文件可以用pickle.dump和load进行写出和写入
    dict_index2word = dict([(index,word) for word,index in dict_word2index.items()])

    model = Seq2Seq_inference(DefaultConfig,len(dict_word2index))
    model.load_state_dict(torch.load('model/best_current.pt'))
    model.eval()  # 这样就不会累积梯度，作预测时记得加这个

    while True:
        source = input("请输入句子：")
        output = model(source,dict_word2index,dict_index2word)
        print(output)


if __name__ == '__main__':
    #train()
    test()
