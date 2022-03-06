import random
import numpy as np
from datahelper import DataSet



class SoftmaxRegression:
    def __init__(self, X_size, y_size):
        self.feature_size = X_size
        self.W = np.random.random(size=(X_size,y_size)) # need to update by BP-algorithm
        self.bias = np.random.random(size=(1, y_size)) # need to update by BP-algorithm


    def softmax(self, output):
        # softmax(x) = exp(x) / (\sum^C{exp(x)})
        sum_ex = np.sum(np.exp(output), axis=1, keepdims=True)
        return np.exp(output) / sum_ex  # numpy broadcast mechanism


    def compute_loss(self, X, y): # cross-entropy function
        # L(w) = - 1/N \sum{y\log(\hat{y})} | y: Truth \hat{y}: softmax(W^T X)
        N = len(X)
        log_y_hat = np.log(self.softmax(np.matmul(X, self.W)+self.bias))
        y_mul_log_yhat = y * log_y_hat  # element-wise multiplication

        return - (1 / N ) * np.sum(y_mul_log_yhat)


    def update_by_gradientDescent(self, X, y, lr=1e-3):
        # L(w)'/ w = - (1 / N ) * \sum{ x (y-\hat{y})^T} -> shape = self.W.shape
        # L(b)'/ b = - (1 / N ) * \sum{(y-\hat{y})^T} -> shape = self.bias.shape
        N = len(X)
        y_hat = self.softmax(np.matmul(X, self.W)+self.bias)
        gradient_w = - (1 / N) * np.sum(np.matmul(np.transpose(X),(y - y_hat)),axis=0, keepdims=True) # shape=(X_size,y_size)
        gradient_b = - (1 / N) * np.sum((y - y_hat),axis=0, keepdims=True)  # shape=(1,y_size)
        self.W = self.W - lr * gradient_w
        self.bias = self.bias - lr * gradient_b


    def acc(self, X, y):
        predict = np.argmax(self.softmax(np.matmul(X, self.W) + self.bias),axis=1)
        y_ = np.argmax(y,axis=1)
        N = len(X)
        return np.sum(y_ == predict) / N


#### test code ####
## test softmax and loss function
# test = SoftmaxRegression(X_size=4,y_size=3)
# x = np.random.random(size=(3,4))  # 3 samples, 4 feature size
# y = np.array([[1,0,0],[0,1,0],[0,0,1]])  # 3 class: [0,1,2]
# print(x)
# print(test.softmax(x))
# print(test.softmax(x).sum(axis=1))  # sum is one

#
# ## test our Cross-entropy
# print(test.compute_loss(x,y))
#
# ## test torch Cross-entropy
# import torch
# from torch.nn import CrossEntropyLoss
# ce_torch = CrossEntropyLoss()
# print(ce_torch(torch.tensor(np.matmul(x, test.W)+test.bias),torch.tensor([0,1,2])))
#
#
# ## test gradient
# print(test.W,test.bias)
# test.update_by_gradientDescent(x,y,lr=0.01)
# print('update by bp:',test.W,test.bias)
#
#


if __name__ == '__main__':
    data_get = DataSet(sample=0.03)
    trn_x, trn_y, val_x, val_y, test_x, text_y = data_get.SplitData(frac=(0.7, 0.1, 0.2))
    x_feats = trn_x.shape[-1]
    y_feats = trn_y.shape[-1]
    model = SoftmaxRegression(x_feats, y_feats)
    batch_size = 16
    epochs = 500
    lr = 0.5

    for e in range(1,epochs+1):
        batchs = data_get.batch_iter(trn_x, trn_y,batch_size=batch_size)
        for X, y in batchs:
            loss = model.compute_loss(trn_x, trn_y)
            acc = model.acc(trn_x, trn_y)
            print('epochs {} loss: {} acc: {}'.format(e,loss,acc))

            model.update_by_gradientDescent(trn_x, trn_y, lr=lr)


        val_x_batch, val_y_batch = next(iter(data_get.batch_iter(val_x, val_y,batch_size=len(val_x))))
        acc_val = model.acc(val_x_batch,val_y_batch)
        print('=====> val acc: {}'.format(acc_val))


    print('===========test=========')
    test_x_batch, test_y_batch = next(iter(data_get.batch_iter(test_x, text_y, batch_size=len(test_x))))
    print('Final test acc: {}'.format(model.acc(test_x_batch, test_y_batch)))


    ## test sklearn-package SoftmaxRegression
    X, y = next(iter(data_get.batch_iter(trn_x, trn_y, batch_size=len(trn_x))))
    test_x, test_y = next(iter(data_get.batch_iter(test_x, text_y, batch_size=len(test_x))))
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X, np.argmax(y,axis=1))
    predictions = classifier.predict_proba(test_x)
    predict_y = np.argmax(predictions,axis=1) # the max logits is predict class
    true_y = np.argmax(test_y,axis=1)
    N = len(X)
    print('sklearn-package acc:' ,np.sum(predict_y == true_y) / N)


    # Final test acc: 0.19666666666666666
    # sklearn-package acc: 0.22952380952380952













