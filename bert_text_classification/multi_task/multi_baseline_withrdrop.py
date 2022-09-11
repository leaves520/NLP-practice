#! -*- coding:utf-8 -*-

import argparse
import os
import random
from time import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import tensorflow as tf
from bert4keras.backend import keras, set_gelu, K
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import open
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.tokenizers import Tokenizer
from keras.backend import sparse_categorical_crossentropy
from keras.layers import Lambda, Dense, SpatialDropout1D, SpatialDropout2D, Dropout, Reshape, Conv2D, MaxPooling2D, \
    Concatenate, Flatten
from keras.losses import Loss, kld, cosine_proximity
from keras.models import load_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

os.environ['TF_KERAS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
set_gelu('tanh')  # 切换gelu版本

num_classes_label_one = 2
num_classes_label_two = 11
num_classes_label_three = 69
maxlen = 150
batch_size = 64

label_map_one = {'租车售前': 0, '其他': 1}

label_map_two = {la.strip(): idx for idx, la in enumerate(open('label_two.txt', encoding='utf-8-sig'))}
label_map_three = {la.strip(): idx for idx, la in enumerate(open('label_three.txt', encoding='utf-8-sig'))}

config_path = './albert/albert_config.json'
checkpoint_path = './albert/albert_model.ckpt.data-00000-of-00001'
dict_path = './albert/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')


def my_confusion_matrix(y_true, y_pred):
    """输出混淆矩阵
    """
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print("confusion_matrix(left labels: y_true, up labels: y_pred):")
    buf = ["labels"]
    for i in range(len(labels)):
        buf.append(str(labels[i]))
    print("\t    ".join(buf))  # 输出标签
    buf.clear()
    for i in range(len(conf_mat)):
        buf.append("label" + str(labels[i]))
        for j in range(len(conf_mat[i])):
            buf.append(str(conf_mat[i][j]))
        print("\t    ".join(buf))
        buf.clear()
    print()


# 数据导入
def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


def load_excel(filename, sheetname, text_column, label_column_one, label_column_two, label_column_three):
    xlsx_data = pd.read_excel(filename, sheet_name=sheetname)
    all = []
    for i in tqdm(range(0, len(xlsx_data))):
        text = str(xlsx_data.iloc[i][text_column]).lower().replace(" ", "")
        label_one = str(xlsx_data.iloc[i][label_column_one])
        label_two = str(xlsx_data.iloc[i][label_column_two])
        label_three = str(xlsx_data.iloc[i][label_column_three])
        if text is None or text == '' or label_one is None or label_one == '' or label_one not in label_map_one or label_two is None or label_two == '' or label_two not in label_map_two or label_three is None or label_three == '' or label_three not in label_three:
            continue
        all.append((text, label_map_one[label_one], label_map_two[label_two], label_map_three[label_three]))
    random.shuffle(all)
    percent7 = int(0.8 * len(all))
    percent9 = int(0.99 * len(all))
    train_data = all[0:percent7]
    valid_data = all[percent7:]
    test_data = all[percent9:]
    # print('训练集类别分布：', " 1 : ", len([x for x in train_data if x[1] == 1]), " 0 : ",
    #       len([x for x in train_data if x[1] == 0]))
    # print('验证集类别分布：', " 1 : ", len([x for x in valid_data if x[1] == 1]), " 0 : ",
    #       len([x for x in valid_data if x[1] == 0]))

    return train_data, valid_data, test_data


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels_one, batch_labels_two, batch_labels_three = [], [], [], [], []
        for is_end, (text, label_one, label_two, label_three) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            token_replace_dict = {98: 101, 99: 102}
            token_ids = [token_replace_dict[i] if i in token_replace_dict else i for i in token_ids]

            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels_one.append([label_one])
                batch_labels_two.append([label_two])
                batch_labels_three.append([label_three])

            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids, length=maxlen)
                batch_segment_ids = sequence_padding(batch_segment_ids, length=maxlen)
                batch_labels_one = sequence_padding(batch_labels_one)
                batch_labels_two = sequence_padding(batch_labels_two)
                batch_labels_three = sequence_padding(batch_labels_three)
                yield [batch_token_ids, batch_segment_ids], [batch_labels_one, batch_labels_two, batch_labels_three]
                batch_token_ids, batch_segment_ids, batch_labels_one, batch_labels_two, batch_labels_three = [], [], [], [], []


from keras.losses import kullback_leibler_divergence as kld

alpha = 4


def categorical_crossentropy_with_rdrop(y_true, y_pred):
    """配合上述生成器的R-Drop Loss
    其实loss_kl的除以4，是为了在数量上对齐公式描述结果。
    """
    loss_ce = K.sparse_categorical_crossentropy(y_true, y_pred)  # 原来的loss
    loss_kl = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
    return K.mean(loss_ce) + K.mean(loss_kl) / 4 * alpha


class PtmModel(object):
    def __init__(self,
                 model_path,
                 _mode: str,
                 filename: str,
                 sheetname: str,
                 text_column: str,
                 label_column_one: str,
                 label_column_two: str,
                 label_column_three: str,
                 epochs=20,
                 *args,
                 **kwargs):

        self.xlsx_path = filename
        self.sheet = sheetname
        self.text_column = text_column
        self.label_column_one = label_column_one
        self.label_column_two = label_column_two
        self.label_column_three = label_column_three
        self.label_map_one = label_map_one
        self.label_map_two = label_map_two
        self.label_map_three = label_map_three
        self.label_index_map_one = {v: k for k, v in self.label_map_one.items()}
        self.label_index_map_two = {v: k for k, v in self.label_map_two.items()}
        self.label_index_map_three = {v: k for k, v in self.label_map_three.items()}
        self.model_path = model_path
        self.epochs = epochs
        if _mode == 'train':
            # 派生为带分段线性学习率的优化器。
            # 其中name参数可选，但最好填入，以区分不同的派生优化器。
            # 这里的Strategy可以换成想用的，因为其他三个还是experimental的状态，建议用这个
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                # 加载预训练模型
                bert = build_transformer_model(
                    config_path=config_path,
                    #                     checkpoint_path=checkpoint_path,
                    model='albert',
                    return_keras_model=False,
                )

                output = Lambda(lambda x: x[:, 1:-1], name='ALL-token')(bert.model.output)
                output = Dropout(0.5)(output)

                albert_output = Reshape((maxlen - 2, 312, 1))(output)

                # textcnn 提取一级标签关键词
                conv_one = []
                filter_sizes = [2, 3, 4, 5, 6, 7]  # 使用6个不同尺寸的卷积核
                for fsz in filter_sizes:
                    l_conv_one = Conv2D(filters=128, kernel_size=(fsz, 312), activation='relu')(albert_output)
                    l_pool_one = MaxPooling2D((maxlen - fsz - 1, 1))(l_conv_one)
                    conv_one.append(l_pool_one)

                mer_one = Concatenate(axis=3)(conv_one)
                output = Flatten()(mer_one)
                # output = Reshape((len(filter_sizes)*128,))(mer)

                # 使用DNN投射到三个分类空间
                labelone_output = Dense(
                    units=512,
                    activation='relu',
                    kernel_initializer=bert.initializer
                )(output)
                labelone_output = Dropout(0.35)(labelone_output)

                labeltwo_output = Dense(
                    units=512,
                    activation='relu',
                    kernel_initializer=bert.initializer
                )(output)
                labeltwo_output = Dropout(0.35)(labeltwo_output)

                labelthree_output = Dense(
                    units=512,
                    activation='relu',
                    kernel_initializer=bert.initializer
                )(output)
                labelthree_output = Dropout(0.35)(labelthree_output)

                # 获得分类添加门控层
                output1 = Dense(
                    units=num_classes_label_one,
                    activation='softmax',
                    kernel_initializer=bert.initializer
                )(labelone_output)

                output2 = Dense(
                    units=num_classes_label_two,
                    activation='softmax',
                    kernel_initializer=bert.initializer
                )(labeltwo_output)

                output3 = Dense(
                    units=num_classes_label_three,
                    activation='softmax',
                    kernel_initializer=bert.initializer
                )(labelthree_output)

                # 定义模型的时候放到镜像策略空间就行
                self.model = keras.models.Model(bert.model.input, [output1, output2, output3])
                self.model.summary()
                # optimizer=Adam(1e-5),  # 用足够小的学习率
                self.model.compile(loss=categorical_crossentropy_with_rdrop,
                                   optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}),
                                   metrics=['accuracy'], )
        elif _mode == 'predict':
            self.model = load_model(self.model_path, custom_objects={'categorical_crossentropy_with_rdrop':categorical_crossentropy_with_rdrop})

    def train(self):
        # 1、构建数据集
        train_data, valid_data, test_data = \
            load_excel(self.xlsx_path, self.sheet, self.text_column, self.label_column_one, self.label_column_two,
                       self.label_column_three)

        # 2、转换数据集
        train_generator = data_generator(train_data, batch_size)
        valid_generator = data_generator(valid_data, batch_size)
        test_generator = data_generator(test_data, batch_size)

        # 3、构建评估器
        evaluator = Evaluator(self.model, train_generator, valid_generator, test_generator)

        # 4、按Epoch训练
        self.model.fit(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=self.epochs,
            callbacks=[evaluator]
        )
        # 5、训练结束 查看最终效果
        # self.model = load_model('bestModel.h5')
        # print(u'final test acc: %05f\n' % (evaluate(test_generator, self.model, "最终测试")))

    def predict(self, text: str):
        """单句预测
        """
        """predict label"""
        predict_token_ids, predict_segment_ids = [], []
        token_replace_dict = {98: 101, 99: 102}
        # tokenizer
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        token_ids = [token_replace_dict[i] if i in token_replace_dict else i for i in token_ids]
        predict_token_ids.append(token_ids)
        predict_segment_ids.append(segment_ids)
        # padding
        predict_token_ids = sequence_padding(predict_token_ids, length=maxlen)
        predict_segment_ids = sequence_padding(predict_segment_ids, length=maxlen)
        # predict
        result = self.model.predict([predict_token_ids, predict_segment_ids])

        label_indices_one = np.argmax(result[0], axis=1)
        label_prob_one = result[0][:, label_indices_one]
        label_indices_two = np.argmax(result[1], axis=1)
        label_prob_two = result[1][:, label_indices_two]
        label_indices_three = np.argmax(result[2], axis=1)
        label_prob_three = result[2][:, label_indices_three]
        predict_label_one = [self.label_index_map_one[index] for index in label_indices_one]
        predict_label_two = [self.label_index_map_two[index] for index in label_indices_two]
        predict_label_three = [self.label_index_map_three[index] for index in label_indices_three]
        # 处理结果

        return predict_label_one[0], label_prob_one[0], predict_label_two[0], label_prob_two[0], predict_label_three[0], \
               label_prob_three[0]

    def getBadCase(self, dataPath, sheetname, textCol, labeloneCol, labeltwoCol, labelthreeCol):
        xlsx_data = pd.read_excel(dataPath, sheet_name=sheetname)
        texts = xlsx_data[textCol]
        labeloneCols = xlsx_data[labeloneCol]
        labeltwoCols = xlsx_data[labeltwoCol]
        labelthreeCol = xlsx_data[labelthreeCol]
        newLabel1 = []
        newProb1 = []
        newLabel2 = []
        newProb2 = []
        newLabel3 = []
        newProb3 = []
        for i in tqdm(range(len(texts))):
            text = texts[i]
            label1, prob1, label2, prob2, label3, prob3 = self.predict(text)
            newLabel1.append(label1)
            newProb1.append(prob1)
            newLabel2.append(label2)
            newProb2.append(prob2)
            newLabel3.append(label3)
            newProb3.append(prob3)
        xlsx_data["newlabelone"] = newLabel1
        xlsx_data["newProb1"] = newProb1
        xlsx_data["newlabeltwo"] = newLabel2
        xlsx_data["newProb2"] = newProb2
        xlsx_data["newlabelthree"] = newLabel3
        xlsx_data["newlabel3"] = newProb3
        xlsx_data.to_excel("predictionRes.xlsx", index=False)


def evaluate(data, model, module: str = ""):
    total, right_label1, right_label2, right_label3, right, rightOneOnly, rightTwoOnly = 0., 0., 0., 0., 0., 0., 0.
    t = time()
    for x_true, y_true in data:
        # x_true  [batch_size, max_length]  [32, 450]
        # print(type(x_true))
        # print(x_true)
        y_pred_label1 = model.predict(x_true)[0].argmax(axis=1)
        y_pred_label2 = model.predict(x_true)[1].argmax(axis=1)
        y_pred_label3 = model.predict(x_true)[2].argmax(axis=1)
        # print(y_pred_label1, y_pred_label2)
        # print(y_true)
        y_true_label1 = y_true[0][:, 0]
        y_true_label2 = y_true[1][:, 0]
        y_true_label3 = y_true[2][:, 0]
        total += len(y_true_label1)

        right_label1 += (y_pred_label1 == y_true_label1).sum()
        right_label2 += (y_pred_label2 == y_true_label2).sum()
        right_label3 += (y_pred_label3 == y_true_label3).sum()
        rightOneOnly += ((y_pred_label1 == y_true_label1) & ~(y_pred_label2 == y_true_label2)).sum()
        rightTwoOnly += (~(y_pred_label1 == y_true_label1) & (y_pred_label2 == y_true_label2)).sum()
        right += ((y_pred_label1 == y_true_label1) & (y_pred_label2 == y_true_label2) & (
                    y_pred_label3 == y_true_label3)).sum()
        # print(y_pred_label1,y_true_label1)
        # print(y_pred_label2, y_true_label2)
        # print(right_label1 / total, right_label2 / total, right / total)
    # print(module + " Evaluate Average Elapse: %d ms" % int(round((time() - t) * 1000 / total)))
    return right_label1 / total if right_label1 > 0 else 0, \
           right_label2 / total if right_label2 > 0 else 0, \
           right_label3 / total if right_label3 > 0 else 0, \
           right / total if right > 0 else 0, \
           rightOneOnly / total if rightOneOnly > 0 else 0, \
           rightTwoOnly / total if rightTwoOnly > 0 else 0, \
           right_label3 / total if right_label3 > 0 else 0


class Evaluator(keras.callbacks.Callback):
    def __init__(self, model,
                 train_generator,
                 valid_generator,
                 test_generator, ):
        self.best_val_acc = 0.
        self.best_val_acc_one = 0.
        self.best_val_acc_two = 0.
        self.best_val_acc_three = 0.
        self.best_val_acc_onlyOne = 0.
        self.best_val_acc_onlyTwo = 0.
        self.best_val_acc_onlyThree = 0.

        self.model = model
        self.train_generator = train_generator
        self.valid_generator = valid_generator
        self.test_generator = test_generator

    def on_epoch_end(self, epoch, logs=None):
        val_acc_one, val_acc_two, val_acc_three, val_acc, val_acc_onlyOne, val_acc_onlyTwo, val_acc_onlyThree = evaluate(
            self.valid_generator, self.model, "验证集")
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_acc_one = val_acc_one
            self.best_val_acc_two = val_acc_two
            self.best_val_acc_three = val_acc_three
            self.best_val_acc_onlyOne = val_acc_onlyOne
            self.best_val_acc_onlyTwo = val_acc_onlyTwo
            self.best_val_acc_onlyThree = val_acc_onlyThree
            # model.save_weights('best_model.weights')
            self.model.save('./model_file/multi_baseline-{}.h5'.format(str(val_acc)))
        # todo 保存每一个epoch的模型 便于手动挑选
        # test_acc = evaluate(self.test_generator, self.model, "测试集")
        print(
            u'val_acc_one: %.5f,val_acc_two: %.5f,val_acc_three: %.5f,val_acc: %.5f, best_val_acc: %.5f, '
            u'best_val_acc_one: %.5f, best_val_acc_two: %.5f, best_val_acc_three: %.5f\n' % (
                val_acc_one, val_acc_two, val_acc_three, val_acc, self.best_val_acc, self.best_val_acc_one,
                self.best_val_acc_two, self.best_val_acc_three))

        # print(
        #     u'val_acc_onlyOne: %.5f,val_acc_onlyTwo: %.5f,best_val_acc_onlyOne: %.5f, best_val_acc_onlyTwo: %.5f\n' % (
        #         val_acc_onlyOne, val_acc_onlyTwo, self.best_val_acc_onlyOne, self.best_val_acc_onlyTwo))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-mode', '--mode', type=str, default='train', help='脚本执行模式: 训练 or 预测')
    parser.add_argument('-i', '--inpath', type=str, default='输入文件路径', help='')
    parser.add_argument('-o', '--outpath', type=str, default='输出文件路径', help='')
    # 获取相关参数
    args = parser.parse_args()
    mode = args.mode
    inpath = args.inpath
    outpath = args.outpath

    # getbadcase
    mode = 'badcase'

    inpath = "td20220720_filters.xlsx"
    sheetname = "Sheet1"
    colname = "msg"
    labelone = "labelone"
    labeltwo = "labeltwo"
    labelthree = "labelthree"
    model_path = ""
    if mode == 'train':
        # (self, model_path, _mode: str, filename: str, sheetname: str, text_column: str, label_column: str, *args,
        # **kwargs):
        # todo 待将这些固定参数替换为可变输入参数
        ptm_model = PtmModel('', mode, inpath, sheetname,
                             colname, labelone, labeltwo, labelthree)
        ptm_model.train()

    elif mode == 'predict':
        t0 = time()
        ptm_model = PtmModel(model_path,
                             mode, '', '', '', '')
        print(" 模型加载耗时 t0 : %d ms" % int(round((time() - t0) * 1000)))
        text1 = ""
        text2 = "如果价格太低我们可以不接单哈"
        print(text1 + '\t' + str(ptm_model.predict(text1)))
        print(text2 + '\t' + str(ptm_model.predict(text2)))
    else:
        ptm_model = PtmModel('./model_file/multi_baseline-0.3008849557522124.h5', 'predict', '', '', '', '', '', '')
        ptm_model.getBadCase(inpath, sheetname, colname, labelone, labeltwo, labelthree)