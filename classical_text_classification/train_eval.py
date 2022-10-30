# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
import logging
import os
from Contract_loss import SupConLoss

ce = torch.nn.CrossEntropyLoss()
kld = torch.nn.KLDivLoss(reduction="none")
suploss = SupConLoss(temperature=0.1, scale_by_temperature=True)

def Rdroploss(y_pred, y_true, alpha=4):
    """配合R-Drop的交叉熵损失"""

    loss1 = ce(y_pred, y_true)
    loss2 = kld(torch.log_softmax(y_pred[::2], dim=1), y_pred[1::2].softmax(dim=-1)) + \
            kld(torch.log_softmax(y_pred[1::2], dim=1), y_pred[::2].softmax(dim=-1))

    return loss1 + torch.mean(loss2) / 4 * alpha




def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w", encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass



def train(config, model, train_iter, dev_iter, test_iter):

    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    ## 冻结embedding,不训练
    # for param in model.named_parameters():
    #     if param[0] in ['embedding']:
    #         param[1].requires_grad = False
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    dev_best_acc = - float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    ## 记录训练过程日志
    writer = SummaryWriter(log_dir=config.log_path)
    logger = get_logger(os.path.join(config.log_path, 'logs-remark.txt'))
    config.logger = logger
    config.logger.info(f'====== Using rdrop = {config.rdrop} ======')

    for epoch in range(config.num_epochs):
        config.logger.info('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels, _) in enumerate(train_iter):

            # outputs, sample_feature = model(trains)
            outputs = model(trains)
            if config.rdrop == False:
                loss = F.cross_entropy(outputs, labels)
            else:
                loss = Rdroploss(outputs, labels)

            # loss_sup = suploss(sample_feature, labels=labels)
            # loss = loss + loss_sup

            loss.backward()

            # 对抗训练
            model.attack(0.3)
            loss_adv =  Rdroploss(model(trains), labels)
            loss_adv.backward(retain_graph=True)
            model.restore()

            # 梯度更新
            optimizer.step()
            model.zero_grad()

            if total_batch % 5 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                config.logger.info(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1

            if total_batch - last_improve > config.require_improvement:
                config.logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break

        if flag:
            torch.save(model.state_dict(), config.save_path.replace('.ckpt', f'-acc{dev_best_acc}.ckpt'))
            break

    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    config.logger.info(msg.format(test_loss, test_acc))
    config.logger.info("Precision, Recall and F1-Score...")
    config.logger.info(test_report)
    config.logger.info("Confusion Matrix...")
    config.logger.info(test_confusion)
    time_dif = get_time_dif(start_time)
    config.logger.info(f"Time usage:{time_dif}")


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    all_contexts = []
    with torch.no_grad():
        for texts, labels, context in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)

            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            all_contexts.append(context)


    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4, labels=range(len(config.class_list)))
        confusion = metrics.confusion_matrix(labels_all, predict_all)

        all_cons = []
        import pandas as pd
        da = pd.DataFrame(columns=['text', 'gold', 'predict'])
        for a in all_contexts:
            all_cons.extend(a)
        da.text = all_cons
        da.gold = labels_all
        da.predict = predict_all
        # da = da[da['gold']!=da['predict']]

        da.to_excel(f'./{config.log_path}/badcase.xlsx')

        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)



def predict(config, model, test_iter):
    # test
    def evaluate(config, model, data_iter, test=False):
        model.eval()
        predict_all = np.array([], dtype=int)
        all_contexts = []
        with torch.no_grad():
            for texts, context in data_iter:
                outputs = model(texts)
                predicts = torch.max(outputs.data, 1)[1].cpu().numpy()
                predict_all = np.append(predict_all, predicts)
                all_contexts.append(context)

        labelmap = {idx:d for idx, d in enumerate(config.class_list)}
        final_all = []
        for p in predict_all:
            final_all.append(labelmap[p])

        return final_all

    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    result = evaluate(config, model, test_iter, test=True)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    return result


