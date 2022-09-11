# coding: UTF-8

from transformers import AdamW, get_linear_schedule_with_warmup
from preprocess import get_time_dif
from sklearn import metrics
import torch
import numpy as np
import logging
import time
import os
import torch.nn as nn
import torch.nn.functional as F

ce = torch.nn.CrossEntropyLoss()
kld = torch.nn.KLDivLoss(reduction="none")


def Rdroploss(p, q, labels, alpha=4):
    """配合R-Drop的交叉熵损失"""
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    kl_loss = (p_loss + q_loss) / 2
    ce_loss = 0.5 * (ce(p, labels) + ce(q, labels))

    return ce_loss + alpha * kl_loss


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


def eval(model, config, iterator, flag=False, numbers='test'):
    model.eval()
    total_loss = 0
    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    all_contexts = []
    with torch.no_grad():
        for batch, labels, context in iterator:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=labels)

            loss = outputs[0]
            logits = outputs[1]

            total_loss += loss
            true = labels.data.cpu().numpy()
            pred = torch.max(logits.data, 1)[1].cpu().numpy()
            all_labels = np.append(all_labels, true)
            all_preds = np.append(all_preds, pred)

            all_contexts.append(context)

    acc = metrics.accuracy_score(all_labels, all_preds)

    ## 在测试集评估时需要计算各种指标
    if flag:
        report = metrics.classification_report(all_labels, all_preds, target_names=config.label_list, digits=4,
                                               labels=range(len(config.label_list)))
        confusion = metrics.confusion_matrix(all_labels, all_preds)

        # 输出测试集的badcase
        all_cons = []
        import pandas as pd
        da = pd.DataFrame(columns=['text', 'gold', 'predict'])
        for a in all_contexts:
            all_cons.extend(a)
        da.text = all_cons
        da.gold = [config.label_list[idx] for idx in all_labels]
        da.predict = [config.label_list[idx] for idx in all_preds]
        da = da[da['gold']!=da['predict']]
        da.to_excel(f'./{config.saved_model_dir}/badcase_{numbers}.xlsx')

        return acc, total_loss / len(iterator), report, confusion

    return acc, total_loss / len(iterator)


def test(model, config, iterator, numbers='test'):
    model.load_state_dict(torch.load(config.saved_model))
    start_time = time.time()
    acc, loss, report, confusion = eval(model, config, iterator, flag=True, numbers=numbers)
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    config.logger.info(msg.format(loss, acc))
    config.logger.info("Precision, Recall and F1-Score...")
    config.logger.info(report)
    config.logger.info("Confusion Matrix...")
    config.logger.info(confusion)
    time_dif = get_time_dif(start_time)
    config.logger.info(f"Time usage:{time_dif}")


def train(model, config, train_iterator, dev_iterator, test_iterator):
    model.train()
    start_time = time.time()

    logger = get_logger(os.path.join(config.saved_model_dir, 'logs' + str(start_time) + '.txt'))
    config.logger = logger

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_optimizer = model.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = len(train_iterator) * config.num_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=t_total)

    total_batch = 0
    last_improve = 0
    break_flag = False
    best_dev_loss = float('inf')
    best_dev_acc = - float('inf')
    for epoch in range(config.num_epochs):
        logger.info("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        for _, (batch, labels, contxt) in enumerate(train_iterator):

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=labels)

            loss = outputs[0]
            logits = outputs[1]

            # rdrop
#             outputs1 = model(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 token_type_ids=batch["token_type_ids"],
#                 labels=labels)
#             logits1 = outputs[1]
#             loss = Rdroploss(logits, logits1, labels, alpha=4)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if total_batch % config.log_batch == 0:
                true = labels.data.cpu()
                pred = torch.max(logits.data, 1)[1].cpu()
                acc = metrics.accuracy_score(true, pred)
                dev_acc, dev_loss = eval(model, config, dev_iterator)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    torch.save(model.state_dict(), config.saved_model)
                    improve = "*"
                    last_improve = total_batch
                # if dev_acc > best_dev_acc:
                #     best_dev_acc = dev_acc
                #     torch.save(model.state_dict(), config.saved_model)
                #     improve = "*"
                #     last_improve = total_batch
                else:
                    improve = ""

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Batch Train Loss: {1:>5.2}, Batch Train Acc: {2:>6.2%}, Val Loss: {3:>5.2}, Val Acc: {4:>6.2%}, Time: {5} {6}'
                logger.info(msg.format(total_batch, loss.item(), acc, dev_loss, dev_acc, time_dif, improve))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                logger.info("No improvement for a long time, auto-stopping...")
                break_flag = True
                break
        if break_flag:
            break

    test(model, config, test_iterator, numbers='test')






## multi-task training setting

def get_loss_multi(y_pred, y_true):
    return (nn.functional.cross_entropy(y_pred[0], y_true[0]) + nn.functional.cross_entropy(y_pred[1], y_true[1])) / 2


def evaluate_multi(y_pred, labels):
    total, right_label2, right_label3, right, rightOneOnly, rightTwoOnly = 0., 0., 0., 0., 0., 0.,

    y_pred_label2 = y_pred[0].argmax(axis=1)
    y_pred_label3 = y_pred[1].argmax(axis=1)

    y_true_label2 = labels[0]
    y_true_label3 = labels[1]
    total += len(y_true_label2)

    right_label2 += (y_pred_label2 == y_true_label2).sum()
    right_label3 += (y_pred_label3 == y_true_label3).sum()
    right += ((y_pred_label2 == y_true_label2) & (y_pred_label3 == y_true_label3)).sum()

    return right_label2 / total if right_label2 > 0 else 0, \
           right_label3 / total if right_label3 > 0 else 0, \
           right / total if right > 0 else 0,


def eval_multi(model, config, iterator, flag=False):
    model.eval()

    accs = 0
    one, two = 0, 0
    total_loss = 0
    with torch.no_grad():
        for batch, labelones, labbeltwos in iterator:
            outputs = model(batch)
            labels = [labelones, labbeltwos]
            total_loss += get_loss_multi(outputs, labels)
            true = [label.data.cpu() for label in labels]
            pred = [logits.data.cpu() for logits in outputs]
            rightone, righttwo, acc = evaluate_multi(pred, true)
            accs += acc
            one += rightone
            two += righttwo

        print(one / len(iterator), two / len(iterator))
    return accs / len(iterator), total_loss / len(iterator)


def test_multi(model, config, iterator):
    model.load_state_dict(torch.load(config.saved_model))
    start_time = time.time()
    acc, loss = eval_multi(model, config, iterator, flag=True)
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    config.logger.info(msg.format(loss, acc))
    time_dif = get_time_dif(start_time)
    config.logger.info(f"Time usage:{time_dif}")


def train_multitask(model, config, train_iterator, dev_iterator, test_iterator):
    model.train()
    start_time = time.time()

    logger = get_logger(os.path.join(config.saved_model_dir, 'logs' + str(start_time) + '.txt'))
    config.logger = logger

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_optimizer = model.named_parameters()
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    t_total = len(train_iterator) * config.num_epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=t_total)

    total_batch = 0
    last_improve = 0
    break_flag = False
    best_dev_loss = float('inf')
    for epoch in range(config.num_epochs):
        logger.info("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        for _, (batch, labelones, labeltwos) in enumerate(train_iterator):

            labels = [labelones, labeltwos]
            outputs = model(batch)

            loss = get_loss_multi(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if total_batch % config.log_batch == 0:
                true = [label.data.cpu() for label in labels]
                pred = [logits.data.cpu() for logits in outputs]
                _, _, acc = evaluate_multi(pred, true)
                dev_acc, dev_loss = eval_multi(model, config, dev_iterator)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    torch.save(model.state_dict(), config.saved_model)
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Batch Train Loss: {1:>5.2}, Batch Train Acc: {2:>6.2%}, Val Loss: {3:>5.2}, Val Acc: {4:>6.2%}, Time: {5} {6}'
                logger.info(msg.format(total_batch, loss.item(), acc, dev_loss, dev_acc, time_dif, improve))
                model.train()

            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                logger.info("No improvement for a long time, auto-stopping...")
                break_flag = True
                break
        if break_flag:
            break

    test_multi(model, config, test_iterator)
