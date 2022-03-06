import torch
import torch.optim as optim
from datahelper import CoNLL_DataSet
from model import CnnLstmCrf


EPOCHS = 100
batch_size = 10
lr = 1e-3

dataset = CoNLL_DataSet(sample=0.01)
print(dataset.vocabulary.size)
trn, val, test = dataset.CreateTorchDataset()
trn_dataloader, val_dataloader, test_dataloader = \
    dataset.CreateDataLoader(trn, shuffle=True, batch_size=batch_size), \
    dataset.CreateDataLoader(val, batch_size=4), dataset.CreateDataLoader(test, batch_size=4)
word_size, char_size, tag_size = dataset.vocabulary.size.values()

model = CnnLstmCrf(char_size=char_size,
                   word_size=word_size,
                   tag_nums=tag_size,
                   start_idx=dataset.vocabulary.label2idx('<S>'),
                   end_idx=dataset.vocabulary.label2idx('<E>'),
                   char_dim=30, char_max_length=dataset.MaxCharLen, cnn_WindowSize=3, cnn_FilterNums=30,
                   word_dim=100, hidden_state=100, num_layers=1,
                   dropout=0.5)

optimer = optim.SGD(model.parameters(),lr=lr, momentum=0.9, weight_decay=0.05)


def get_entity(tags):
    entity = []
    prev_entity = "O"
    start = -1
    end = -1
    for i, tag in enumerate(tags):
        if tag[0] == "O":
            if prev_entity != "O":
                entity.append((start, end))
            prev_entity = "O"
        if tag[0] == "B":
            if prev_entity != "O":
                entity.append((start, end))
            prev_entity = tag[2:]
            start = end = i
        if tag[0] == "I":
            if prev_entity == tag[2:]:
                end = i
    return entity  # list(tuple): return the entity from start_idx to end_idx



for e in range(EPOCHS):
    model.train()
    for i, (word_x, char_x, y, mask) in enumerate(trn_dataloader):
        loss = model.compute_loss(word_x, char_x, y, mask)
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
        correct_num = 0
        predict_num = 0
        truth_num = 0
        with torch.no_grad():
            for word_x, char_x, y, mask in val_dataloader:
                pred_path = model.predict(word_x, char_x, mask)

                ground_truth = [y_i[mask[i]==1].tolist() for i, y_i in enumerate(y)]
                for hyp, gold in zip(pred_path, ground_truth):
                    assert len(hyp) == len(gold)
                    hyp = list(map(lambda x: dataset.vocabulary.idx2label(x), hyp))
                    gold = list(map(lambda x: dataset.vocabulary.idx2label(x), gold))
                    predict_entities = get_entity(hyp)
                    gold_entities = get_entity(gold)
                    correct_num += len(set(predict_entities) & set(gold_entities))
                    predict_num += len(set(predict_entities))
                    truth_num += len(set(gold_entities))

            # calculate F1 on entity
            precision = correct_num / predict_num if predict_num else 0
            recall = correct_num / truth_num if truth_num else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

            print('epochs{} ==> f1:{} | precision:{} | recall:{}'.format(e, f1, precision, recall))

