import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class TextModel():
    def __init__(self, model, tokenizer, device, maxlen=512, label_map_two=''):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.maxlen = maxlen
        self.label_map_two = label_map_two
        self.label_index_map_two = {v: k for k, v in label_map_two.items()}

    def predict(self, text: str):
        """单句预测
        """
        """predict label"""
        inputs = self.tokenizer.batch_encode_plus(
                        [text],
                        padding="max_length",
                        max_length=self.maxlen,
                        truncation="longest_first",
                        return_tensors="pt",
                        return_token_type_ids=True,
                        return_attention_mask=True)

        inputs = inputs.to(self.device)
        result = self.model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            token_type_ids=inputs["token_type_ids"],)[0]  # shape: [1, num_class]

        result = F.softmax(result, dim=1)  # logits -> 概率
        result = result.cpu().detach().numpy()
        label_indices_two = np.argmax(result, axis=1)
        label_prob_two = result[:, label_indices_two]
        predict_label_two = [self.label_index_map_two[index] for index in label_indices_two]

        return predict_label_two[0], label_prob_two[0]


    def getBadCase(self, dataPath, sheetname, textCol):
        xlsx_data = pd.read_excel(dataPath, sheet_name=sheetname)
        texts = xlsx_data[textCol]
        newLabel2 = []
        newProb2 = []

        for i in tqdm(range(len(texts))):
            text = texts[i]
            label2, prob2= self.predict(text)

            newLabel2.append(label2)
            newProb2.append(prob2)

        xlsx_data["newlabeltwo"] = newLabel2
        xlsx_data["newProb2"] = newProb2
        xlsx_data.to_excel("predictionRes.xlsx", index=False)


    def predictBatch(self, texts):
        """
            多句batch预测
        """
        """predict label"""
        inputs = self.tokenizer.batch_encode_plus(
                        texts,
                        padding="max_length",
                        max_length=self.maxlen,
                        truncation="longest_first",
                        return_tensors="pt",
                        return_token_type_ids=True,
                        return_attention_mask=True)

        inputs = inputs.to(self.device)
        result = self.model(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            token_type_ids=inputs["token_type_ids"],)[0]  # shape: [1, num_class]

        result = F.softmax(result, dim=1)  # logits -> 概率
        result = result.cpu().detach().numpy()
        label_indices_two = np.argmax(result, axis=1)
        label_prob_two = result[:, label_indices_two]
        predict_label_two = [self.label_index_map_two[index] for index in label_indices_two]

        return predict_label_two, label_prob_two


    def getBadCaseBatch(self, dataPath, sheetname, textCol, batchsize=8):
        xlsx_data = pd.read_excel(dataPath, sheet_name=sheetname)
        texts = xlsx_data[textCol]
        newLabel2 = []
        newProb2 = []

        numebers = len(texts) // batchsize - 1
        for i in tqdm(range(numebers)):
            if i < numebers - 1:
                text = texts[i*batchsize:(i+1)*batchsize]
            else:
                text = texts[i*batchsize:]

            label2, prob2= self.predictBatch(text.tolist())

            newLabel2.extend(label2)
            newProb2.extend(prob2)

        assert len(xlsx_data) == len(newProb2) == len(newLabel2)

        xlsx_data["newlabeltwo"] = newLabel2
        xlsx_data["newProb2"] = newProb2
        xlsx_data.to_excel("predictionRes.xlsx", index=False)

if __name__ == '__main__':
    ## 模型设置
    base_model ='./pretrained_bert/bert_base_chinese'
    saved_model = './after_data_cars_single/bert_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    maxlen=512
    label_map_two = {la.strip(): idx for idx, la in enumerate(open('./after_data_cars_single/label.txt', encoding='utf-8-sig'))}

    ## 数据格式
    dataPath = './after_data_cars_single/cars_after_two.xlsx'
    sheetname = 'Sheet1'
    textcol = 'msg'


    tokenizer = AutoTokenizer.from_pretrained(base_model)
    bert_config = AutoConfig.from_pretrained(base_model, num_labels=len(label_map_two))
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(base_model, "pytorch_model.bin"),
        config=bert_config)

    model.load_state_dict(torch.load(saved_model, map_location=torch.device('cpu')))
    model.eval()
    model.to(device)

    predictModel = TextModel(model=model,tokenizer=tokenizer,
                             device=device, maxlen=maxlen, label_map_two=label_map_two)

    predictModel.getBadCase(dataPath=dataPath, sheetname=sheetname, textCol=textcol)
    # predictModel.getBadCaseBatch(dataPath=dataPath, sheetname=sheetname, textCol=textcol)

