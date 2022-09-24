### 环境

python 3.7

torch 1.7

tqdm

sklearn

transformers 4.8.1



### 预训练 BERT 模型

从 huggingface 官网上下载 bert-base-chinese 模型权重、配置文件和词典到 pretrained_bert 文件夹中，下载地址：https://huggingface.co/bert-base-chinese/tree/main



python main.py --mode train --data_dir ./data --pretrained_bert_dir ./pretrained_bert


