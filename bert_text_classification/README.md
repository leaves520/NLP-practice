### 环境

python 3.7

torch 1.7

tqdm

sklearn

transformers 4.8.1



### 预训练 BERT 模型

从 huggingface 官网上下载 bert-base-chinese 模型权重、配置文件和词典到 pretrained_bert 文件夹中，下载地址：https://huggingface.co/bert-base-chinese/tree/main



#### 文本分类模型训练：

```shell
python main.py --mode train --data_dir ./data --pretrained_bert_dir ./pretrained_bert
```



#### Demo演示

文本分类 demo 展示：

```shell
python main.py --mode demo --data_dir ./data --pretrained_bert_dir ./pretrained_bert
```



#### 模型预测

对 data 文件夹下的 input.txt 中的文本进行分类预测：

```shell
python main.py --mode predict --data_dir ./data --pretrained_bert_dir ./pretrained_bert --input_file ./data/input.txt
```

