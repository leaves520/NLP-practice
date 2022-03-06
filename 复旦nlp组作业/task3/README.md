### 任务三：基于注意力机制的文本匹配

输入两个句子判断，判断它们之间的关系。参考[ESIM](https://arxiv.org/pdf/1609.06038v3.pdf)（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第7章
   2. Reasoning about Entailment with Neural Attention <https://arxiv.org/pdf/1509.06664v1.pdf>
   3. Enhanced LSTM for Natural Language Inference <https://arxiv.org/pdf/1609.06038v3.pdf>
   4. https://github.com/pengshuang/Text-Similarity/blob/master/models/ESIM.py
   5. [ESIM整体模型结构图](ESIM.jpg)
2. 数据集：https://nlp.stanford.edu/projects/snli/
3. 实现要求：Pytorch
4. 知识点：
   1. 注意力机制
   2. token2token attetnion
   3. 利用nltk对文本数据进行预处理, 主要使用分句，分词，停用词表，词还原成词干或词根
      1. from nltk.tokenize import sent_tokenize, word_tokenize 
      2. from nltk.corpus import stopwords
      3. from nltk.stem.wordnet import WordNetLemmatizer
   4. [使用Glove预训练语料](https://github.com/stanfordnlp/GloVe)
   5. [nltk—常用文本处理流程](https://blog.csdn.net/qq_34464926/article/details/82745413)
   6. 可参考中文文本处理流程的[py代码](https://github.com/Alic-yuan/nlp-beginner-finish/blob/master/task3/data.py)
