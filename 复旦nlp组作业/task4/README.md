### 任务四：基于LSTM+CRF的序列标注

用LSTM+CRF来训练序列标注模型：以Named Entity Recognition(命名实体识别)为例。

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、11章
   2. https://arxiv.org/pdf/1603.01354.pdf
   3. https://arxiv.org/pdf/1603.01360.pdf
   4. [LSTM+CRF 解析（原理篇）](https://zhuanlan.zhihu.com/p/97829287), [LSTM+CRF 解析（代码篇）](https://zhuanlan.zhihu.com/p/97858739)
2. 数据集：CONLL 2003，https://www.clips.uantwerpen.be/conll2003/ner/
3. 实现要求：Pytorch
4. 知识点：
   1. 评价指标：precision、recall、F1
   2. 无向图模型、CRF
   3. part-of-speech (POS) tagging: 词性标注
