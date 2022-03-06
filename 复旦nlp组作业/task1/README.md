### 任务一：基于机器学习的文本分类

实现基于logistic/softmax regression的文本分类

1. 参考
   1. [文本分类](../文本分类.md)
   2. 《[神经网络与深度学习](https://nndl.github.io/)》 第2/3章
   3. https://github.com/Alic-yuan/nlp-beginner-finish
2. 数据集：
   1. 采用gaussic中文数据集，https://github.com/gaussic/text-classification-cnn-rnn
   2. 网盘链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud (下载文件中的cnews.train.txt，放到dataset目录下即可，由于本地机器没计算资源，因此选取部分数据做实验)
   3. 中文常用停用词下载链接: https://github.com/goto456/stopwords
3. 实现要求：NumPy
4. 需要了解的知识点：
   1. 文本特征表示：Bag-of-Word，N-gram (实验仅仅使用了tf-idf的BOW， 且没有使用词频截断，因此文本的特征向量十分稀疏)
   2. 分类器：logistic/softmax  regression，损失函数、（随机）梯度下降、特征选择
   3. 数据集：训练集/验证集/测试集的划分
   4. 使用jieba对文本进行分词
5. 实验：
   1. 分析不同的特征、损失函数、学习率对最终分类性能的影响
   2. shuffle 、batch、mini-batch 


