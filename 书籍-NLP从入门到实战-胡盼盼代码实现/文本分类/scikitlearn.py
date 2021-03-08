from sklearn.datasets import fetch_20newsgroups
# 新闻数据，属于多分类问题
categories = ["alt.atheism","talk.religion.misc","sci.space"]
data_train = fetch_20newsgroups(subset='train',shuffle=True,categories=categories)
data_test = fetch_20newsgroups(subset="test",shuffle=True,categories=categories)

# # 使用词袋模型把文本转为向量，一般有两种技术：CountVectorizer和TfidfVectorizer
# # 1、使用CountVectorizer, 这个是统计每个句子，词典中单词出现的次数 (亦即频次)表示，长度为字典
# from sklearn.feature_extraction.text import CountVectorizer
# count_vect = CountVectorizer()
# X_train = count_vect.fit_transform(data_train.data)
# print(count_vect.vocabulary_)

# 2、使用TfidfVectorizer, 综合了词在某一文本出现的频次以及所有在文本中的出现频次两衡量词的重要性
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()
X_train = tfidf_vect.fit_transform(data_train.data)
X_test = tfidf_vect.transform(data_test.data)
Y_train = data_train.target

# 训练和预测
# 1、使用朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,f1_score
clf = MultinomialNB(alpha=0.1)
clf.fit(X_train,Y_train)
predicted = clf.predict(X_test)
print("naive_bayes:")
print("F1_score:{}".format(f1_score(data_test.target,predicted,average="macro")))
print("accuracy_score:{}".format(accuracy_score(data_test.target,predicted)))

# 2、使用决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score
clf = DecisionTreeClassifier(criterion="entropy")
clf.fit(X_train,Y_train)
predicted = clf.predict(X_test)
print("decision_tree:")
print("f1_score:{}".format(f1_score(data_test.target,predicted,average='macro')))
print("accuracy_score:{}".format(accuracy_score(data_test.target,predicted)))

# 3、使用多层感知机
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score
clf = MLPClassifier(solver="adam",learning_rate="constant",learning_rate_init=0.01,max_iter=500,alpha=0.01)
clf.fit(X_train,Y_train)
predicted = clf.predict(X_test)
print("neural_network:")
print("f1_score:{}".format(f1_score(data_test.target,predicted,average='macro')))
print("accuracy_score:{}".format(accuracy_score(data_test.target,predicted)))


# 多分类问题下F1_score计算
'''
micro: 对多个混淆矩阵的各项分别进行加和，基于此得到查准率和查全率进行F1计算
macro: 基于每个混淆矩阵得到的查准率和查全率计算多个F1值，之后进行求和平均
'''

# 提高模型性能可考虑以下方向
'''
1.文本预处理，可以通过可视化了解数据概况以及尝试更多的文本表征方式
2.模型本身，可用利用网格搜索来寻求更好的参数
3.模型融合：bagging和boosting、 随机森林和GBDT、xgboost、lightGBM
'''