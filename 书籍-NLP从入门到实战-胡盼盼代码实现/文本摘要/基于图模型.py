# 基于图模型中的 TextRank，此方法将句子集合当作一张图，每个句子为图中的节点

import numpy as np
import pandas as pd
import csv
import nltk
import re
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx  # 用于概率图模型的构建
nltk.download("stopwords")
nltk.download("punkt")

def load_data(file_path,title):
    df = pd.read_csv("tennis_articles_v4.csv")
    return df[title]

def pre_data(data):
    sens = [sent_tokenize(s) for s in data]  # 对每一篇文本进行分句操作  list[list],list[i][j]表示第i个文本的第j个句子
    sens = [t for s in sens for t in s]
    # 加载词干提取器
    stemmer = nltk.stem.porter.PorterStemmer()
    # 加载停用词
    stopwords = set(nltk.corpus.stopwords.words("english"))
    # 句子预处理，主要包括字母小写化、去除标点符号、去除停用词
    def pre_sen(sen):
        sen = sen.lower()
        sen = re.sub(r"[^a-zA-Z]",r"",sen)
        pre_sen = [stemmer.stem(w) for w in sen if w not in stopwords] # 去除停用词并且进行词形还原
        assert len(pre_sen) != 0
        return pre_sen

    pre_sens = [pre_sen(sen) for sen in sens]  # 对每一个句子进行预处理
    assert len(pre_sens) == len(sens)
    return sens,pre_sens  # 因为是抽取式摘要生成，所以需要原文的信息

# 对句子相似度进行计算
def create_sim_mat(pre_sens,emb_file,emb_size=100):
    # 读取词向量文件，glove已训练好的词向量对句子进行编码
    emb = pd.read_csv(emb_file,sep=' ',header=None,quoting=csv.QUOTE_NONE,index_col=[0]) # index_col，取第一列作为index
    dict_word_emb = emb.T.to_dict('series')

    # 将句子转化为句向量
    sens_vec = []
    for s in pre_sens:
        # 句向量的值为句子中所有词向量的平均值
        if len(s) != 0:
            v = sum([dict_word_emb[w] for w in s]) / len(s)
        else:
            v = np.zeros((emb_size,))

        sens_vec.append(v)

    # 建立句子与句子间的相似度矩阵
    n = len(pre_sens)
    sim_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i != j :
                sim_mat[i][j] = cosine_similarity([sens_vec[i]],[sens_vec[j]])[0,0]

    return sim_mat

# 使用networkx将相似度矩阵转化为图模型，并且基于pagerank算法将句子按照重要性程序进行排序，
def summarize(sens,sim_mat,topK=10):
    nx_graph = nx.from_numpy_array(sim_mat) # 根据相似度矩阵建图
    scores = nx.pagerank(nx_graph) # 利用pagerank算法给句子(图中节点)评分
    ranked_sens = sorted(((scores[i],s) for i,s in enumerate(sens)) ,reverse=True) # 将句子按得分大小降序排序
    for i in range(topK):
        print(ranked_sens[i][1])

if __name__ == '__main__':
    file_path = "tennis_articles_v4.csv"
    title = "article_text"
    emb_file = "glove.6B.100d.txt" # 词向量文件路径
    data = load_data(file_path,title) # 加载数据
    sens,pre_sens = pre_data(data) # 处理数据
    sim_mat = create_sim_mat(pre_sens,emb_file)
    summarize(sens,sim_mat) # 获取摘要

