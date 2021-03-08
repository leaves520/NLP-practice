import pandas as pd
import nltk # 自然语言、文本预处理常用的语料库以及工具
from nltk.tokenize import sent_tokenize  # 分句
import re
nltk.download("stopwords") # 下载常用停词
nltk.download("punkt")  # 下载分词模型

def load_data(file_path,title):
    df = pd.read_csv(file_path)
    return df[title]

def pre_data(data):
    sens = [sent_tokenize(s) for s in data] # 对每一篇文本进行分句，统一放在一个数组里，list[list]
    sens = [t for s in sens for t in s]  # list

    # 加载词干提取器
    stemmer = nltk.stem.porter.PorterStemmer()
    # 加载停用词
    stopwords = set(nltk.corpus.stopwords.words("english"))

    # 句子预处理，主要包括字母小写化、去除停用词、去除标点符号
    def pre_sen(sen):
        sen = sen.lower()
        sen = re.sub(r"[^a-zA-Z]",r" ",sen)
        sen = nltk.tokenize.word_tokenize(sen)
        pre_sen = [stemmer.stem(w) for w in sen if w not in stopwords]
        # print(pre_sen)
        # assert len(pre_sen) != 0
        return pre_sen

    pre_sens = [pre_sen(sen) for sen in sens]

    assert len(pre_sens) == len(sens)
    return sens,pre_sens  # 由于最终需要输出源文中的句子作为摘要，因此同时返回未处理过的以及处理过的句子集合


def topNwords(pre_sens,n=50):
    words = [w for s in pre_sens for w in s ]
    word_fre = nltk.FreqDist(words) # 对每一个句子进行单词统计
    print(word_fre)
    # 对字典按照values进行降序排序，然后取相应的前n个单词(高频单词)，这句写得漂亮
    topN_words = [w[0] for w in sorted(word_fre.items(),key=lambda d: d[1],reverse=True)][:n]
    return topN_words

# 根据句子中的关键词的信息计算其重要性
def sen_score(sen,topN_words,cluster_threshold):
    '''
    :param sen: 句子
    :param topN_words: 关键词表
    :param cluster_threshold: # 簇的阈值
    '''

    #计算每个句子的得分
    # 标记每个句子中出现topN中单词的位置
    word_idx = []
    for w in topN_words:
        try:
            word_idx.append(sen.index(w)) # 对每个句子找相应的topN单词的下标
        except:
            pass

    word_idx.sort() # 排序，保证后面按单词顺序去处理句子
    # 如果句子中不存在topN中的单词，得分为0
    if len(word_idx) == 0:
        return 0

    # 根据句子中topN单词出现的位置将句子转化为簇的集合形式
    clusters = []
    cluster = [word_idx[0]]
    i = 1
    while i < len(word_idx):
        # 当topN单词间的距离小于一定的阈值时，这些单词以及其之间非topN单词共同形成簇
        if word_idx[i] - word_idx[i-1] < cluster_threshold:
            cluster.append(word_idx[i])
        else:
            clusters.append(cluster)
            cluster = [word_idx[i]]

        i += 1

    clusters.append(cluster)

    # 计算每个句子中所有簇的得分，最高分为句子的得分
    max_score = 0
    for c in clusters:
        words_important = len(c) # 每个簇中topN单词的个数
        words_total = c[-1] - c[0] + 1 # 每个簇中所有单词的个数(包括topN之间的非top单词)
        # 得分计算公式
        score = words_important**2 / words_total # 关键词越密集，句子重要程度越高
        if score > max_score:
            max_score = score

    return max_score # 返回句子的得分

if __name__ == '__main__':
    file_path = "tennis_articles_v4.csv"
    title = "article_text"
    cluster_threshold = 5 # 设置topN语句间的距离阈值
    topK = 10 # 设置摘要句子的个数
    data = load_data(file_path,title) # 加载数据
    sens,pre_sens = pre_data(data) # 处理数据
    topN_words = topNwords(pre_sens) # 计算topN

    scores = []
    for i,pre_sen in enumerate(pre_sens):
        score = sen_score(pre_sen,topN_words,cluster_threshold)
        sen =sens[i]
        scores.append((score,sen))

    scores = sorted(scores,reverse=True)

    for i in range(topK):
        print(scores[i][1])
