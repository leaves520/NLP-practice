from gensim.models import word2vec
import jieba

raw_texts = ["你站在桥上看风景","看风景的人在楼上看你","明月装饰了你的窗子","你装饰了别人的梦"]
texts = [[word for word in jieba.cut(text,cut_all = True)] for text in raw_texts] # 对raw_texts中每一个句子进行分词

# model训练
# sentences:用于训练词向量的语料，Size：词向量的维度，window：上下文窗口大小，sg：0 cnow ，1 skip-gram, min_count:需要计算词向量的最小词频
# iter:随机梯度下降法中的迭代最大次数，默认5. alpha:随机梯度下降法迭代的初始步长，默认0.025
model = word2vec.Word2Vec(sentences=texts,size = 5,window=3,sg=1,negative=5,min_count=1,iter=5,alpha=0.025)

print(model.wv["你"]) # 输出某个词的词向量
print(model.wv.similarity("窗子","楼上")) # 输出两个词的相似性
print(model.wv.similar_by_word("你",topn=3)) # 输出某个词的top相似性

