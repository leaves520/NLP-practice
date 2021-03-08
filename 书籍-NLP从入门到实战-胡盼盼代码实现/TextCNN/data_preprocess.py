# import pandas as pd
#
# data = pd.read_csv('data/movie_review_raw.csv')
# data_positive = data[data['tag']=='pos']
# data_positive = data_positive.sample(5000,replace=False)
#
# data_negative = data[data['tag']=='neg']
# data_negative = data_negative.sample(5000,replace=False)
#
# data_positive.to_csv('data/pos_5k.csv',index=False)
# data_negative.to_csv('data/neg_5k.csv',index=False)

# # 文本预处理
# import pickle
# import pandas as pd
# import re
# from nltk.tokenize import word_tokenize
# import nltk
# stemmer = nltk.stem.porter.PorterStemmer() # 词形还原
# stopwords = set(nltk.corpus.stopwords.words("english")) # 停词表
#
# def process(df,title):
#     def re_split(s):
#         s = s.lower()
#         s = re.sub(r'[^a-zA-Z]',r' ',s)
#         s = word_tokenize(s)
#         s = [stemmer.stem(w) for w in s if w not in stopwords]
#         return s
#
#     all_sen = df[title].to_list()
#     all_sen = [re_split(sen) for sen in all_sen]
#     assert len(all_sen) == len(df)
#
#     word_set = set([word for sen in all_sen for word in sen] + ['<PAD>'])
#     word2index = dict([(word,i) for i,word in enumerate(word_set)])
#
#     with open('data/word_dict','wb') as f:
#         pickle.dump(word2index,f)
#
#     return pd.DataFrame({'tokenized':all_sen})
#
#
# data_pos = pd.read_csv("data/pos_5k.csv")
# data_neg = pd.read_csv('data/neg_5k.csv')
# all_data = pd.concat([data_pos,data_neg],axis=0).reset_index(drop=True) # drop =True，去除原来的index
#
# token = process(all_data,'text')
# data = pd.concat([all_data,token],axis=1)
#
# data.to_csv('data/processed.csv',index=False)


# 将文本数据转为word index list 代表文本
import pandas as pd
import pickle
data = pd.read_csv('data/processed.csv')
with open('data/word_dict','rb') as f:
    word_dict = pickle.load(f)

def word2index(s,word_dict,max_length = 10):
    from ast import literal_eval # 由于pandas保存list后会转为str，再次读取时要转list
    s = literal_eval(s)
    l = len(s)
    if l >= max_length:
        return [word_dict[w] for w in s[:max_length]]
    else:
        tmp_s = s + ['<PAD>']*(max_length-l)
        return [word_dict[w] for w in tmp_s]

data['word_index'] = data['tokenized'].apply(lambda x : word2index(x,word_dict))
data['label'] = data['tag'].apply(lambda x: 1 if x=='pos' else 0)

data = data[['word_index','label']]
data.to_csv('data/train_test.csv',index = False)

