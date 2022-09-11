import functools
import warnings

import numpy as np

warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt



def buliddata(data):
    assert isinstance(data, pd.DataFrame)
    def concatmsg(text):
        ts = text.split('<ctrip>')
        msg = []
        for idx, t in enumerate(ts):
            msg.append(t.split('<=>')[-1])

        # return ','.join(msg)

        # 过滤转人工话术前的一些干扰文字
        msg = ','.join(msg)
        try:
            idx = msg.index('转人工')
            msg = msg[idx:]
        except:
            pass

        return msg


    def writetxt(filepath, data):
        with open(filepath, encoding='utf8', mode='w') as f:
            for d in data:
                note = '\t'.join(d)
                f.write(note + '\n')


    def getlabels(t, maplabel):
        return str(maplabel[t])


    def getlevellabels(t):  # 过滤2级标签 <= 500个的
        return t.split(' ')[1]

    da = data[['callid','callStatus', 'word']]
    da['tmp'] = da['callStatus']
    da.rename(columns={'callStatus':'final_label'},inplace=True)
    da['msg'] = da['word'].apply(concatmsg)

    da['final_label'] = da['final_label'].apply(getlevellabels)

    labels = da.final_label.unique().tolist()
    maplabel = {la: idx for idx, la in enumerate(labels)}
    getlabels = functools.partial(getlabels, maplabel=maplabel)
    da['labels'] = da['final_label'].apply(getlabels)

    t = da['labels'].value_counts().to_frame().reset_index()
    valid_list = t[t['labels'] >= 500]['index'].to_list() # 筛选labels数目大于50的
    da = da[da['labels'].isin(valid_list)]
    # X_data, test  = train_test_split(da, test_size=0.2, random_state=42)
    # train, val = train_test_split(X_data, test_size=len(test), random_state=42)

    da['callStatus'] = da['tmp']
    da.to_excel('./badcase/租车IM售前_filters_sessionid.xlsx', index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    data1 = pd.read_excel('./租车IM售前_13k.xlsx')
    buliddata(data1)