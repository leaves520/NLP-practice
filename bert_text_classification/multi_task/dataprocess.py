'''
    处理成multitask能识别的xlsx格式。
'''

import pandas as pd

def to_xlsx_multi(data):
    def concatmsg(text):
        ts = text.split('<ctrip>')
        msg = []
        for idx, t in enumerate(ts):
            if idx < 4:
                msg.append(t.split('<=>')[-1])
            else:
                break

        return ','.join(msg)

    def getlevel_one(t):
        return t.split(' ')[0]

    def getlevel_two(t):
        return t.split(' ')[1]

    def getlevel_three(t):
        return t.split(' ')[2]



    da = data[['callStatus','word']]
    da.rename(columns={'callStatus':'final_label'},inplace=True)
    da['msg'] = da['word'].apply(concatmsg)

    da['labelone'] = da['final_label'].apply(getlevel_one)
    da['labeltwo'] = da['final_label'].apply(getlevel_two)
    da['labelthree'] = da['final_label'].apply(getlevel_three)

    labels_twos = da.labeltwo.unique().tolist()
    labels_threes = da.labelthree.unique().tolist()

    def writetxt(filepath, data):
        with open(filepath, encoding='utf8', mode='w') as f:
            for d in data:
                note = '\t'.join(d)
                f.write(note + '\n')

    writetxt(filepath='./label_two.txt', data=[[l] for l in labels_twos])
    writetxt(filepath='./label_three.txt', data=[[l] for l in labels_threes])

    tmp = da[['msg','labelone','labeltwo','labelthree']]
    tmp.to_excel('./td20220720_filters.xlsx',index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    data1 = pd.read_excel('./租车IM售前_filters.xlsx')
    to_xlsx_multi(data1)