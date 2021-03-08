import numpy as np
import jieba
import pickle
class DataParse:
    def __init__(self,DefaultConfig):
        self.opts = DefaultConfig

    def data_read(self):
        # 读取数据
        with open(self.opts.data_path,'r',encoding='utf8') as f:
            data_raw = f.readlines()

        # jieba对句子进行中文分词，将 \t , \n作为句子的开始以及结束标志符号，将句子以数组的方式保存
        texts = [jieba.lcut('\t' + line[4:]) for line in data_raw]
        # 根据文本结构，奇数项为输入的句子(源语句)，偶数项为输出句子(目标语句)
        input_texts = texts[::2]
        target_texts = texts[1::2]
        return input_texts,target_texts

    def data_parse(self,input_texts,target_texts):
        '''
        :param input_texts:  源语句
        :param target_text:  目标语句
        :return:
                dict_len: 字典长度
                encoder_input_data: 编码器输入，将分词后的源语句对应为词典中的index并且统一长度(取源语句中最长的，不足补零)
                decoder_input_data: 解码器输入
                decoder_target_data: 解码器输出
                    解码器输入和输出都为目标语句，并且相差一位
                    \t 词1 词2     词n  (输入)
                    词1 词2 .。。。 \n  (输出)
        '''

        # 计算词典
        texts = input_texts + target_texts
        dict_words = set()
        for text in texts:
            for w in text:
                if w not in texts:
                    dict_words.add(w)

        # 获取字典的长度
        dict_len = len(dict_words)
        # 生成词与index的映射表并保存
        dict_word_index = dict([(word,i) for i,word in enumerate(dict_words)])
        with open(self.opts.dict_path,'wb') as f:
            pickle.dump(dict_word_index,f)

        # 分别算出输入句子和输出句子的最大长度，即是句子中单词个数的max
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])

        # 初始化矩阵,全为0, 因为input_texts和target_texts是pair出现的，所以是一对一的关系，长度一致
        # 这些矩阵都是用来存单词在word_dict中的index的
        encoder_input_data = np.zeros((len(input_texts),max_encoder_seq_length),dtype=np.int32)
        decoder_input_data = np.zeros((len(input_texts),max_decoder_seq_length), dtype=np.int32)
        decoder_target_data = np.zeros((len(input_texts),max_decoder_seq_length,dict_len), dtype=np.float32)
        # decoder_target_data: 文本总数目，文本中最长的长度(单词数)，字典长度。用于模型的目标训练的one-hot向量

        # 将文字输入转化为张量
        for i , (input_text,target_text) in enumerate(zip(input_texts,target_texts)):
            # 将编码器和解码器得输入文字转化为字典中对应的标号
            input_index = self.seq2index(input_text,dict_word_index)
            encoder_input_data[i,:len(input_index)] = input_index
            target_index = self.seq2index(target_text,dict_word_index)
            decoder_input_data[i,:len(target_index)] = target_index
            # 将解码器的输出文字转为one-hot向量，与解码器的输入相比往后偏移一位(因为要做分类任务)
            for t,index in enumerate(decoder_input_data[i,1:]):
                decoder_target_data[i,t,index] = 1.0

        return dict_len, encoder_input_data, decoder_input_data, decoder_target_data


    @staticmethod
    def seq2index(text,dict_word_index):
        '''
        :param text: 中文语句
        :param dict_word_index: 词与标号对应的词典
        :return:  转化为标号的语句
        '''
        # 将输入文本转化为字典中对应的标号
        return [dict_word_index.get(word) for word in text]  # dict.get，找key对应的value

