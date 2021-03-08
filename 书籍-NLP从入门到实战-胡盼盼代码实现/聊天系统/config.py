# 配置了一些文件路径以及模型相关的参数
class DefaultConfig:
    batch_size = 1000  # 批训练大小
    epochs = 30  # 训练轮次
    w2v_size = 128  # 词向量维度
    hidden_dim = 100  # 隐层维度
    optimizer = 'adam'  # 优化方式
    n_steps = 80  # 推理过程中目标句子的最大长度
    dict_path = 'data/dict.pkl'  # 字典保存路径
    data_path = 'data/subtitle.txt'  # 文件保存路径
    model_path = 'model/model_best.h'  # 模型保存路径