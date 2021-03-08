import numpy as np
from hmmlearn import hmm
# 有三个模型，GaussianHMM,GMMHMM，用于连续型变量
# 下面采用的是多项式HMM，是离散变量

states = ["晴","雨","阴"]
n_states = len(states)
observation = ["运动","工作","玩乐","购物"]
n_observation = len(observation)

start_probability = np.array([0.3,0.2,0.5]) # 初始状态分布
transition_probability = np.array([[0.7,0.1,0.2],[0.2,0.6,0.2],[0.3,0.3,0.4]]) # 3 * 3 ,状态转移矩阵
emission_probability = np.array([[0.4,0.1,0.4,0.1],[0.3,0.5,0.1,0.1],[0.3,0.3,0.3,0.1]]) # 3 * 4 , 发射矩阵

# 多项式HMM，适用了离散变量
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# 给定观察序列，根据维特比算法猜测天气(隐状态序列)
actions = np.array([[1,0,1,1,2,3,2]]).T
_,weathers = model.decode(actions,algorithm = "viterbi")
print("行为:", ",".join(map(lambda x:observation[int(x)],actions))) # map(function , iterable_obj)
print("天气:", ",".join(map(lambda x:states[x],weathers)))
