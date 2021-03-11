该实现采用的是Negative Sample的方法来加速模型的训练，使用NCE Loss作为损失函数，具体可参考：

- [The Illustrated Word2vec – Jay Alammar – Visualizing machine learning one concept at a time. (jalammar.github.io)](http://jalammar.github.io/illustrated-word2vec/)
- 求通俗易懂解释下nce loss？ - 隔壁大王的回答 - 知乎 https://www.zhihu.com/question/50043438/answer/947299771



不严格的说明，仅仅从代码的角度来看损失函数的计算：

- 计算中心词与Pos_Target的向量内积，经过LogSigmoid  $log(\frac{1}{1+exp(-x)})$ 得到一个损失
- 同理中心词与Neg_Target也是这样计算，
- 最后正负样本的loss加起来再取负号，即为模型的loss



稍有严格的数学证明角度来看 (**如有不正确请指正**)

- 对中心词的左右C个窗口内的单词(共2 × C 个)以及负采样的 2×C×K个单词。每一个单词都做一个单独的逻辑回归任务(最后过一层Sigmoid)，得到每个单词的概率，$p_{y_{i}}\left(x_{i} ; \theta\right)$。

  从极大似然的角度出发，我们希望极大化如下似然函数：
  $$
  l(\theta)=\log \prod_{i=1}^{n} p_{y_{i}}\left(x_{i} ; \theta\right)=\sum_{i=1}^{n} \log p_{y_{i}}\left(x_{i} ; \theta\right)
  $$
  其中 $p_{k}(x ; \theta)=p(y=k \mid x ; \theta)$ ; 当y=1时，$p_{k}(x ; \theta)=p(x ; \theta)$; 当y=0时，$p_{k}(x ; \theta)=1-p(x ; \theta)$.于是，
  $$
  \log p_{y}(x ; \theta)=y \log p(x ; \theta)+(1-y) \log (1-p(x ; \theta))
  $$
  最大化上式等价于最小化下式，被称为负对数似然损失。
  $$
  -y \log p(x ; \theta)-(1-y) \log (1-p(x ; \theta))
  $$
  

