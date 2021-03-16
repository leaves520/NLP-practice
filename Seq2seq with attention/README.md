**代码和图文解释参考网上各路大神，如有侵权，请联系我，我收到到会第一时间删除!**



**本代码是基于Seq2Seq with Attention的中文问答系统.**

- Encoder采用的单层双向的GRU.  

  - Pytorch双向的`nn.GRU`的输出为**最后一层所有时刻** $h^i_t=[\overrightarrow{h^i_t};\overleftarrow{h^i_1}]$ (其中双向的情况下，$\overrightarrow{h^i_t}$为正向时的隐状态，$\overleftarrow{h^i_1}$为反向时的隐状态.) 组成的向量 $output=\{h_1,h_2,...h_T\}$
  - 双向`nn.GRU`另外一个输出`hidden_states`为**所有层最后时刻**的$h_T$. 不妨设有m层并为双向，时间序列为T，则 $hidden=\{h^1_T,h^2_T,...,h^{m}_T\}$，其中 $h^i_t=[\overrightarrow{h^i_t};\overleftarrow{h^i_1}]$ . 
  - 输入Decoder的隐向量$S_0$采用**最后一层最后时刻的隐状态**，由于$Concat(hidden[-2],hidden[-1])$ 实际等于$output[-1]$，因此我们使用 $S_0 = output[-1]$

  <img src="https://i.loli.net/2020/07/02/g6DaQUHPlG9AqKe.png#shadow" alt="img" style="zoom: 67%;" />



- Attention的计算：
  $$
  E_t=tanh(attn(s_{t-1},H))\\
  \tilde{a_t}=vE_t\\
  {a_t}=softmax(\tilde{a_t})
  $$
  其中，$S_{t-1}$是 t 时刻Decoder中的上一时刻的隐状态，$H$指的就是 Encoder最后一层所有时刻的输出 $output$，$attn()$ 是任意全连接层或神经网络.

  

- Decoder采用的是单层单向的GRU.

  - 如下图仅仅显示Decoder一个输入的情况，$w$为经过$attention$机制后的$context$向量，$z$为Decoder上一时刻的隐藏层状态$S_{t-1}$

  <img src="C:\Users\leaves\AppData\Roaming\Typora\typora-user-images\image-20210316113128904.png" alt="image-20210316113128904" style="zoom:50%;" />

  - 相关计算公式：

  $$
  c=a_tH\\
  s_t=GRU(embed(y_t), c, s_{t-1})\\
  \hat{y_t}=f(embed(y_t), c, s_t)
  $$

  ​		其中， $H$指的就是 Encoder最后一层所有时刻的输出 $output$，$embed(y_t)$ 指的是将 `dec_input` 经过 WordEmbedding 后得到的结果，$f$ 是用于下一个单词预测的前向神经网络。**API `nnGRU` 的参数只有两个，一个输入，另一个隐藏层输入，**因此concat $[Context_{t},embed(y_{t})]$作为这一时刻的输入，而上一时刻隐状态$S_{t-1}$作为隐藏层输入 。同样对于预测任务 $f$ 的输入，是concat的向量 [$S_{t}$、$Context_{t}$、$embed(y_{t})$] , 这时$S_t$是t时刻更新后的隐藏层输出。



**几个注意点:** 

- $Context_{t}$ 的计算: T时刻Decoder上一时刻的隐状态$S_{t-1}$对Encoder所有的Output(最后一层所有时刻的$h_{1}.._{n}$)计算出attention score并进行加权求和。
- $S_t$ 的计算：起始状态$S_0$ 是Encoder最后时刻最后层的输出$h_t$，其余$S_t$是T时刻Decoder (nn.RNN) 接收concat向量$[Context_{t},embed(y_{t})]$ 和上一时刻隐状态$S_{t-1}$ 。
- Seq2seq的Decoder预测任务：T时刻Decoder的预测，定义一个全连接层并接收concat向量 [$h_{t}$、$Context_{t}$、$embed(y_{t})$] 。
- 由于Decoder接收的输入$Context_t$ 要利用Decoder上一时刻的更新的隐状态$S_{t-1}$进行计算，以及API `nn.GRU` 的调用的限制，不能一次性将序列的所有时刻传入`nn.GRU`，只能通过for循环，每次取当前时刻传入`nn.GRU`并更新这一时刻的$h、s$。
- Decoder的输入$y_t$是根据 $target$序列右移得到的。如`target: I like you `—> `dec_input: <STA> I like`
- API `nn.GRU` 的输入格式为$[time-steps,batch-size,input-hid]$

