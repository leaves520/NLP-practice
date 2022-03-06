### 任务五：基于神经网络的语言模型

用LSTM、GRU来训练字符级的语言模型，计算困惑度

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、15章
   1.   [机器翻译, 文本生成等任务评价指标 BLEU, ROUGE, PPL(困惑度)_牛客博客 (nowcoder.net)](https://blog.nowcoder.net/n/196d1ad4f30c4107a02c4b31f2d6a0ec?from=nowcoder_improve)
   1.   [机器翻译与自动文摘评价指标 BLEU 和 ROUGE - 简书 (jianshu.com)](https://www.jianshu.com/p/0afb93fda403)
   1.   [困惑度(perplexity)的基本概念及多种模型下的计算（N-gram, 主题模型, 神经网络） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/114432097)
2. 数据集：唐诗古诗poetryFromTang.txt
3. 实现要求：Pytorch
4. 知识点：
   1. 利用RNN进行文本生成
   2. 评价指标：(详见参考1.1的第15章，第4节—评价方法)
      1. 困惑度. lower is better 
      2. BLEU. higher is better (类似精确率)
      3. ROUGE. higher is better (类似召回率)
