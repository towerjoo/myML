## Lipton et al. - 2015 - A critical review of recurrent neural networks

https://arxiv.org/abs/1506.00019

### 要点

1. RNN的历史
 * 顺序的引入
 * 类似理念的回顾(Jordan network等）
2. 训练RNN
 * backprog through time(BPTT算法）
 * vanishing and exploding gradients （不能收敛的问题）
 * for vanishing gradients(w(i,j)<1)
  - recurrent edge会影响下一个时间点的输出，如果值如果<1，那么会导致输出变小，而变小是时间点的指数级（变大类似）
  - TBPTT(Truncated BackPropagation Through Time)
  - LSTM(Long Short-Term Memory)
 * saddle-free version of Newton's method(Dauphine 2014)
 * Hessian-free truncated Newton's method(Martens 2011)
 * 通常而言牛顿法需要Hessian矩阵,所以在复杂网络中不现实,相比于SGD
3. modern RNN architecture
 * LSTM (Hochreiter 1997)
  - input node: 通常使用tanh activation function
  - input gate: 截断或者放过数据流，sigmoid function
  - internal state: memory cell, 固定的单位weight,保证gradient不会改变
  - forget gate: 原始论文中没有此component，由Gers 2000提出，此component使得network可以学习刷新自己的状态(internal state)
  - output gate: 一个memory cell的输出结果就是internal state * output gate
 * BRNN(Bidirectional Recurrent Neural Networks) (Schuter 1997)
  - 不只可以向前看，还可以向后看，也就是不只前面的信息会影响输出，未来（后面）的信息也会影响输出
  - 但是通常未来的信息是NA的
 * LSTM和BRNN可以结合使用，BLSTM
 * Neural Turing Machines(NTM) (Graves 2014)
  - controller: feedforward或者recurrent网络，向外部接受输入和输出结果，也可以从Memory中读写
  - memory matrix: 提供memory
4. RNN的应用
 * text/video classification
 * image captioning
 * language translation
 * word predicting
5. 自然语言输入输出的representation
 * 输入
  - one-hot encoding: 简单，但效率低（sparse)
  - meaning vector: word2vec
 * 输出
  - softmax vector
6. 结果评估方法
 * 通常没有一个唯一的正确答案（例如翻译,captioning等）
 * BLEU score(与人的评价相关）
 * METEOR
7. 机器翻译
 * Sutskever 2014 2个LSTM，一个encoding model, 另一个decoding model (En to Fr)
 * Auli 2013, Fr to En
8. 图片注解
 * encoding with CNN, decoding with LSTM

### 个人点评

1. review的论文读起来还是很有意思的，而且可以让我们有个big picture
2. sequence to sequence这种模式是很牛逼的，首先打破了传统ML的独立分步的限制，另外很多机器学习的任务都可抽象为这种模式
3. end to end在RNN（及其衍生模型）中更加体现，语音识别、机器翻译过往需要各种中间representation，已经不再需要
4. RNN及其衍生模型虽然取得了不错的结果，但也是黑盒，我们只知道它如何工作，但不知道为什么
5. 一个模型的评估标准很重要，否则我们如何知道新的比旧的好？
6. GD的方法通常在复杂的模型中有很好的表现，但前提是gradient不会vanish或者expolode，新模型重要的是可以保证loss function可以不断下降
