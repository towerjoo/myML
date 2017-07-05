## Neural Turing Machines

https://arxiv.org/pdf/1410.5401

By **DeepMind**


### 要点

1. NTM is a **differentiable computer**, can be trained with GD
2. RNN/LSTM inspires the NTM, i.e the external memory mechanism
3. Neural Turing Machines
    * each component is differentiable, and the whole network is also differentiable
    * able to train with GD
    * content-based addressing: based on content similarity
    * location-based addressing: also apply product operations, etc.
4. controller
    * can use feedforward or RNN, but FF is more transparent(simple)
5. experiments on: Copy, Repeated Copy, Associative, N-Grams, Sort



![NTM](/images/ntm.png)

NTM's architecture

### 个人点评

1. 就像一个人在HN上说的DeepMind的paper总是能够让人眼前一亮，在基础层面的研究可谓是非常的深刻和有力
2. 从这个paper看到的结果，对于一些简单的算法型task，NTM可以做到非常好的结果，特别是像是排序这种，还是很让人惊奇的（如果可以分析DL的排序背后的实现细节，或许也可以给我们程序员以很大的启发）
3. 后续还有一篇DNC，就是这篇paper的续作，也是很值得一读



### Resources

1. https://github.com/carpedm20/NTM-tensorflow
