## Efficient estimation of word representations in vector space

word2vec

https://arxiv.org/abs/1301.3781


### 要点

1. 此篇paper最主要的目的是可以从大量的数据(billions of words with millions of words vocabulary)里学到high-quality vector
2. word offset operation: vector("King") - vector("Man") + vector("Woman") = vector("Queen")
3. 前人工作：NNLM(Neural Network Language Model), 一个包含有线性projection layer和非线性隐藏层的feedforward NN用来学习word vector representation和统计语言模型
4. 具体的结构
    * 复杂度: O=ExTxQ (E: epochs, T: number of words in training set, Q: defined further for each model architecture)
    * feedforward NNLM: input+projection+hidden+output
    * RNNLM: input+hidden+output
    * 作者的新model: 基于作者的研究发现对于大量数据的处理最耗时（瓶颈）的部分是nonlinear hidden layer部分，所以基于此作者提出了
        - CBOW: Continuous Bag-Of-Words Model,类似于feedforward NNLM但是一处了nonlinear hidden layer, 并且projection layer在所有words中共享
        - Continuous Skip-gram Model: 当前的word作为输入，预测相邻的C个word(从结构上来看CBOW和Skip-gram正好相反）
5. 有趣的是这种简单的vector运算就可以找出语义上的结果，当然除了应用在相似度最近的结果，也可以找到相似度最远的结果


### 个人点评

1. DL在图像、语音等较低抽象下已经取得了较好的结果，而在NLP这个较高抽象下则并没有相应的成就，而最大的问题就是representation，如何表达words/paragraphs/sentences/article等,此篇就是解决了word2vec,以及后来的doc2vec等
1. 就像作者说的surprisely，向量的简单运算就可以得出特定规则语义的结果是很出乎意料的，这也说明此种representation的好处，揭示了不同维度的相似度，而不只是word level
1. 作者Mikolov还是很牛逼的，看了他在word2vec Google Group上的一些回复很是负责和严谨，对于学术的态度也是严谨的（例如无法达到doc2vec论文，Mikolov是二作的精度，就在后面的论文中提到了这是不可复现的）
