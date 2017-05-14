## Distributed representations of words and phrases and their compositionality

extensions to word2vec

http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality


### 要点

1. word based to phase based
2. 简单的vector数学计算可以得出一些有意义的语义结果，如vec("Germany")+vec("capital")=vec("Berlin")
3. hierarchical softmax: 将计算复杂度从W降到log_2(W), 使用二叉树来表示输出为W的输出层, 每个word都可以近似地认为从root到leaf的路径来得到概率; 作者使用binary huffman tree这样可以使得常用的word得到更短的编码
4. negative sampling: 简化的NCE
5. subsampling of frequent words: P(wi)=1-sqrt(t/f(wi)), f(wi)是单词wi的频率，t是chosen threshold(一般是10e-5), 这样的选择是因为频率越高的单词往往提供更少的信息，例如the, a, in等，而频率小的单词则提供更多的信息
6. 作者提到了Skip-gram的representation体现了一定的线性结构，这也是为什么简单的vector运算就可以得到比较好的语义结果的原因
7. 作者提到了几个影响结果的hyperparameter: the choice of the model architecture, the size of the vectors, the subsampling rate, and the size of the training window.


### 个人点评

1. 这是word2vec的续作，是对于模型的一些改进，例如negative sampling的使用,subsampling的使用等
2. phase based显然是更为实用的，因为与现实更加吻合
