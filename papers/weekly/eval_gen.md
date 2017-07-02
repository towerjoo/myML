## A note on the evaluation of generative models

http://arxiv.org/abs/1511.01844


**对于generative models并没有一个one fit all的评价标准，而是要根据不同的应用场景来使用合适的metric**


### 要点

1. 目前几个主流的评价标准并不一致，也就是在一个标准下表现好并不一定在另外的标准下表现也好
2. 主流的评价标准
    * log-likelihood
        - difficult to evaluate or even approximate
        - 从人类直觉上看到的非常相似的图片，却可能有非常差的LL
        - 同样，有很大的LL，却可能是非常差的sample（也就是人类来看差别很大）
    * based on samples and nearest neighbors
        - 很小的一个图片的位移，会导致很大的欧氏距离变化，也就会导致一个完全不同的nearest neighbors
    * based on parzen window estimates
        - samples are used to construct a tractable model, then LL is evaluated against the proxy model
3. 作者认为parzen window estimates这种评价方法要避免使用, 因为这种proxy model可能与实际的density差的非常大
4. 作者说明对于generative models的评价并没有一个统一的标准，而是需要根据不同的application context来给出合理的评价，而且在一个application context下表现的很好的model并不能说它就在其它context下表现的同样好

### 个人点评

1. 事实上像火热的GAN等生成模型，都是很难评价的，特别是量化的方法，就像作者说的即使很高的LL，让人来看还是差别太大或者太不自然，所以评价标准是很关键的
2. 之前也提到了，现在很多的GAN模型都是采用人+LL等标准来评价，其实也是被广泛诟病的一个方面，人作为评价标准的引入是很难称得上科学的
3. 评价标准是很重要的，否则我们如何说一个model比另外的好，以及在不同场景下的好坏，这也是目前DL发展的一个制约，一方面大量的Model不断出现，另一方面好的评价标准还没有出来（例如这里的generative model，机器翻译等）


