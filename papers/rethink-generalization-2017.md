## Understanding deep learning requires rethinking generalization

https://arxiv.org/abs/1611.03530

ICLR2017 best paper award

为什么有的model的泛化好于另外的model?其中的原因是什么？

### 要点

1. 如果参数数量大于data sample数量，那么显然是可以做到100% fit的（这里当然说的是有效的数量，例如除去相关的sample等），可以类比方程解，当未知数个数大于方程个数时，方程有无穷多组解
2. 作者通过不改变model结构，只改变label（也就是随机对应），发现Deep neural networks easily fit random labels
    * NN容量够大时，足够记住整个dataset，而不是学得特征
    * 随机化label后model的训练时间并不比正常label的多太多（常数级）
    * 随机化label知识简单的data transformation,而保持其它的属性不变
3. 作者又通过对原始图片引入高斯噪音来观察model的表现
4. 作者得出常见的regularization方法（weight decay, dropout, data arugment等）并不能解释NN的泛化错误
5. 作者有提出了SGD是一种implicit generalization，相比于explict generalization，例如weight decay, dropout等
6. 当d>>n(d: 参数个数，n:sample个数），我们可以有无穷多组解而无论wx=y的y是什么（正确的label还是随机的label），那么无穷多组的解中，你如何判断泛化能力呢？(也即是从多组解中找到泛化最好的那一些解）
7. Xw=y有无穷多组解，w=X^Ta(根据SGD，见论文）,有XX^Ta=y这个公式，此公式只有唯一解，然后通过在MNIST,CIFAR10上发现这样的解泛化能力有很大提高
8. minimum-norm intuition may provide some guidance to new algorithm design, it is only a very small piece of the generalization story


### 个人点评

1. 这种对于模型本质的研究是很值得推崇的，这也是为什么此篇能够获得最佳论文的原因吧
2. deeper的趋向，也就会引入更多的参数，如果相比于data sample的数量都大很多的话，那么究竟是model记住了所有的dataset还是真正学到了规则，那就是需要仔细研究和论证的
3. 从hypothesis中如何选择最佳（训练集的accuracy，测试集的accuracy等）的model是非常重要的问题，作者提到各种显式的generalization都是压缩了hypothesis space但还不够，如何从中选择泛化能力强的是需要仔细研究而的
4. 那么接下来有几个思路：
    * 如何在参数比较少的时候training set的accuracy仍旧很高，这时理论上就不是记住了dataset而是真正学到了规则？
    * 作者的思路，如何在多组解中选择泛化能力最强的解？
5. 看到了在reddit/quora等上关于此篇论文的讨论，有意思的几点：
    * 一个人提到作者的random label实验证明了DNN根本没有泛化能力，而不是有很好的泛化能力
    * brute force memorization当然不是我们希望DNN做的，我们希望DNN可以返现规律，但是我们人类是否也是通过brute force memorization来泛化的？
    * 有人提到更短的训练时间和flatter minimum是是否泛化（学到真正的规律）的特征，我觉得还是很符合直觉的
5. 在作者发表此论文前，有读过作者的一些博客文章，除了学术，也是一个懂得生活的人和经常反思的人，如此的人才是有趣的人


### 参考

1. 一个解读：https://blog.acolyer.org/2017/05/11/understanding-deep-learning-requires-re-thinking-generalization/
2. https://www.reddit.com/r/MachineLearning/comments/5kfs23/r_understanding_deep_learning_requires_rethinking/
3. https://www.quora.com/Why-is-the-paper-%E2%80%9CUnderstanding-Deep-Learning-required-Rethinking-Generalization%E2%80%9C-important
