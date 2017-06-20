## Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour

https://arxiv.org/abs/1706.02677

By Facebook

**our Caffe2-based system trains ResNet- 50 with a minibatch size of 8192 on 256 GPUs in one hour, while matching small minibatch accuracy**

### 要点

1. distributed synchronous SGD
2. linear scaling rule to adjust the learning rate
3. warmup strategy: lower learning rates at the start of training
    * 和我们常见的learning rate decay策略不太一样
    * constant warmup: a low **constant** LR in the first few epochs
    * 主要是因为前期变化太大，不适合下面的LR adjust rule, 所以使用小的LR来缓解问题
    * gradual warmup: 在warmup阶段来逐渐调整
    * first few epochs 作者主要指的是5 epochs
4. 作者指出对于large minibatches, optimization is the main issue, instead of the generalization
5.  Linear Scaling Rule: When the minibatch size is multiplied by k, multiply the learning rate by k.
    * 黑魔法一般，没有合理的数学支持, 当然作者给了一个informal interpretation
6. Distributed SGD
    * Scaling the cross-entropy loss is not equivalent to scaling the learning rate.
    * Apply momentum correction after changing learning rate if using (10)
    * Normalize the per-worker loss by total minibatch size kn, not per-worker size n.
    * Use a single random shuffling of the training data (per epoch) that is divided amongst all k workers
7. 后面是具体的实现和设置，还有在其他相关领域的应用（segmentation, detection等）



### 个人点评

1. training time慢是个很头大的问题，但是每个iteration的gradient又是需要同步的，导致分布计算会很困难
2. FB给的这些工程层面的方法还是比较有帮助的，当然那些黑魔法的rule也是需要好好研究的
3. prediction phase目前也是很慢的（因为有百万级的参数），所以prediction phase的分布式也是很有必要的，否则如何谈实时的应用，或者CS结构的模式？


### Resources

1. https://github.com/facebookincubator/gloo (allreduce alg. by FB)
