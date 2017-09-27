## 100-epoch ImageNet Training with AlexNet in 24 Minutes

http://arxiv.org/abs/1709.05011


### 要点

1. 作者指出训练慢的原因是batch size太小, 而不能有效利用多核处理器
    * 但是通常大的batch size会损失准确度
    * 作者提出使用LARS(Layer-wise Adaptive Rate Scaling)算法来保证在大的batch size前提下不损失准确度
2. 分布式的训练方法
    - Data Parallelism: 训练数据被分为P部分用于不同机器来训练
        * 需要master和训练机器同步的见下面的图
        * 另外一种异步的方法是master(称作Parameter machine)只与j训练机器进行gradient/w/b同步，而不与其它的机器同步
    - Model Parallelism: 将model(NN)分为P部分，也就是并行矩阵计算
3. 大的batch size并且不损失准确度的方法
    * warmup scheme + LARS 算法


![parallel](/images/parallel.png)

分布式训练(需要同步gradient和w/b)的常用办法

![dist](/images/dist_train.png)



### 个人点评

1. 训练效率确实是个大的问题，无论是工业界还是学术界，让整个debug cycle变得很难忍受，所以很乐见这样的文章
2. 里面提到了LARS算法，这个接下来会看看


### Resources

1. [Accurate, large minibatch sgd: Training imagenet in 1 hour](https://arxiv.org/abs/1706.02677)可以对照着来看
