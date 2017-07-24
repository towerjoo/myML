## Densely Connected Convolutional Networks(DenseNet)

https://arxiv.org/abs/1608.06993

**每一层除了和下一层连接外，还和后面每一层连接，这样的结构可以有效缓解梯度消失问题，增强特征传递，鼓励特征复用，大量减少参数量**

### 要点

1. not all layers may be needed and highlights that there is a great amount of redundancy in deep (residual) networks -- 这个发现给了作者灵感
2. DenseNet针对Feature reuse来改进网络结构,相比于ResNet通过网络的深度（表达能力）来改善结果，DenseNet更加高效（参数少）以及更好训练
3. DenseNet细节
    * x_l = H_l([x0, x1, ..., x_l-1]) 其中x_i是第i层layer的数值（输出或输入), H_i是第i层layer的非线性变换
    * H_i使用的是BN+ReLU+Conv3x3的3个变换的复合
    * feature map 大小相同的组成一个Dense Block, 而不同的block之间通过pooling layer来调整大小
    * growth rate: 可以控制network的宽度
    * bottleneck layers: H_i可以再引入conv1x1的结构来较少输入，那么H_i就变为BN-ReLU-Conv1x1-BN-ReLU-Conv3x3
    * compression: 可以定义一个compression rate(0<theta<1)，来控制保留的信息
4. 与ResNets的比较：ResNets使用的是x_i = H_i(x_{i-1}) + x_{i-1} 这里的加（和）作者认为会阻止信息的流动,而DenseNet使用的叠加(concatentate)可以解决这个问题利于信息的流动

![Arch](/images/densenet.png)


### 个人点评

1. 现在的DL趋向于如何解决参数量过大以及梯度不稳定（消失或者爆炸），这篇论文就是很好的一个结果
2. 和ResNets类似，DenseNet的结构还是很简单的，但是改善的效果还是很明显的，这也是它可能取得类似于ResNets影响的原因，简约而不简单
3. 这种变动想想也不是很复杂，很多时候我们直观认为这种做法不行（例如会引入太多的参数），但是只有当真正实验后或许才能给出真正的有意义的结果。（很多人看着ResNets，感觉这样简单的结构自己也会，再看看DenseNet感觉有错过了1个亿）

