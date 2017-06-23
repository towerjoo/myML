## Batch normalization: Accelerating deep network training by reducing internal covariate shift

https://arxiv.org/abs/1502.03167

**Batch Normalization allows us to use much higher learning rates and be less careful about initialization**


### 要点

1. SGD requires careful tuning of model hyperparameters, esp. the learning rate, and initailization
2. saturation problem, gradient vanishing 问题通常通过relu, careful initialization, small LR来解决
3. BN的好处
    * normalization step that fixes the means and variances of layer inputs
    *  benefical on gradient flow(gradient vanishing问题）, by reducing the dependence of gradients on the scale of parameters or initial values
    * can use higher LR without the risk of divergence
    * regularize the model and reduces the need of Dropout
    * make it possible to use saturating nonlinearities by preventing the network from saturation
4. reduce internal covariate shift
    * covariate shift: change in the distribution of network activations due to the change in network parameters during training
5. 一个nonlinear的变换：z=g(Wx+b), 那么使用BN后就变为z=g(BN(Wx)) （b会消去）
6. BN enables higher LR, 作者证明了增大LR不影响梯度传播
7. BN regularizes the model
8. 然后是各种应用后速度的提升和精确度的提升


![BN](/images/bn.png)

上图就是normalization的操作

![BN](/images/bn2.png)

上图是BN的算法, 用来生成下一个网络的输入的变换，所谓的BN变换。

![BN](/images/bn3.png)

上图为BN的训练算法。

引用别人的一个总结:

> 整体的思想“：对于每个隐层神经元，把逐渐向非线性函数映射后向取值区间极限饱和区靠拢的输入分布强制拉回到均值为0方差为1的比较标准的正态分布，使得非线性变换函数的输入值落入对输入比较敏感的区域，以此避免梯度消失问题。因为梯度一直都能保持比较大的状态，所以很明显对神经网络的参数调整效率比较高，就是变动大，就是说向损失函数最优值迈动的步子大，也就是说收敛地快。NB说到底就是这么个机制，方法很简单，道理很深刻。 [source](http://blog.csdn.net/malefactor/article/details/51476961)

另外的一个总结(来自luosha865)

> 1.每个维度单独归一化，以免计算协方差矩阵； 2.用min-batch而不是所有数据的均值和方差做归一化； 3.加两个参数，参数在适当的数值下会还原输入，以免降低(实际上是扩大)原模型表达能力； 1跟2是减少计算量的，3防止降低模型容纳能力的（实际上提高了容纳能力）


### 个人点评

1. 这个论文写得这是太好了，简洁而详细，没有含糊和遮掩
2. BN这样的breakthrough的算法读起来真实棒，简单但是取得的效果太好了
3. 越是简单的方法往往越能取得好的结果（更好的数学支持），更广的应用场景等，但是提出的人是需要非常扎实的数学基础的，就像ResNet那么简单的结构
4. 过瘾！👍👍👍

