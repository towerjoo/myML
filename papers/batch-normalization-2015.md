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


### 个人点评

1. 这个论文写得这是太好了，简洁而详细，没有含糊和遮掩
2. BN这样的breakthrough的算法读起来真实棒，简单但是取得的效果太好了
3. 越是简单的方法往往越能取得好的结果（更好的数学支持），更广的应用场景等，但是提出的人是需要非常扎实的数学基础的，就像ResNet那么简单的结构
4. 过瘾！👍👍👍

