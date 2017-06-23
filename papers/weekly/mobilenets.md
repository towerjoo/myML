## MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

http://arxiv.org/abs/1704.04861

By Google

就像名字一样，这个model希望能够在首先环境下可以构建出延迟小，规模较小，精度不会太多损失的model.

### 要点

1. 为了取得延迟低和size小2个目标
1. MobileNets的相关工作
    * 主要灵感来自于depthwise seperable conv to reduce the computation in the first few layers
    * 另外的方法是shrinking, factorizing, or compressing pretrained networks 
    * distillation: use larger network to teach a smaller network
    * low bit networks
1. MobileNets Architecture
    * 使用depthwise seperable conv（一种factorized conv）的方法将标准的conv layer分成2个layer，1 layer用来filtering, 1 layer用来combining, 这个过程可以很大程度较少运算和model size
        * depthwise conv: apply a single filter per each input channel
        * pointwise conv: 1x1 conv to create a linear combination of depthwise layer
    * 28 layers, 1st layer是full conv, 中间层是depthwise, final layer是FC
1. 2 extra hyperparameters
    * width muliplier: thinner models, 当需要更快和更小的模型时，可以通过更改这个超参数来实现。这个参数用来thin每一次的width, 也就是input channels M变为alpha X M, output channels N变成alpha X N. alpha通常在(0, 1]之间取值。而model的参数量大致是alpha^2
    * resolution muliplier: reduced representation, 其实是分辨率的调整，直观地说就是可以把一个分辨率高的图片（像素多）来进行一定的缩放来降低分辨率，最终达到减少模型参数的目标, model的参数大致是p^2
        


![mobilenets](/images/mobilenet.png)

这个图分子是mobilenets需要的参数数量，分母是标准conv需要的参数量，作者提到MobileNet使用3x3 depthwise seperable conv相比于标准的conv可以节省8-9倍的运算量（在准确度损失非常小的情况下）


![mobilenets params](/images/mobilenet_param.png)

上图是增加2个超参数后的模型参数量，其中alpha是width muliplier, rou是resolution muliplier

### 个人点评

1. 对于实时的classification/detection/segmentation场景，CS的模式通常都不行（因为不确定的网络通信），所以需要客户端自己训练模型或者使用pretrained model，所以研究如何在首先环境下训练和预测是非常重要的
2. latency/accuray/model size三者的平衡就是问题的关键


### Resources

1. http://machinethink.net/blog/googles-mobile-net-architecture-on-iphone/ 这个哥们针对iOS实现的MobileNet我编译和测试了，感觉实时性和准确度还是不错，只是很快就变得很热（应该是百万级的参数导致每一个prediction都需要大量的计算）
2. https://github.com/marvis/pytorch-mobilenet 
3. https://github.com/Zehaos/MobileNet

