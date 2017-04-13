## Very deep convolutional networks for large-scale image recognition

VGGNet

https://arxiv.org/pdf/1409.1556.pdf

整体而言是在不改变其它参数的情况下，增加网络深度来考量其对结果的影响。

### 要点

1. 预处理：每个像素减去RGB平均值
2. 3x3 conv. layer with stride=1 (3x3是最小的包含上下左右中间的大小）
3. max pooling with 2x2 with stride=2
4. N层conv. layer紧跟3层全连接layer，构成了整个结构，通过调整N来研究其对结果的影响
5. 作者所有实验的网络（除了一个）都没有使用local response normalization(LRN)，作者认为使用对结果没有改进，而且增加了运算开销
6. 分析
  * 使用3x3 三次相当于使用7x7一次，为什么使用3x3?
    - 减少了参数81%，3(3^2C^2)=27C^2 vs. 7^2C^2=49C^2(C是图片的channel数）
    - 用了3次ReLUs（非线性）相比于1次,更加discriminative
  * 使用1x1? 增加了ReLUs单元,只是同维度的映射紧跟一个非线性单元（Network in Network使用此结构）
  * 更少的训练epoches（i.e 74, 相比于AlexNet)
    - greater depth, smaller conv
    - pre-initialization of certain layers
  * scale考虑进去，training scale S, S belongs to [256;512]
  * Since the fully-convolutional network is applied over the whole image, there is no need to sample multiple crops at test time(AlexNet测试时使用10个变形，而VGGNet则不需要，这里是把倒数第三层改为7x7, 最后2层FC改为1x1 Conv，从而整个网络成了full Conv. )
  * 测试时将测试图片rescale到Q, Q不需要等于S，也可以有多个取值
  * 测试时也需要水平翻转, 以及多组cropping也可以提高准确度
7. 结果：对多个N(11-19)进行了测试，也对多个数据集进行了测试（使用对imagenet而pretrained的model来测试其它数据集），都取得了很好的结果

### 个人点评

1. VGGNet的思路倒是非常简单的，固定参数，只增加深度，但是取得的效果还是很不错的
2. 文中提到了4核GPU的使用，19层的需要训练2-3周，所以pretrained的model显得很有意义（最近自己使用对imagenet的ResNet pretrained的model大致是500M左右，免去了自己训练的成本）
3. 1x1相比于3x3结果更差，说明了并非size越小越好
4. VGGNet用在localization也是很有意义的（最近刚好有个类似的需求需要实现）
5. 自从AlexNet突破性地出现后，2012年后基本都是对其进行改进，例如深度，size，stride，normalization等等，并没有在结构上有大的突破和创新，希望不久可以看到新的东西

