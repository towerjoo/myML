## Imagenet classification with deep convolutional neural networks

AlexNet

http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

### 要点

1. 自己实现了GPU优化的2D卷积和其它操作 http://code.google.com/p/cuda-convnet/
2. 一次训练需要5-6天，作者认为受限于硬件导致不能增加网络的深度，而不是更多的深度没有意义
3. 预处理
    * 所有都处理为256x256
    * 对于尺寸小于256的，先拉伸到256
    * 对于长方形的，取中心区域
4. 使用ReLUs替代之前的tanh或者sigmoid, 达到25% training error需要的时间更短（6倍于tanh)
5. 在多个GPU并行训练
6. local response normalization
7. overlapping pooling: 与常规的s=z不同，作者使用s < z,这样会产生overlapping，但是结果证明，可以更好避免overfitting
8. 8层网络
    * 5层卷积网络
    * 3层全连接网络
9. 较小过拟合
    * data argmentation
        - 从256x256随机平移(水平映射转换等）等转换成224x224的图片作为输入,将训练集扩大了2048倍,测试时选取测试图片(256x256)的4个角和中心点形成的224x224的图片（5个）和水平映射转换（5个）总共10个图片的预测结果平均，得到最终的结果
        - 改变图片RGB通道的intensity(PCA)
        - dropout: 50%概率设置neuro的输出为0，让model学习更多的robust feature，增加后让收敛的iteration数增加了一倍
10. 结果：做了各种对比和分析


### 个人点评

1. AlexNet可以算上是DL在图片分类领域取得的**突破性**进展（对比前一年ImageNet的结果），作者的paper也是非常详细说明了model的结构，原因及其对结果的影响，也比较短，非常好的论文
2. PCA对于图片intensity特征值的改进说明了领域知识和数学知识对于model研究的重要性
3. 2048倍的数据生成也是很关键，避免了位移和水平翻转的影响
4. dropout对于结果的影响也很有意思，特别是作者提到网络应该减少对相邻neuro的依赖，而需要学习到更加robust的特征
5. 其它的几个模型的改进也很有意思，例如normalization, overlapping pooling等
6. 多个GPU并行训练也是当下硬件性能受限的一个比较好的办法
