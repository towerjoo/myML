## Deep Residual Learning for Image Recognition

ResNet

http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf

作者是希望增加网络深度同时保证参数数量合适的前提下使得网络可以顺利优化（收敛），而ResNet就是这样的一个结果。

### 要点

1. 152层（8X于VGGNet)
2. 网络深度增加后会出现的问题
    * 梯度消失或者梯度爆炸，导致不能收敛
    * 更深的网络可以收敛时，会出现degradation问题，也就是training error增加（深度越深，training error越高), 这个不是因为Overfitting
2. fit the residual mapping instead of desired original mapping(F(x)=H(x)-x => H(x)=F(x)+x, 这里F(x)是residual 映射， H(x)是原始映射）, 作者的意思是residual mapping会更加容易优化, 当取得F(x)最优时，可以通过将residual part->0来使得H(x)取得最优。
3. F(x)可以是很灵活的，例如2-3层全连接网络,也可以是Conv. 当然如果是1层，则会蜕变为y=W1x+x线性变换，这是就意义不大（没有优势）
4. residual结构的引入不会增加计算（处理element wise的简单加法）和参数量（参数量不变，相比于普通的深度网路）
5. y=F(x, {Wi})+Ws x 这里，Ws只在输入输出维度变化时需要（相应调整x的维度）, F则非常灵活，参考3
6. not use dropout
7. +x这部分当维度增加时有2种思路
    * identity: 多出的维度使用0作为padding, 不会增加计算成本和参数数量
    * projection: 使用Projection，需要更多的计算成本和参数数量
8. 作者对+x这部分做了3组实验（A:全部identify, B:维度增加部分projection其它identity，C:全部projection)，发现结果A < B < C，但是差别并不大，作者认为，padding 0的部分并没有使用residual learning导致结果会差点
9. 对于F作者选择使用bottleneck design, i.e 1x1, 3x3, 1x1三层Conv. 1x1这里用来降低维度以及后续恢复时增加维度，使得3x3面对低维度的输入输出（这里和identity shortcut配合使用，如果使用projection shortcut会比较低效）

### 个人点评

1. 从AlexNet提出后，大家都从深度和每个Conv. 的结构来尝试提高最终结果的准确度，也都取得了一定的进展，例如VGGNet等，但修改结构的并不多见，例如GoogLeNet和此篇的ResNet
2. 优化问题是深度学习的核心，我们构建了一个我们认为足够牛的网络结构，但当网络不收敛时，如何继续下去就是问题的核心，而优化的知识是最为重要的
3. 引入shortcut这种方法回过头来看似乎很简单，但是背后的思想和理论支持才是为什么作者可以提出此方法的原因，而非我们读者只是觉得简单
4. 一方面随着深度增加我们怕underfitting(training error增加），另一方面深度增加参数增加，我们又怕overfitting（test error增加，例如作者实验的10^3网络），那么尽量增加深度并且保持参数数量合适应该是我们需要重点考量和关注的（作者一直强调residual learning不会增加模型参数但可以增加深度和加快优化就是这个意思）

### 讨论

1. ResNet是不是RNN的一种？

J. Schmidhuber claims it's the same thing as an LSTM without gates. (http://people.idsia.ch/~juergen/microsoft-wins-imagenet-through-feedforward-LSTM-without-gates.html)
