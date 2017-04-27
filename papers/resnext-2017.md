## Aggregated residual transformations for deep neural networks

ResNeXt

https://arxiv.org/abs/1611.05431

### 要点

1. 作者实现提到了hyperparameter的问题，这些人为加入的参数当然是越少越好，例如作者提到的VGGNet/ResNet的不断重复的网络单元，这样只增加depth就好；而另一方面Inception这样的网络通过精心设计网络结构来达到高效的方式又是另外的一个思路（更多的hyperparameter)，作者决定结合二者，i.e repeating layers and exploiting the split-transform-merge
2. 相关工作
    * multiple branch Conv. : Inception的多个分支，ResNet的2个分支等（解决梯度消失的问题）
    * grouped Conv. : 最早是AlexNet的利用多个GPU
    * Compressing Conv. : 在网络复杂度和精确度之间平衡
    * Ensembling
3. 具体实现
    * 对于简单神经单元的解释: \sum{i, D}(w_i*x_i)
        - splitting: 将x分割为低维度的多个参数x_i
        - transforming: w_i*x_i这里是进行了scale
        - aggregation: 所有的变换再进行合并
    * 根据作者对于NN中单个单元的分析，作者认为可以修改这里的变换，也就是将w_i*x_i修改为更加general的函数T(x)，此处可以为任何函数
    * 相应的，单个单元就变为: F(x)=\sum{i,C}T_i(x)这样的结构, 这里的T作者称作transformation, C就是transformation的数量，作者称作cardinality, 始终的C可以为任意数，不需要等于D
    * 作者将此结构称作 **Network-In-Neuron**
4. 作者进行了一些实验，然后发现C(cardinality）的大小对于网络的性能影响大于depth/width的影响


### 个人点评


1. 在2016年ImageNet的竞赛中，此结构最后得到了第二名，相比于优化参数或者ensembling的方法，此方法有意思的是简单而且易于扩展，作者发现对于单个神经单元的修改来增加网络的能力，是很有意思的
