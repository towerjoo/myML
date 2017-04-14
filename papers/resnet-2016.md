## Going deeper with convolutions

GoogLeNet

https://arxiv.org/abs/1409.4842

此篇是说明Inception V1

### 要点

1. 22 layers 深度网络
2. 12倍少于AlexNet的参数数量，但是更高的准确度
3. 1.5 billion multiply-adds in inference time 保证现实的可用性（即使是大的数据集）
4. 1x1 size的使用主要来自Lin's Network In Network结构，原因：
    * dimension redution modules to remove computational bottlenecks
    * not only increase the depth, but also the width(without a significant penalty)
5. 参考R-CNN的2步法
    * 首先找到对象的location region(category-agnostic, low level cues)
    * 然后使用CNN对region进行分类
6. 增加depth和width是增加模型性能的不二之选，但是
    * 更多的参数(overfitting)
    * 更多的计算能力要求
7. Inception结构：如何使用dense component来近似sparsed structure of Conv. (当下的硬件对dense component计算更加高效，而Conv. Net具有稀疏性，所以如何使用dense来近似sparse是关键）
8. 具体的实现：
    * 70% ratio of dropout(与AlexNet, VGGNet的50%不同）
    * 1x1后紧跟3x3(或者其它的Conv, e.g 5x5, 7x7)
    * DepthConcat是个有趣的实现
        -  the suggested architecture is a combination of all those layers with their output filter banks concatenated into a single output vector forming the input of the next stage.
        - adding an alternative par- allel pooling path in each such stage should have additional beneficial effect
9. input image processing: 不同于AlexNet，将图片缩放到4个不同大小，256,288,320,352(较短边的长度），然后取left, center, right square（或者top, middle, bottom square如果是portraint), 对于square image，我们再取4个角，和中心（224x224), 水平翻转和缩放到224(4+1+1=6)的变形,每张输入图片会形成4x3x6x2=144个不同的crop, 然后对144个结果进行平均得到最终的结果(作者也试了对于144个结果进行max polling等，都比平均的效果差）
10. 结果和分析
    * approximating the expected optimal sparse structure by readily available dense building blocks is a viable method for improving neural networks for computer vision
    * 在ImageNet classification和detection中都取得了第一的结果，第二是VGGNet

### 个人点评

1. 相比于对于AlexNet的简单的深度改进，GoogLeNet使用了一个新的module，也就是所谓的Inception Module，它将filter banks拼接为一个新的output vectorUI为下一层的输入
2. 1x1的使用在宽度和深度的增加上成为可能（计算性上）
3. 输入144个crop然后平均取得的效果
4. 用dense来近似sparse以提高效率的方式
5. 模型的改进（Inception module的引入）是值得研究和关注的
