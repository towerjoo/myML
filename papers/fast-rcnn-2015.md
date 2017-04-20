## Fast r-cnn

Fast RCNN

http://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.html


### 要点

1. 作者之前提出的RCNN，有几个不足：
    * training is a multi-stage pipeline
    * training is expensive in space and time
    * object detection is slow
2. 因为分了几个stage，导致RCNN没有share computation，存在昂贵的重复计算, SPPnets是从sharing computation来改进RCNN
3. SPPnets则共享计算出的feature map，可以对RCNN的test time提高10-100倍，trainging time提高3倍
4. 作者指出SPPnets的不足：multiple stage pipeline, 不能更新Conv. net限制了精度
5. Fast RCNN有几个好处
    * 更高的检测准确度mAP
    * single stage using a multi-task loss
    * 训练时可以更新所有的层
    * 不需要将feature cache到硬盘
6. Fast RCNN架构
    * 一张完整的图片和集合的object proposal最为模型的输入，对于图片通过多层Conv. net和max pooling来处理生产conv feature map, 对于输入的object proposal通过RoI(region of interest) pooling layer来从conv feature map中获取一个feature vector; 每个feature vector会通过FC来处理最终分为2个处理分支：1）对K个object class和catch-all background class 通过softmax来生成可能性值 2)另外的一层对K个object class输出4位的位置信息值（坐标）
    * RoI pooling layer
        - 参考的SPPnets的spatial pyramid pooling layer
        - 使用max pooling对任何valid region of interest通过HxW(H, W是hyperparameter)的方框来转换为小的feature map
        - 对每一个channel都进行
    * 与pretrained model的集成
        - 将pretrained model的最后一层的max pooling层替换为RoI pooling layer
        - 将pretrained model的最后一层FC和softmax层替换为上面提到的sibling layer（分类和坐标）
        - 将pretrained model修改为可以支持2种输入：a list of images, a list of RoI in those images
    * fine-tuning for detection
        - hierarchical sampling
        - multi-task loss: 联合损失函数，也就是联合了class和location的loss
    * scale invariance
7. Fast RCNN用来检测：输入时图片和object proposal（2000或者45K更高），通过训练好的Fast RCNN来标记处每个proposal的class和可能性
8. 使用truncated SVD来加速检测(30%的改善)
9. 增加object proposal是否可以提高准确度，作者的实验表明：首先会有很小的改善，然后会下降
    - sparse object proposals seems to improve the detector quality
    - dense object proposals seems to hurt the detector quality
    - 这个结果似乎有些反直觉


### 个人点评

1. RBG是Berkeley的大牛，不仅学术好，而且工程能力也强，不仅有高质量的paper(R-CNN系列），而且可以实现的非常高效，据读过实现代码的人反馈，代码写的非常棒
2. R-CNN很有意思，像是高手过招，RBG提出了R-CNN，Kaiming He在此基础上给个改进（SPPnets)，然后RBG有在此基础上改进（Fast R-CNN)，然后Kaiming He有提出Faster R-CNN). 其实学术界和产业界有如此的“过招”是非常好的。
3. single-stage的过程还是很好的，因为提高了效率，降低了模型的复杂度
4. Fast R-CNN模型非常好，简单而且高效，就像别人评价Kaiming He的ResNet一样，简单高效，这才是模型之美，而不是各种复杂的集成学习要优雅太多（似乎自ResNet后对于ImageNet的提交或者winner都是这种集成学习）
