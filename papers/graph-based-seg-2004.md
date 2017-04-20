## Effcient graph-based image segmentation

http://www.springerlink.com/index/n8110427355x2312.pdf


### 要点

1. 作者算法的优势
    - Capture perceptually important groupings or re- gions, which often reflect global aspects of the image.
    - highly effcient: O(nlog n) for n image pixels with low constant factor



### 个人点评

1. 从作者的介绍来看，好像很自然：selective search来找到可能的区域，用pre-trained的CNN来对区域进行特征提取，然后对区域进行classifier，但是作者能够对CNN的pool5,fc6,fc7(AlexNet)最后3层（非Conv层）来研究对于结果的影响是需要对CNN有很深理解的（研究的结果是：detection的性能更多取决于Conv. layer而不是其它的层，所以可以使用更前面的layer来减少参数量）
2. 当然这是作者R-CNN系列的第一篇，后续还有fast-rcnn等
3. 从实际的使用来看，性能（所谓的实时detection）也是不错的

