## Speed/accuracy trade-offs for modern convolutional object detectors

http://arxiv.org/abs/1611.10012

**在不同的设置下对比主要的detector模型（R-FCN, SSD, YOLO）以及如何在不同场景下根据speed/accuracy来选择**

### 要点

1. contribution
    * concise survey of different detector system
    * unified implementation using TensforFlow for different system
    * fewer proposals+Faster R-CNN can speed it up without losing too much accuracy. SSD is less sensitive to feature extractors comparing to other systems
    * several different combinations between meta-arch and feature extractors
2. meta architectures
    * 把输入图片进行分割，然后通过CNN来classifier，最终形成detection box, 例如R-CNN
    * proposal也可以通过NN来进行, 首先给出anchors(priors, default boxes)，对于每个anchor训练Model来进行 1）class prediction 2)offset the anchor needs to shift to fit the groundtruth bounding box, 例如Faster R-CNN, Multibox
3. SSD(Single Shot Detector)
    * single feedforward CNN to directly predict classes and anchors offsets without a 2nd stage per-proposal classification
4. Faster R-CNN
    * 1st stage, using RPN(region proposal network) to predict the class-agnostic box proposals
    * 2nd stage, the proposals are used to crop features from the same feature map, then fed to the FC to predict a class and bounding box
    * **very influential** in other detector systems
5. R-FCN (Region based Fully Convolutional Networks)
    * similar to Faster R-CNN, but instead of cropping features from the same layer where region proposals are predicted, crops are taken from the **last layer of features prior to prediction**
    * 这种方法减少了per region的计算量
6. unified settings
    * for feature extractors: VGG-16, Resnet-101, Inception V2, Inception V3, Inception Resnet(v2), MobileNet
    * number of proposals: 10-300
    * input size: M pixels in the shorter edge, M=600 hi-res, M=300 low-res
7. 结论：faster R-CNN+fewer proposal(50)通常是一个好的平衡


![comparision](/images/OD.png)


### 个人点评

1. 这种工程性的比较还是很有帮助的，如何平衡speed/accuracy
2. 很可惜没有将YOLO作为比较对象


### Resources

1. https://github.com/tensorflow/models/tree/master/object_detection
