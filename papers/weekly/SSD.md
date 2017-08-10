## SSD: Single Shot MultiBox Detector

http://arxiv.org/abs/1512.02325



### 要点

1. contributions
    * one shot detector, faster than YOLO
    * predicting score/box offset for default bounding box(避免了proposal的开销）
    * prediction on different scales
2. model
    * multi-scale feature maps: 逐渐减小scale的layer，来实现在不同scale下的detection
3. training
    * matching: best jaccard overlap
    * 6 default boxes for each feature map location, i.e {1, 2, 3, 1/2, 1/3} and for 1, also use sqrt(s_k*s_k+1) scale ratio
4. model analysis
    * data argumentation is crutial
    * more default box shapes is better
    * Astrous is faster(use astrous version of VGG16)
    * multiple output layers for different relsoutions is better

![Framework](/images/ssd_framework.png)

SSD Framework

![Architecture](/images/ssd_arch.png)

Comparing to YOLO

### 个人点评

1. SSD比较有意思的是逐渐减小scale的layer直接作为output layer来进行判定
2. 略去了proposal的阶段，这也是SSD可以成为realtime detection的很重要的原因
3. 结合Google的另一篇关于各种detector system的比较，正好可以在工程中进行实践
4. 和YOLO的比较也是特别有意思的地方


### Resources

1. https://github.com/weiliu89/caffe/tree/ssd
