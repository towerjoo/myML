## Selective Search for Object Recognition

http://link.springer.com/article/10.1007/s11263-013-0620-5


### 要点

1. 常见的方法，例如穷举法，但是sample space太大，必须做出限定，限定后通常sample space还是很大，而且可能有miss的风险
2. 作者提出了data driven selective search
    - 也学习了穷举法的思想，不错过任何的可能sample
    - 目标：class-independent, data-driven selective search strategy来生成较小集合的高质量的对象位置
3. 相关的研究（只针对图像识别）
    - exhaustive search: slide window来实现穷举，当然sample space会很大；通常使用较粗的搜索窗口和固定的长宽比, e.g HOG, 作者的selective search也是穷举法的一种，不过使用了图片的结构信息,也不限定搜索窗口和窗口比例，能够较快生成较小数量的可能位置
    - segmentation: 生成多个foreground/background分割,学习分割的foreground是完整对象的可能性，一次对segmentation进行评分. 作者提到此方法依赖于一个strong algorithm来识别可能的segmentation，而这个算法显然是更加依赖于人和domain knowledge； 作者的方法则使用多种不同分组标准和表示方法来自动找到可能的位置
    - 其它的方法诸如对对象的shape来pretrained的classifer来识别可能的区域，Bag-of-Words使用visual words来识别可能的区域等。
4. 作者方法的创新性
    - 相比于穷举法，作者使用segementation来生产少量的类别无关的位置
    - 相比于依赖单一strong算法的segmentation, 作者使用多种策略来处理
    - 相比于随机的sample box，作者使用bottom-up grouping来生成好的对象位置
5. 具体的过程
    - 使用[graph-based segmentation](papers/graph-based-seg-2004.md)方法进行初始的分割
    - 贪婪算法来将区域组合
        * 计算相邻区域的相似度，相似度最高的进行组合
        * 对新组合的区域也使用同样的方法来计算相邻区域相似度并组合最高相似度的相邻区域
        * 重复上面的过程，知道整个图片成为一个区域
    - 相似度s(ri, rj) 是区域ri和rj的相似度，作者希望相似度的计算是基于特征，而特征可以传播，例如当i,j合并为t时，rt的特征值可以直接从ri, rj来计算，而不用再对像素进行计算
    - diversity strategy
        * different color space
        * different similarity measures(s_color颜色, s_texture纹理, s_size吃尺寸, s_fill填充，然后s(ri,rj)=a_1*s_color+a_2*s_texture+a_3*s_size+a_4*s_fill计算出总的相似度, a_i属于{0, 1}
        * varying the starting regions
6. 最后是和已有的实现来进行对比


### 个人点评

1. 因为R-CNN就是用这个方法的，所以顺便看了下这篇论文
2. 一张图片是结构性的，而且不同对象之间可以重叠（汽车和轮胎)等复杂关系，所以需要穷举不同的组合
3. 作者使用的自下向上根据相似度来组合region是个有意思的方法
4. 作者基于不同的diversity strategy来达成穷举也是很好的尝试
5. 从作者的分析和R-CNN的结果来看，质量和数量的平衡以及性能也是不错的

1. 从作者的介绍来看，好像很自然：selective search来找到可能的区域，用pre-trained的CNN来对区域进行特征提取，然后对区域进行classifier，但是作者能够对CNN的pool5,fc6,fc7(AlexNet)最后3层（非Conv层）来研究对于结果的影响是需要对CNN有很深理解的（研究的结果是：detection的性能更多取决于Conv. layer而不是其它的层，所以可以使用更前面的layer来减少参数量）
2. 当然这是作者R-CNN系列的第一篇，后续还有fast-rcnn等
3. 从实际的使用来看，性能（所谓的实时detection）也是不错的

