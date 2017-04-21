## Faster r-cnn

Faster RCNN

http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks

### 要点

1. Faster RCNN其实是解决object proposal的问题，而不是detection的问题，因为作者意识到检测的过程的瓶颈是object proposal的时间
2. 比较了主流的object proposal的方法(Selective Search, EdgeBoxes等）等方法，都是相对比较慢
3. 作者是使用Conv. net来做object proposal
4. 作者提出了RCN(Region Proposal Network): takes an image(of any size) and outputs a set of rectangular object proposals, each with the objectness score
    * 使用sliding window和Conv. net来做object proposal
    * 做object proposal的Conv. net feature map也可以在Fast R-CNN上复用
5. one stage detection vs. two-stage proposal+detection
    * Overfeat使用的是one stage，也就是使用sliding window来做object proposal的同时做category classifer
    * RPN+Fast R-CNN则是two stage, 首先使用RPN做object proposal(category independent)然后用R-CNN来做localization和classifier
    * 作者提出two stage这种瀑布式的方法更加准确和高效，相比于one stage


### 个人点评

1. 此篇并非对Fast R-CNN的改进，还是使用Conv. net来做object proposal然后feed到Fast R-CNN来使用
2. 作者argue瀑布式的方法更加高效和准确，这个可能需要更多的理论和工程方面的证明，直觉来讲，one stage应该不输于two stage
3. R-CNN的相关论文基本读完了，下来看看overfeat和FCN在detection/segmentation的模型
