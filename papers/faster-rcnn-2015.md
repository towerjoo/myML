## Faster r-cnn

Faster RCNN

http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks

### 要点

1. Faster RCNN其实是解决object proposal的问题，而不是detection的问题，因为作者意识到检测的过程的瓶颈是object proposal的时间
2. 比较了主流的object proposal的方法(Selective Search, EdgeBoxes等）等方法，都是相对比较慢
3. 作者是使用Conv. net来做object proposal


### 个人点评

