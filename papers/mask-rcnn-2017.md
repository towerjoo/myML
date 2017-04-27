## Mask r-cnn

Mask RCNN

https://arxiv.org/abs/1703.06870

### 要点

1. Mask RCNN的想法是在已有Faster RCNN的基础上增加一个分支，此分支就是一个简单的FCN用来预测每个RoI(Region of Interest)的mask, 加上已有的用来预测box, class的分支，构成了整个network
2. 具体说明
    * 其它与faster RCNN完全相同，只是并行地增加了一个mask predicting的分支
    * 使用的lost function: L = L_cls + L_box + L_mask 也就是三个输出各自的lost function
    * L_cls和L_box与Faster RCNN完全相同
    * we apply a per-pixel sigmoid, and define L_mask as the average binary cross-entropy loss
    * 使用FCN来对RoI进行mask predicting
    * RoIAlign: 这是作者说的最为关键的一个layer, 因为rounding的原因，会出现一个平移的错位，这个misalignment对于classification来说没有问题（classification可以很好地处理translation），但是对于pixel level的segmentation就是个大问题，于是作者使用不rounding来解决这个问题，也就是之前使用round(x/16)，作者使用x/16
    * 作者使用*bilinear interpolation*来compute the exact values of the input features at four regularly sampled locations in each RoI bin, and aggregate the result (using max or average)
3. 作者说后续会放出实现的代码，目前还没有，不过一个清华的学生实现了一个版本：https://github.com/CharlesShang/FastMaskRCNN.git



### 个人点评


1. 这篇就是RBG和Kaiming He你来我往的学术讨论的最新续作，此次不再是局限在object proposal, 或者detection，而是自然而然地过渡到了sementic segmentation(准确的说应该是instance level segmentation)
2. RoIAlign回过头来看是如此地自然和简单，但是能看看到问题和解决办法就要求对于网络的结构和细节有足够精确的了解，这就像众所周知的故事：画一条线值1元，知道在哪画值9999
3. 目前还没有比较好的实现，从github上那个实现（清华）还不是特别好，我看下自己是否可以参与下，也会持续关注作者自己的实现
