##  Towards perspective-free object counting with deep learning

https://www.researchgate.net/profile/Daniel_Onoro/publication/306118084_Towards_perspective-free_object_counting_with_deep_learning/links/57b2c3f108ae95f9d8f5d5cc.pdf

**使用DL来对image patches的object density maps进行回归从而计算相应的对象数量**



### 要点

1. 贡献
    * Counting CNN to perform an accurate regression of object density maps from image patches
    * object densities can be estimated without the need fo any perspective map or other geometric information of the scene
1. counting objects model
    * model the counting problems as one of the **object density estimation**
1. Counting CNN(CCNN)
1. Hydra CNN: 基于CCNN, scale-aware solution

![CCNN](/images/ccnn.png)

CCNN的网络结构

![Hydra CNN](/images/hydra_cnn.png)

Hydra CNN结构


### 个人点评

1. naive的办法是找到对象然后计数，显然效率会差很多，因为计数并不需要知道具体的对象位置信息，所以density maps到不妨是个好办法
1. 从结果来看还是不错的，通过density maps来估算数量在速度和精度上都有不错的表现


### Resources

1. https://github.com/gramuah/ccnn
