## Sobolev Training for Neural Networks

http://arxiv.org/abs/1706.04859

By Google

**使用Sobolev训练NN，在训练期间除了target values外，还引入了target derivatives, 来提高predictor的质量（精度和泛化能力）**

### 要点

1. Sobolev training not only match the target value, but also the target derivatives
2. contributions
    * Sobolev training
    * explore the implications of matching derivatives
    * empirical demonstrating that Sobolev training leads improved performance and generalization
3. 可以将f(x)的1..K阶导数加入到训练过程中，如果feature比较多或者K比较大，可以使用stochastic approximations
3. 实验
    * regression on artifical data
    * model distillation
    * synthetic gradients
4. 最后作者提到可以将这种思路推广到导数以外的属性，只要这些属性与分布的不确定性相关即可

![bigpicture](/images/sobolev.png)

整体思路。


![formular](/images/sobolev_formular.png)

the loss function of Sobolev training.



### 个人点评

1. 将更多的信息引入到训练中，通常都会有一定的效果提升，但是也有相应的成本，例如
    * 适应条件，例如Sobolev就需要知道相应的微分，而这个通常不一定可以提供，这就使其适应场景大打折扣
    * 时间、复杂度相应于取得的效果的权衡
2. 作者提到的derivatives在某些场景下是可以获取的（例如有解析式的分布）, 或者作者说的**sometimes these quantities may be simply observable**

