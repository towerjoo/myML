## Are GANs Created Equal? A Large-Scale Study

by **Google Brain**

### 要点

1. GAN模型如何客观地评价？
2. GAN模型评估的难点
	* one cannot explicitly compute the probability p_g(x), classic measures, such as log-likelihood on the test set, cannot be evaluated
	* focused on qualitative comparison, such as comparing the visual quality of samples (can be misleading)
3. 常用的评估模型
	* Inception Score (IS) is based on the fact that a good model should generate samples for which, when evaluated by the classifier, the class distribution has low entropy
		- The conditional label distribution of samples containing meaningful objects should have low entropy
		- The variability of the samples should be high
	* Fre ́chet Inception Distance is computed by considering the difference in embedding of true and fake data
4. FID(cannot handle overfitting well)
	* a memory GAN which simply stores all training samples would score perfectly under both measures
	* IS cannot capture the classes missing, while FID can, i.e precision and recall
5. 作者在定义好的标准和实验环境下对目前流行的GAN模型进行了实验分析
6. 结论
	* FID is a reasonable metric due to its robustness with respect to mode dropping and encoding network choices
	* empirical evidence presented herein imply that algorithmic dif- ferences in state-of-the-art GANs become less relevant, as the computational budget increases(这似乎基本成了一个通用的说法，甚至事实，也就是说算法变得越来越不重要，当其他相关因素有很大改进时，例如算力的提高，例如数据量的提高等）
	* no empirical evidence that the new GAM models outperform the original GAN model
	* future GAN research should be more experimentally systematic and models should be compared on a neutral ground.

![GAN comparing](/images/gan-comp.png)

### 个人点评

1. 很多东西都是单向的（或者接近单向的），例如密码的加解密，例如时间的走向，例如图片的表示（二进制）和生成（如何生成一个范冰冰），一个方向的计算总是几何数量级的难于相反方向的计算（或者理论上不可能），这也就形成一些实践的基础（如加密算法，人的老死，图片的刑侦价值等），但是我们仍能看到一些可能性，例如算力的极大提高时，一些问题会迎刃而解，而或许时间也并非不可逆
2. 一个东西，无论是社科还是数理，当需要比较其优劣时，一个相对客观的比较标准往往是最为重要的，否则各个参与者都会使用自己的标准来声称自己达到了最牛逼的结果，但是往往这个标准也是很难构建的（或者形成共识），数理模型还好，社科我们不是总在提什么特色吗？

### 讨论

1. Ian Goodfellow对于GAN paper review的[一些建议](https://twitter.com/goodfellow_ian/status/978339478560415744)
