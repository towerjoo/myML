## Regularized Evolution for Image Classifier Architecture Search

by **Google Brain**

### 要点

1. 在archietecture search(或者叫model search)主要比较Evolution算法和RL方法
2. 贡献
	* a variant of the tournament selection evolutionary algorithm, which we show to work better in this domain
	* the first controlled comparison of RL and evolution, in which we show that evolution matches or surpasses RL
	* novel evolved architectures, AmoebaNets, which achieve state-of-the-art results
3. 方法
	* 搜索算法
		- NRE(Non Regularized Evolution): 有个随机sampling的population(models)，选择最优的Model进行变种生成子model，加入到
 	      population中，**淘汰掉最差的model**，重复过程
		- RE(Regularized Evolution): 有个随机sampling的population(models)，选择最优的Model进行变种生成子model，加入到
 	      population中，**淘汰掉最老的model**，重复过程(作者所谓的类比自然进化规律）
4. 实验和结果
	* 作者比较了小规模(CIFAR-10)在NRE和RE, RE和RL的结果
	* 作者比较了大规模(CIFAR-100, ImageNet)在NRE和RE, RE和RL的结果
	* AmoebaNets: Evolution算法找到的model, 达到了state of art的结果
5. RE的结果优于NRE，作者的分析是(作者提到这只是猜想)
	* RE: all models have a short lifespan, This requires that its surviving lineages remain good through the generations
	* NRE: these models may have reached their high accuracy by luck during the noisy training process
6. 结论
	* evolution exhibits considerably better performance than reinforcement learning at early search stages, suggesting it may be the better choice when fewer compute resources are available

### 个人点评

1. 这是[上篇](https://github.com/towerjoo/myML/blob/master/papers/evol-img-cls-2018.md)的一个续篇
2. 作者提到的RE（淘汰最老的instead of 最差的）的性能更好的猜想，还是很有意思的，只是显得比较残酷
3. architectures search的最大瓶颈还是算力，这个在本文中还是没有很好的解决办法, 结果就是理论上可行，而实践中不可行
4. evolution算法最后找到的model达到了state of art也是很有趣的，所以只要算力允许，算法去找model肯定是优于专家自行设计的（归根结底，无论是算法还是人工都是搜索空间的问题，足够大的搜索空间当然会有更好的结果）
