## A few useful things to know about machine learning

Pedro Domingos from Univ. of Washington


### 要点

1. Learning = Representation + Evaluation + Optimization
	* Representation: some formal language computer can handle
	* Evaluation: objective function or scoring function
	* Optimization: key to the efficiency of the learner
2. generalization counts A LOT
	* training data vs. test data
	* we don’t have access to the function we want to optimize! We have to use training error as a surrogate for test error, and this is fraught with danger.
	* since what we try to optimize is just a proxy function, which means we might not need to fully optimize it.(local optimum might be better than global optimum)
3. data alone is not enough
	* no free lunch theorem: no learner can beat random guessing over all possible functions to be learned
	* in real world, some assumption is often the case, e.g smoothness, limited dependency, etc.
4. overfitting has many faces
	* bias: tendency to consistently learn the same wrong thing
	* variance: tendency to learn random things irrespective of the real signal
	* strong false assumptions can be better than weak true ones, because a learner with the latter needs more data to avoid overfitting
	*  to combat overfitting: cross-validation, regularization, etc.
5. intuition fails in high dimensions
	* curse of dimensionality
	* Naively, one might think that gath- ering more features never hurts, since at worst they provide no new information about the class. But in fact their bene- fits may be outweighed by the curse of dimensionality.
6. theoretical guarantees are not what they seem
7. feature engineering is the key
	* one of the most interesting parts, where intuition, creativity and “black art” are as important as the technical stuff.
	* domain specific
	* features might correlate (XOR problem)
8. more data beats a cleverer algorithm
	* a dumb algorithm with lots and lots of data beats a clever one with modest amounts of it
	* it pays to try the simplest learners first
9. learn many models not just one
	* model ensembles, e.g bagging, stacking
	* should not be confused with BMA(Bayesian model averaging)
10. simplicity doesn't imply accuracy
	* not apply Occam's Razor
	* The generalization error of a boosted ensemble continues to improve by adding classifiers even after the training error has reached zero.
	* contrary to intuition, **there is no necessary connection between the number of parameters of a model and its tendency to overfit**
	* the size of the hypothesis space is only a rough guide to what really matters for relating training and test error: the procedure by which a hypothesis is chosen.
	* simpler hypotheses should be preferred because simplicity is a virtue in its own right, not because of a hypothetical connection with accuracy
11. representable doesn't imply learnable
	* e.g standard decision tree learners cannot learn trees with more leaves than there are training examples.
	* if the hypothesis space has many local optima of the evaluation function, as is often the case, the learner may not find the true function even if it is representable.
	* Some representations are exponentially more compact than others for some functions
	* Finding methods to learn these deeper representations is one of the major research frontiers in machine learning
12. correlation doesn't imply causation
	* correlation is a sign of a potential causal connection
	* Machine learning is usually applied to *observational* data, where the predictive variables are not under the control of the learner, as opposed to *experimental* data, where they are (对比实验是需要的，来确定因果性，否则只能是相关性）



![learner](/images/learner.png)

![learner](/images/bias_var.png)

### 个人点评

1. 对于新手这些结论性的guide line通常还是很有帮助的，只是这些通常都是一些how/what，而不是why，而懂得why才是更上一个台阶的基础
2. 2012年的文章，6年后的今天可能会有一些obsoleted，例如关于feature engineering（DL出来后解决end to end的问题，而避免或者弱化feeature engineering）,但是总体上可以看到一些从业者的心得

### 讨论

1. 在GPU等硬件、DNN等模型算法不断提高的基础上，feature engineering的作用是否已经降低？

这个问题其实我还是没有资格回答的，我的直观感受是DL所谓的end to end的学习方法是有条件的，通常受制于效率（时间，内存，数据）而通常在实际中是不现实的（但是理论上应该是可行的），而FL作为中间一部引入domain specific knowledge通常会提高效率，在实际场景中更加适用（当然效率指的是运行效率，开发效率则相比DL会大大增加），所以通常需要平衡二者，而不是放弃任一个
