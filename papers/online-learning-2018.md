## Online Learning: A Comprehensive Survey


### 要点

1. Online learning: comparing to traditional offline learning(or batch learning), online learning can learn from a sequence of instances
2. categories
	* supervised online learning: with label(with full feedback)
	* unsupervised online learning: without label(without feedback)
	* semisupervised online learning: with partial labels(with partial feedback)
3. theories
	* statistical learning
	* convex optimizaiton
		- first order methods: Online Gradient Descent(OGD)
		- second order methods: Online Newton Step(ONS)
		- regularization
	* game theory
		- game between predictor and environment
4. supervised online learning
	* first order methods: only exploit the 1st order derivative information
		- Perceptron
		- Winnow
		- Passive-Aggressive learning(PA): PA generally outperforms Perceptron
		- Online Gradient Descent(OGD)
	* second order methods: both 1st order and 2nd order derivative information(accelerate the converge, but often make the computation more complex)
		- Second Order Perceptron(SOP)
		- Confidence Weighted Learning(CW)
		- Adaptive Regularization of Weight Vectors
		- Soft Confidence weighted Learning
5. future directions
	* concept drifting: the target concepts to be predicted may change over time in unforeseeable ways
	* explore **large-scale** online learning for **real-time** big data analytics
	*  the “variety” in online data analytics tasks(differnt sources with different structured data)
	* address the data veracity issue by improving the robustness of online learning algorithms particularly when dealing with real data of poor quality


![online learning](/images/online_learning.png)

### 个人点评

1. 50多页的综述粗略读后感觉很是overwhelming，每一个点背后都有多少的paper和前人的努力
2. 这样的综述文章虽说只是对于过去的梳理和总结，但是也是很好的入手和对现状了解的入口

### 讨论

只能感慨：道阻且长（，行则将至）
