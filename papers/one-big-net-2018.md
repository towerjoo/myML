## One Big Net For Everything


### 要点
1. Many RNN-like models can be used to build general computers
	* each neuro is a processor
	* compress by pruning neurons and connections
2. One Big RNN for Everything
	* Extra goal-defining input patterns to encode user-given tasks
	* Incremental black box optimization of reward-maximizing RNN controllers
	* Unsupervised prediction and compression of all data of all trials
	* Compressing all behaviors so far into ONE
	* Store Behavioral Traces
		- once noise might become regular later
		- human can store 100 year sensory input at a reasonable resolution
	* Theory of Algorithmic Information (AIT) or Kolmogorov Complexity
3. conclusion
	* apply the AIT argument to show how a single recurrent neural network called ONE can incrementally absorb more and more control and prediction skills through rather efficient and well-understood gradient descent-based compression of desirable behaviors, including behaviors of control policies learned by past instances of ONE through neuroevolution or similar general but slow techniques
	* gradient-based compression of policies and data streams simplifies ONE, squeezing the essence of ONE’s previously learned skills and knowledge into the code implemented within the recurrent weight matrix of ONE itself


![One Big Net](/images/one-big-net.png)

### 个人点评

1. 更偏于架构层次的论述，缺少实际的实现
2. 整个paper并没有给出一个实现，没有代码和结果分析，只是停留在大的架构层次上，而这种one net for all的想法我想凡是
   DL业内的人都有想过，而且想法和paper中的也大体相似吧，只是Schmidhuber或许因其名声和多年的ML经验在架构和构思上更
   深入和合理一些


### 讨论

1. 整体上我是不喜欢这样风格的paper的，就像我们提出了一个方案或者设想，而没有去实现，那么意义有多少？而且也没有给出
   具体的实现建议。当然谁又不喜欢one for all的model呢？
