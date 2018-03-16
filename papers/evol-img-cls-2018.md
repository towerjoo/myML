## Large-Scale Evolution of Image Classifiers

by **Google Brain**

### 要点

1. 作者提出使用evolution algorithm来自动训练model
	* evolution algorithm应用到模型训练中
 	* 减少engineer的参与
2. 作者提出只要有足够的算力，进化算法训练出的模型的精确度可以达到人工训练的模型
3. 相关工作
	* neuro-evolution(modify weight, add connection between nodes, insert node while splitting connection)
		- NEAT algorithm: direct encoding(similar to DNA)
		- indirect encoding
	* search algorithm(RL, Q-learning， etc.)
4. 作者的方法
	* 有个训练好的model的model *population*, 每次从中选择2个进行比较，性能差的直接*kill*，好的变成*parent*，系统对其*mutation copy*
      形成*child*，将*child* model放回*population*，重复整个过程，知道性能达到要求
	* 比较2个model的*fitness*还是非常computation heavy的，所以作者使用了并行、Lock free等技术
	* 初始条件： Every evolution experiment begins with a population of simple individuals, all with a learning rate of 0.1. They are all very bad performers. Each initial individual constitutes just a single-layer model with no convolutions
	* weight inheritance from parent to child 
5. 实验和结果
	* 250 workers
	* 94%+ accuracy against CIFAR-10, 70%+ for CIFAR-100 without change
6. 分析
	* meta-parameters(to escape the local optima trap)
		- population size
		- number of training steps
7. 结论
	* neuro-evolution is capable of constructing large, accurate networks for two challenging and popular image classification benchmarks
	* neuro-evolution can do this starting from trivial initial conditions while searching a very large space
	* the process, once started, needs no experimenter participation
	* the process yields fully trained models


![Model Compare](/images/model-compare.png)

![Search Alg](/images/search-alg.png)

Mutation Strategy(每次从中随机选择一个对parent进行修改，生成child)
![mutation strategy](/images/mutation.png)

![Results](/images/evolv-result.png)

### 个人点评

1. 读Google的论文还是比较爽的，清晰、简洁、完整，这篇论文就很好
2. 无论是人工还是算法，我们最终还是在search一个最好的参数集（结构，权重等），那么就像Thinking Fast and Slow书中所说的
   通常算法会优于人工的（主要是因为所谓的skill幻觉）
3. 相比于一些已有的只是搜索hyperparameter(例如Random Forest的超参数和结构的搜索），NN还会涉及到模型的结构本身，所以作者
   提到了算力的问题
4. 算力目前也算是DL前进的一个瓶颈，所谓算力过剩的传说在AI时代也算是终结了，我们期待更高性能的硬件，或者新的结构（量子计算机？）

### 讨论

等待阅读这片论文的续作
