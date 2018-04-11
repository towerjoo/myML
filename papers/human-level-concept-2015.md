## Human-level concept learning through probabilistic program induction

published in **Science** about **one shot learning**

**Bayesian program learning (BPL) framework, capable of learning a large class of visual concepts from just a single example and generalizing in ways that are mostly indistinguishable from people**

### 要点

1. Bayesian Program Learning: learns **simple stochastic programs to represent concepts**, building them compositionally from parts, subparts, and spatial relations.
	* BPL’s advantage points to the benefits of modeling the underlying causal process in learning concepts.
	* Learning to learn was studied separately at the type and token level by disrupting the learned hyperparameters of the generative model.
	* Compositionality was evaluated by comparing BPL to a matched model that allowed just one spline-based stroke, resembling earlier analysis- by-synthesis models for handwritten characters that were similarly limited
2. Results(compare BPL with human and other approaches)
	* human: 4.5% error rate
	* BPL: 3.3%
	* other: >=8%
3. discussion
	* the principles of compositionality, causality, and learning to learn will be critical in building machines that narrow this gap(with human)
	* BPL still sees less structure in visual concepts than people do
	* shed light on the neural representations of concepts and the development of more neurally grounded learning models



the generative model
---

![BPL](/images/bpl.png)

BPL result
---
![BPL](/images/bpl-result.png)

### 个人点评

1. 将对象分解为part(不同粒度的part/subpart)和relation, 然后使用recurrent network来进行组合，生成最终的对象；整个intuition也是比较简单和常见，就像我们写程序的模块化，或者累积木的部件，但是作者提到的**组合，因果和learning to learn**中的后2者还是很关键的技术
	* 事实上本文分解parts/relations也是mannual的，而不是学习得来的，这或许也是一个遗憾
2. one shot learning的关键是从one(or few)个例子中学习到其中的因果关系，继而可以进行clasification, generation, inference等
	* 人脑似乎有很强大的分解和组合能力，特别是在因果关系上（关键特征提取），而分解的粒度和组合的方法，都是ML可以参考学习的

### 讨论

1. matlab(official code): https://github.com/brendenlake/BPL
2. python: https://github.com/MaxwellRebo/PyBPL
3. https://www.quora.com/What-is-Bayesian-Program-Learning-BPL
