## One-Shot Generalization in Deep Generative Models

from **DeepMind**


### 要点

1. one-shot generalization: an ability to generalize to new concepts given just one or a few examples
2. two approaches for one-shot generalization:
	* a probabilistic model that combines a deep Boltzmann machine with a hierarchical Dirichlet process to learn hierarchies of concept categories as well as provide a powerful generative model
	* the ability of probabilistic models to perform one-shot generalization, using Bayesian program learning, which is able to learn a hierarchical, non-parametric generative model of handwritten characters
3. Contributions
	* sequential generative models that provide a generalization of existing approaches, allowing for sequential generation and inference, multi-modal posterior approximations, and a rich new class of deep generative models
	* the combination of attentional mechanisms in more powerful models and inference has in advancing the state-of-the-art in deep generative models.
	* our generative models have the ability to perform one-shot generalization.
4. Varieties of Attention
	* Attending to parts of a scene, ignoring others, analyzing the parts that we focus on, and sequentially building up an interpretation and understanding of a scene
	* any mechanism that allows us to **selectively route information from one part of our model to another** can be regarded as an attentional mechanism
	* inference and generation
		- generative process uses **writing or generative attention**
		- inference process uses **reading attention** (similarly that uses in image classification)
	* Spatially-transformed attention
		- more powerful than selecting a patch of an image
		- more general that provides invariance to shape and size
5. Iterative and Attentive Generative Models
	* In this framework, we can think of the generative model as a decoder of the latent variables, and the inference model as its inverse, an encoder of the observed data into the latent description
	* use **free energy objective** to optimize
	* hidden canvas functions
		- additive canvas
		- Gated Recurrent Canvas
	* difference with other VAEs
		- the introduction of the hidden canvas into the generative model that provides an important richness to the model
		- allows the model be sampled without feeding-back the results of the canvas c_t to the hidden state h_t(reduce model parameters)
		- different type of attentions used
6. One-Shot Generalization(3 tasks to evaluate)
	* Unconditional Generation
	* Novel variations of a given exemplar
	* Representative samples from a novel alphabet
7. limitation
	* the need for reasonable amounts of data
	* our models can perform one-shot generalization, they do not perform one-shot learning


![Spatial Transform](/images/spatial-transformer.png)

![graph](/images/one-shot-dgm.png)

### 个人点评

1. DeepMind的paper还是一如既往的好读，但是限于自身的知识，也是读的一知半解，但是从paper上可以看到很多有价值的信息，以及严谨的研究方法
2. One Shot Learning(Generation)是通往AGI的必由之路，因为受限的新环境总是缺少example的，我们的model应该尽量减少对于example的依赖，就像
   人类的one shot learning类似
3. generation model还是非常有趣的，就像我们学习画画，能够画出自己的画带来的快乐大致是多余能够欣赏画的。


### 讨论

后面再读下Lake的那篇Science大作.
