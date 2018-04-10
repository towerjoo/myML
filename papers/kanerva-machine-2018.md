## THE KANERVA MACHINE:A GENERATIVE DISTRIBUTED MEMORY

by **DeepMind**


### 要点

1. efficiently use memory
	* DNC: slot based external memory, not shared between slots
	* other approaches might need a big amount of memory or drop some information when averaging embedding
	* Kanerva: sparse distributed memory
2. VAE
3. Kanerva Machine
	* generative model
	* reading inference model
	* writing inference model: trade-off between preserving old information and writing new information(using Bayes rule)
4. discussion
	* combines slow-learning neural networks and a fast-adapting linear Gaussian model as memory
	* generalise Kanerva’s memory model to continuous, non-uniform data while maintaining an analytic form of Bayesian inference
	* learns to to store information in a compressed form by taking advantage of statistical regularity in the images via the encoder at the perceptual level, the learned addresses, and Bayes’ rule for memory updates.
	* employ an exact Bayes’ update-rule without compromising the flexibility and expressive power of neural networks


VAE
---
![VAE](/images/vae.png)

Generative Model
---

![gen-model](/images/gen-model.png)

Reading Inference Model
---

![inf-model](/images/inf-read.png)

Writing Inference Model
---

![inf-model](/images/inf-write.png)

Training Loss Function
---

![train](/images/kanerva-train.png)

### 个人点评

1. DeepMind的文章很严谨，但是通常也是math heavy的，通常对于engineering backend的同学不友好，这篇也是一样
2. one-shot learning自然是很重要的，而其中的math还需要一些时日才能理解，或许流程应该是: 读懂->理解->修改（创建自己的model/machine）

### 讨论

math还需要加强。甚至当看到很多所谓的*Math for Deep learning*的书籍或者paper后，还是很难理解这里的term和公式，或许还需要一些更系统的
math学习。
