## Visual Interpretability for Deep Learning: a Survey


### 要点

1. this paper's scope:
	* Visualization of CNN representations in intermediate network layers
	* Diagnosis of CNN representations
	* Disentanglement of “the mixture of patterns” encoded in each filter of CNNs.
	* Building explainable models
	* Semantic-level middle-to-end learning via human- computer interaction.
2. Visualization of CNN representations
	* gradient-based methods: compute gradients of the score of a given CNN unit w.r.t. the input image
	* the up-convolutional net: inverts CNN feature maps to images
	* a method to accurately compute the image-resolution receptive field of neural activations in a feature map
3. Diagnosis of CNN representations
	* analyze CNN features from a global view
	* extracts image regions that directly contribute the network output for a label/attribute to explain CNN representations of the label/attribute
	* estimation of vulnerable points in the feature space of a CNN
	* refine network representations based on the analysis of network feature spaces
	* discover potential, biased representations of a CNN
4. Disentangling CNN representations into explanatory graphs & decision trees
5. Learning neural networks with interpretable/disentangled representations
	* interpretable CNN
	* interpretable R-CNN
	* Capsule networks
	* Information maximizing generative adversarial nets
		- The InfoGAN maximizes the mutual information between certain dimensions of the latent representation and the image observation
6. Evaluation metrics for network interpretability
	* Filter interpretability
	* Location instability
7. Network interpretability for middle-to-end learning
8. conclusion: The approach for transforming a pre-trained CNN into an explanatory graph has been proposed and has exhibited significant efficiency in knowledge transfer and weakly-supervised learning.

![decision](/images/cnn-explain.png)

### 个人点评

explainable NN 是一直以来DL研究人员很关注的领域，因为why/how总比what要重要，也是确保和我们预期相符的基础（inference的过程，是否有歧视, bias等），
但是如何研究可解释性，怎么评估可解释性也是目前来看比较困难的，但是结果往往与我们的直观预期是不同的，例如我们期望的解释性的直觉是视觉感知下的解释，
例如对于判定一只猫的解释，头、尾巴等，但研究领域最终形成的方法往往与我们直觉是不同的，就像我们梦想可以飞是我们自己可以自由飞翔，当时实现的结果是
我们坐飞机飞行。

### 讨论

可解释性是非常基础的研究，也是后续应用的基础（例如军事、医学），我们不止要看到好的结果，还要足够自信整个推理的过程，这样我们才能够
放心地来应用DL到各个领域。
