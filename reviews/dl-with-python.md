# Deep Learning with Python

## Overall Review

Intuition building is very important and sometimes it requires a great gift from writers to convey.
Definitely this book does a great job on intuition building. With the intuition, you won't be lost in
the dark forest(if it's proper to say DL is like a dark forest), and cannot jump out.

It's also very suitable for the readers from engineering background, rather than academic background, which
means "using less math, more codes". This might give you an impression of shallow understanding(buddy, we're learning **deep learning**, not
*shallow learning* right?), and in some perspective that's the case, but that's also a wise tradeoff. The author
might not want to scare you away with a bunch of math terms, while he just wants to usher you into this field and
gives you some intuition about it, or even gives you some practical codes to see some magic with your own machine. Bearing that in your
mind, you won't be disappointed.

After the **shallow learning** of *deep learning* from this book, you can find some other books with heavy math if you like. And finally you
can gain a solid grasp towards the theory, but also you can build the real world AI applications as an AI engineer.

And I gave my chapter based reading notes as below.

Quick Link

* [Chapter 1: What is Deep Learning](https://github.com/towerjoo/myML/blob/master/reviews/dl-with-python.md#chapter-1-what-is-deep-learning)
* [Chapter 2: the mathematical building blocks of neural networks](https://github.com/towerjoo/myML/blob/master/reviews/dl-with-python.md#chapter-2-the-mathematical-building-blocks-of-neural-networks)
* [Chapter 3: Getting started with neural networks]()

## Chapter 1: What is Deep Learning?

1. symbolic AI: human crafts machine with predefined rules
	* more rules will cover more conditions/situations/tasks, i.e *more intelligent*
	* expert system
	* vulunerable to change/unseen situation
2. Turing Test
3. learn from data, i.e **machine learning**
	* pattern/rule recognition
	* more engineering-oriented rather than theory oriented(comparing to statistics)
4. representation of data(or encoding)
	* different way to look at data(some famous encoding methods for NLPs, CVs, etc.)
	* **Machine-learning models are all about finding appropriate representations for their input data**
	* Learning, in the context of machine learning, describes an automatic search process for better representations
	* ML: searching for useful representations of some input data, within a predefined space of possibilities, using guidance from a feedback signal
5. Deep Learning
	* puts an emphasis on learning successive layers of increasingly meaningful representations
	* a multistage way to learn data representations
	* it allows a model to learn all layers of representation jointly, at the same time, rather than in succession (greedily, as it’s called)
6. other ML methods
	* Probabilistic modeling(statistics to data analysis, e.g Naive Bayes algorithm)
	* Kernel methods(e.g SVM)
		- can introduce the nonlinear transformation to make the data points seperatable
		- kernel function(map from initial space to target representation space)
	* decision trees, random forests, gradient boosting machines
7. how about stacking layers of shallow model?
	* fast-diminishing returns to successive applications of shallow-learning methods
8. modern ML landscape(from Kaggle)
	* **gradient boosting** is used for problems where structured data is available, shallow-learning problems(e.g XGBoost library)
	* **deep learning** is used for perceptual problems such as image classification
9. why now for DL
	* hardware
	* data
	* algorithm(esp. the gradient propagation, i.e vanishing or explosion problem)
	* investments
	
![dl](/images/dl-arch.png)
**The big picture of DL**

## Chapter 2: the mathematical building blocks of neural networks

1. a basic neural network
	* MNIST dataset(buildin to keras, the helloworld problem of DL)
	* The core building block of neural networks is the layer, a data-processing module that you can think of as a *filter for data*
	* some data goes in(to layer), and it comes out in a more useful form
	* layers extract **representations** out of the data fed into them(representations here might not be something that we human understand)
	* Dense layer(used in Keras) means **full connected layer**(densely connected, abrv. FC)
	* to train(or compile), we also need
		- loss function: how to measure its performance on the training data
		- optimizer: how to update the network parameters based on the loss function
		- metrics to monitor between training and testing: when to stop, accuracy, etc.
2. data representations for neural network
	* tensor: tensors are a generalization of matrices to an arbitrary number of dimensions(for tensor, dimension is also called axis)
		- matrix: 2D tensors, rectangular grid of numbers
		- vectors: 1D tensors, contains an array of numbers
		- scalars: 0D tensors, contains only one number
		- 3D or higher dimensional tensors: e.g the mnist dataset, the whole training set is something like (60000, 28, 28) 3D tensor.
	* tensor key attributes
		- number of axes(rank): also called *ndim* in numpy
		- shape: how many dimensions along each axis
		- data type: or dtype in numpy
	* data batches: in general, the first axis (axis 0, because indexing starts at 0) in all data tensors you’ll come across in deep learning will be the samples axis(i.e the number of samples)
		- DL handles samples in batch
		- batch = train_images[:128]
		- batch = train_images[128:128*2]
		- When considering such a batch tensor, the first axis (axis 0) is called the batch axis or batch dimension
	* real world data tensors
		- vector data: 2D tensors (samples, features)
		- timeseries data or sequence data: 3D tensors (samples, timesteps, features), by convention, **time axis is always the second axis.**
		- images: 4D tensors (samples, height, width, channels) or (samples, channels, height, width), grayscale image has 1 dimension channel, and color images have 3 dimensions channel
		- video: 5D tensors (samples, frames, height, width, channels) or (samples, frames, channels, height, width)
3. tensor operations
	* keras.layers.Dense(512, activation='relu') can be interpreted as `output = relu(dot(W, input) + b)`, input is a 2D tensor 
		- dot product
		- add
		- relu: max(0, x)
	* element-wise: e.g relu applied to tensors, which can be efficiently parallel executed
	* broadcasting: e.g add between a matrix(2D) and a vector(1D), i.e operations applied to two tensors with different shapes. we can understand broadcasting as below(but it's not implemented in that way)
		- Axes (called broadcast axes) are added to the smaller tensor to match the ndim of the larger tensor
		- The smaller tensor is repeated alongside these new axes to match the full shape of the larger tensor
	* dot product: is not like element-wise multiply(in numpy, this is done by A * B), for dot product, it's np.dot(A, B), it requires A.shape[0] == B.shape[1]
		- 我们所谓的矩阵乘法，而不是点乘
	* tensor reshaping: rearange its rows and columns to match a target shape
		- it's very possible the shape a layer expects is different from the shape of input, so we need to **reshape**
		- 这个是最容易出错的地方，特别是对于高维的tensor
		- 还记得Andrew Ng在ML课上经常提到的对于维度的检查？这也是线性几何中矩阵运算做容易出错的地方
4. geometry explanation
	* finding neat representations for complex, highly folded data manifolds
	* incrementally decomposing a complicated geometric transformation into a long chain of elementary ones
5. training loop
	* Draw a batch of training samples x and corresponding targets y.
	* Run the network on x (a step called the forward pass) to obtain predictions y_pred.
	* Compute the loss of the network on the batch, a measure of the mismatch between y_pred and y.
	* Update all weights of the network in a way that slightly reduces the loss on this batch(the most difficult step, employ gradient based optimization)
6. gradient-based optimization
	* A gradient is the derivative of a tensor operation
	* with a function f(W) of a tensor, you can reduce f(W) by moving W in the opposite direction from the gradient: for example, W1 = W0 - step * gradient(f)(W0) (where step is a small scaling factor)
	* SGD: stochastic gradient descent, stochastic means random, i.e picks n samples randomly, n can be 1 to N(the whole samples)
		- mini-batch SGD is often used, e.g n=128
	* in reality, such approach is almost impossible to reach the global minimum, but a local minimum is often acceptable
	* when updating weights, we also have some **optimizers**
		- momentum: convergence speed and local minima(compare to physics, a small ball rolling down the loss curve, count both *slope* and *velocity*)
		- Adagrad
		- RMSProp
7. backpropagation algorithm(BP)
	- calculate derivative using **chain rule**
	- symbolic differentiation: (used in TensorFlow), compute a gradient function for the chain (by applying the chain rule) that maps network parameter values to gradient values, this will make the calculation very easy and only need to call such function


![dot product](/images/dot-product.png)


![chap2 summary](/images/chap2-summary.png)
