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
* [Chapter 3: Getting started with neural networks](https://github.com/towerjoo/myML/blob/master/reviews/dl-with-python.md#chapter-3-getting-started-with-neural-networks)

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

## Chapter 3: Getting started with neural networks

1. layers: the building blocks of deep learning
	* layer compatibility: every layer will only accept input tensors of a certain shape and will return output tensors of a cer- tain shape
		- layer = layers.Dense(32, input_shape=(784,)): a layer that will only accept as input 2D tensors where the first dimension is 784 (axis 0, the batch dimension, is unspecified, and thus any value would be accepted). This layer will return a tensor where the first dimension has been transformed to be 32.
		- 上面代码 the model will take as input arrays of shape (*, 784), 也就是说batch size可以任意取值, output arrays of shape (*, 32)
		- Keras中第一个layer需要specify input_shape，其他的layer都不需要显式的设置，Keras会自动确定（前一个layer的output shape和后一个layer的input shape匹配）
2. models: a directed, acyclic graph of layers
	* By choosing a network topology, you constrain your space of possibilities (hypothesis space) to a specific series of tensor operations, mapping input data to output data(缩小搜索空间)
	* more an art than a science （所谓炼丹师😅）
3. Loss functions and optimizers:keys to configuring the learning process
	* for multiloss networks, all losses are combined (via averaging) into a single scalar quantity
	* binary crossentropy for a two-class classification problem
	* categorical crossentropy for a many-class classification problem
	* mean-squared error for a regression problem
	* connectionist temporal classification (CTC) for a sequence-learning problem
	* 当然如果我们要提出我们自己的Net(ResNet, AlexNet等），我们可能需要设计自己的loss function
4. IMDB example
	* we need to preprocess the input data to make sure they have the same length(e.g one-hot encoding, etc.)
	* how many layers to use
	* how many hidden units to choose for each layer: a bigger number means a higher dimensions, i.e more computations(representation capacity), but might overfit
	* plotting the history data is interesting and helpful(since it might be overfitting very soon)
5. classifying newswires
	* one lable multiclass classification problem
	* the hidden layer should have >46 dimensions to have enough representation capacity(<46 will make the output after that layer drop some information, which cannot be recovered in the subsequent layers)
	* you might need to one-hot encode the input data(or use the builtin *to_categorical* util function, do the same thing)
	* the output layer should have N dimensions(N == the class number)
	* multiclass classification we often use *categorical_crossentropy* loss fuction(compare to binary classification's *sigmoid*)	
	* also need to encode the labels
		- one-hot encoding using *to_categorical*(the loss function will be *categorical_crossentropy*)
		- encoding as integers(the loss function will be *sparse_categorical_crossentropy*)
5. housing prediction
	* regression problem
	* normalize the data(training and test)
		- if not, the training process will be much harder(but not impossible)
		- normalize(data) = (data - mean) / std;
		- never use the test data to calculate mean and std(you should not touch the test data when training, but you can apply the normalization using the mean/std calculated using the training data)
	* for regression, we often use *mean squared error*(i.e MSE) as the loss function
	* we also use the *mean absolute error*(i.e MAE) as the metric
	* if the training data is too few, we might use *k-fold* validation
	

![keras](/images/keras-arch.png)

![chap3 summary](/images/chap3-summary.png)


## Chapter 4: Fundamentals of machine learning

1. four branches of ML
	* supervised learning: examples with label to learn the mapping from input to output
		- most popular ones
		- sequence generation
		- syntax tree prediction
		- object detection: regression(bounding box coordination) + classification(for the object)
		- image segmentation
	* unsupervised learning
		- dimensionality
		- clustering
	* self-supervised learning: supervised learning without human in the loop
		- next frame for a video given past frames, next word in the sentence
		- autoencoder
	* reinforcement learning
		- maximize reward
		- self-driving cars, robotics, resource management, education
2. evaluate ML models
	* split data into training, validation, test
	* recipes
		- simple hold-out validation: train=S[:n], val=S[n:], suffer with a small S
		- k-fold validation: K partitions with same size, for each partition i, train against remaining K-1 samples, validate against the i-th samples; final score is the average
		- iterated k-fold validation with shuffling: apply k-fold validation many times and shuffle before each time, evaluate PxK models(P: iterations, K: folds)
	* tips
		- data representativeness: if the data are in order, you might need to shuffle to gain a better representativeness
		- the arrow of time: pick the test set **posterior** to the data in training set
		- redundancy in data: make sure training data and validation data are disjoined
3. data preprocessing, feature engineering, feature learning
	* data preprocessing
		- vectorization: inputs/targets for NN should be tensors of float
		- normalization: take small values(typicially between 0 and 1), be homogenous(all data roughly in same range), often(but not always necessary) to ensure mean=0, std=1
		- handling missing values: it's often safe to input missing values as 0(if 0 is not a meaningful value in data), NN will learn to ignore 0 for the feature value; if test data have missing values, but training data don't, we might need to fill some missing samples to training data
	* feature engineering: the process of using your own knowledge about the data and about the machine-learning algorithm at hand (in this case, a neural network) to make the algorithm work better by applying hardcoded (nonlearned) transformations to the data before it goes into the model
		- making a problem easier by expressing it in a simpler way
4. overfitting and underfitting
	* get more training data
	* regularization
		- reduce the network size: bigger size, bigger capacity(memorization capacity), which means a perfect dictionary-like mapping between input and output(hard to generalize); with a smaller capacity, model tends to learn the compressed representation which has a better prediction
		- add weight regularization: simpler model tends to generalize better(Occam's razor), simple means the distribution of the parameters has less entropy(or fewer parameters), normally make the weights to take small values(i.e more regular) using weight regularization. L1 regularization: absolute value of the weight coefficients(L1 norm of weights); L2 regularization: square of the value of the weight coefficients(L2 norm) also called **weight decay**
		- add dropout: introducing noise in the output values of a layer can break up happenstance patterns that aren’t significant (what Hinton refers to as con- spiracies), which the network will start memorizing if no noise is present. drop out randomly a number of output features of the layer during training.
5. universal workflow of ML
	* define the problem and assemble the dataset
	* choose a measure of success
	* decide on an evaluation protocol
	* prepare data
	* develop a model that does better than the baseline
	* scaling up: developing a model that overfits
		- add more layers
		- make the layers bigger
		- train for more epochs
	* regularize your model and tune your hyperparameters
		- add dropout
		- different architectures: add/remove layers
		- add L1/L2 regularization
		- different hyperparameters
		- feature engineering: add/remove features	

![clock](/images/fe-clock.png)

![regularization](/images/reg-chart.png)

![drop out](/images/dropout-chart.png)

![act and loss](/images/act-loss.png)

![summary](/images/chap4-summary.png)
