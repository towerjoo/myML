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
* [Chapter 4: Fundamentals of machine learning](https://github.com/towerjoo/myML/blob/master/reviews/dl-with-python.md#chapter-4-fundamentals-of-machine-learning)
* [Chapter 5: Deep learning for computer vision](https://github.com/towerjoo/myML/blob/master/reviews/dl-with-python.md#chapter-5-deep-learning-for-computer-vision)
* [Chapter 6: Deep learning for text and sequences](https://github.com/towerjoo/myML/blob/master/reviews/dl-with-python.md#chapter-6-deep-learning-for-text-and-sequences)
* [Chapter 7: Advanced Deep Learning Best Practices](https://github.com/towerjoo/myML/blob/master/reviews/dl-with-python.md#chapter-7-advanced-deep-learning-best-practices)
* [Chapter 8: Generative deep learning](https://github.com/towerjoo/myML/blob/master/reviews/dl-with-python.md#chapter-8-generative-deep-learning)
* [Chapter 9: Conclusions](https://github.com/towerjoo/myML/blob/master/reviews/dl-with-python.md#chapter-9-conclusions)

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

## Chapter 5: Deep learning for computer vision

1. example of convnet
	* a convnet takes as input tensor of shape (image_hieght, image_width, image_channels), e.g for MNIST, it is (28, 28, 1)
2. convolution operation
	* why it's better than dense connected layer?
		- Dense layers learn global patterns in their input feature space, while convolution layers learn local patterns (small 2d window)
		- the pattern convnet learns is translation invariant
		- they can learn spatial hierachies of patterns
	* convolution operates on 3D tensors, called **feature map**, with two spatial axes(i.e height, width) as well as **depth** axis(also known as **channels**)
		- convolution layer converts a input feature map to an output feature map(depth can be arbitrary, no longer stands for channels, but for filters)
		- feature map: every dimension in the depth axis is a feature(or filter), and the 2D tensor output[:, :, n] is the 2D spatial map of the response of the filter over the input
	* key parameters of the convolution layer
		- size of patches extracted from inputs: typically 3X3, or 5X5
		- depth of the output feature map: e.g 32, or 64, etc.
		- in Keras, it is `Conv2D(output_depth, (window_height, window_width))`
		- the output width and height might differ from input, for 1) border effects(can be countered by padding), 2) strides
		- by default, stride=1, but it can be > 1, which means the height/width of the feature map is downsampled by stride(e.g 2), but in practice stride!=1 is rarely used, and we use max pooling to downsample
	* max pooling: extracting windows from the input feature maps and outputting the max value of each channel
		- it's conceptually similar to convolution, except that instead of transforming local patches via a learned linear transformation (the convolution kernel), they’re transformed via a hardcoded max tensor operation
		- max pooling is usually done with 2X2 windows and stride 2, to downsample the feature map by 2, while convolution is often done with 3X3 window without stride(=1)
		- the reason to use downsampling is to reduce the number of feature-map coefficients to process, as well as to induce spatial-filter hierarchies by making successive convolution layers look at increasingly large windows (in terms of the fraction of the original input they cover)
		- max pooling is not the only way to achieve downsampling, other ways include: 1) strides 2) average pooling
		- features tend to encode the spatial presence of some pattern or concept over the different tiles of the feature map (hence, the term feature map), and it’s more informative to look at the maximal presence of different features than at their average presence. So the most reasonable subsampling strategy is to first produce dense maps of features (via unstrided convolutions) and then look at the maximal activation of the features over small patches, rather than looking at sparser windows of the inputs (via strided convolutions) or averaging input patches, which could cause you to miss or dilute feature-presence information.
3. training a convnet on a small dataset
	* data preprocessing
		- read the picture files
		- decode the JPEG content to RGB grids of pixels
		- convert these into floating-point tensors
		- rescale the pixel values(between 0 and 255) to the [0, 1] interval 
	* data augmentation(to mitigate thee issue of too few examples): use Keras's helper function
	* use a pretrained convnet
		- feature extraction: using the representations learned by a previous network to extract interesting features from new samples. These features are then run through a new classifier, which is trained from scratch. representations learned by conv base are likely more generic and therefore more reusable. representations learned by the classifier will necessarily be specific to the set of classes on which the model was trained. Layers that come earlier in the model extract local, highly generic feature maps (such as visual edges, colors, and textures), whereas layers that are higher up extract more-abstract concepts (such as “cat ear” or “dog eye”)
			* fast feature extracting without data augmentation: recording the output of conv_base on your data and using these outputs as inputs to a new model(no need to run the convnet for every input image, so much cheaper)
			* feature extracting with data augmentation: extending the conv_base model and running it end to end on the inputs(each input will go thru the convnet, which is much more expensive, and might require to run on a GPU), need to freeze the conv_base before training(to not change the weights for conv_base)
		- feature tuning: consists of unfreezing a few of the top layers of a frozen model base used for feature extraction, and jointly training both the newly added part of the model (in this case, the fully connected classifier) and these top layers
			* Add your custom network on top of an already-trained base network
			* freeze the base network
			* train the part you added
			* unfreeze some layers in the base network
			* jointly train both the unfreezed layers and the part you added
	* wrapping up
		- Convnets are the best type of machine-learning models for computer-vision tasks. It’s possible to train one from scratch even on a very small dataset, with decent results
		- On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way to fight overfitting when you’re working with image data
		- It’s easy to reuse an existing convnet on a new dataset via feature extraction. This is a valuable technique for working with small image datasets	
		- As a complement to feature extraction, you can use fine-tuning, which adapts to a new problem some of the representations previously learned by an existing model. This pushes performance a bit further
4. Visualizing what convnets learn
	* visualizing intermediate convnet outputs (intermediate activations)
		- the features extracted by a layer become increasingly abstract with the depth of the layer
		-  acts as an information distillation pipeline
	* visualizing convnets filters
		- each layer in a convnet learns a collection of filters such that their inputs can be expressed as a combination of the filters
	* visualizing heatmaps of class activation in an image
		- understanding which parts of a given image led a convnet to its final classification decision

	

![convnet](/images/convnet-cat.png)

![response map](/images/resp-map.png)

![Conv How](/images/conv-how.png)

![Feature extraction](/images/feature-extraction.png)

![fine tuning](/images/fine-tune.png)

![summary](/images/chap5-summary.png)


## Chapter 6: Deep learning for text and sequences

1. working with text data(often in the level of words not character)
	* vectorizing text(remeber NN expects a tensor of float)
		- segment text into words, and transform each word into a vector
		- segment text into characters and transform each character into a vector
		- extract n-grams of words or characters, and transform each n-gram into a vector. N-grams are overlapping groups of multiple consecutive words or characters
	* n-grams and bag-of-words: a way of feature engineering, when using lightweight, shallow text-processing models. DNN won't need to have such explicit groups, while the model itself will learn them implicitly.
	* one-hot encoding of words and characters: qera has the builtin helper function to ease the process
	* word embeddings(learn from data)
		- learn word embeddings jointly with the main task you care about.
		- load into your model word embeddings that were precomputed using a different ML task than the one you're trying to solve(pretrained word embeddings). This is similar to use the pretrained CV models(e.g imagenet models) as the base for the new task, since our training data might be too few.
		- word embeddings(vectors) will reflect the distance between different words to make the subsequent sequence model exploit this useful information
2. understanding recurrent neural networks
	* an RNN is a for loop that reuses quantities computed during the previous iteration of the loop, nothing more
		- due to the gradient vanishing issue, RNN is not useful in practice, use LSTM or GRU instead
	* LSTM: saves information for later, thus preventing older signals from gradually vanishing during processing
3. advanced use of RNN
	* the temperature prediction problem
	* setup the baseline to beat up(e.g we make next day's temperature same as current temperature as the baseline)
		- intuitive baseline
		- simple cheap ML model as baseline
	* figting overfitting
		- recurrent dropout: the same dropout mask (the same pattern of dropped units) should be applied at every timestep, instead of a dropout mask that varies randomly from timestep to timestep
	* stacking recurrent layers(to increase the network capacity)
		- Because you’re still not overfitting too badly, you could safely increase the size of your layers in a quest for validation-loss improvement
		- Adding a layer didn’t help by a significant factor, so you may be seeing diminishing returns from increasing network capacity at this point.
	* using bidirectional RNNs
4. sequence processing with convnets
	* a faster cheaper alternative to RNNs on some problems(esp. NLP tasks, since the position might not be that important, comparing to temperature)
	* use 1D convolutions, extracting local 1D patches (subsequences) from sequences
	* 1D pooling: subsampling, e.g max-pooling, average pooling
5. combine CNNs and RNNs to process long sequences
	* sometimes, sequences are too long and it's too expensive to process directly using RNNs, so using CNNs to downsample(make the sequences shorter, but keep the high level features) might be a good way to proceed

![onehot-vs-wordembedding](/images/onehot-vs-word.png)

![rnn](/images/rnn-unroll.png)

![lstm](/images/lstm-track.png)

![bidirectional rnn](/images/bidirecional-rn.png)

![1d convnet](/images/1d-convnet.png)

![cnn-rnn](/images/cnn-rnn.png)

![summary](/images/chap6-summary.png)

## Chapter 7: Advanced Deep Learning Best Practices

1. Complicated model might have 1+ inputs and 1+ outputs
2. Keras's functional API: each layer can be viewed as a function of input tensor and output tensor
	* multi-inputs model: concatenate inputs
	* multi-outputs model: combine the losses to one to be able to train, e.g sum them up
	* Directed acyclic graphs of layers: e.g Inception module, residual connection
	* layer weight sharing: layer reuse
	* Models as layers: viewed as a *big* function(layer)
3. inspecting and monitoring deep learning models using keras callbacks and TensorBoard
	* keras callbacks: model checkpointing, early stopping, dynamically adjusting parameters, logging metrics, visualing the representations
	* TensorBoard: visualize the detail of model training
4. Getting the most out of your models
	* batch normalization: e.g after a Conv, or a Dense
	* Depthwise separable convolution: separating the learning of spatial features and the learning of channel-wise features
	* hyperparameters optimization: some tools to help finding the optimal hyperparameters automatically(random search, etc.)
	* model ensembling: ensemble different models that are as good as possible, and as different as possible(diversity), e.g blending deep learning with shallow learning

![summary](/images/chap7-summary.png)

## Chapter 8: Generative deep learning

1. Text generation with LSTM
	* sampling process
		- greedy sampling: always sample the one with highest probability(too many repeating)
		- stochastic sampling: sample based on their probability, e.g "a" is .3, then it has 30% to be sampled as the next char
	* control the randomness
		- more entropy: means the results might be more random, and more interesting, surprising
		- less entropy: means the results might be more realistic, but less surprising
		- all candidates have the same probability to sample from: highest entropy, but not realistic(or follow the correct the syntax)
		- greedy sampling: lowest entropy, too many duplications, less interesting
		- based on the softmax probability is a tradeoff of the two extreme conditions, but we can also control the randomness to make it more realistic or interesting
		- add a temperature parameter to control the randomness
2. DeepDream
	* try to maximize the activation of entire layers rather than that of a specific filter, thus mixing together visualizations of large num- bers of features at once
	* start not from blank, slightly noisy input, but rather from an existing image—thus the resulting effects latch on to preexisting visual patterns, distorting elements of the image in a somewhat artistic fashion
	* input images are processed at different scales (called octaves), which improves the quality of the visualizations
3. Neural style transfer
4. Generating images with variational autoencoders(VAE)
	* An encoder module turns the input samples input_img into two parameters in a latent space of representations, z_mean and z_log_variance.
	* You randomly sample a point z from the latent normal distribution that’s assumed to generate the input image, viaz=z_mean+exp(z_log_variance)* epsilon, where epsilon is a random tensor of small values.
	* A decoder module maps this point in the latent space back to the original input image.
5. Introduction to generative adversarial networks(GAN)
	* It’s a dynamic system where the optimization process is seeking not a minimum, but an equilibrium between two forces

![lang model](/images/lang-model.png)

![gen image](/images/gen-image.png)

![summary](/images/chap8-summary.png)

## Chapter 9: Conclusions

1. key concepts in review
	* DL vs ML vs AI (different scopes, and the former one is a subset of the latter one)
	* think about DL: turning meaning into vectors, into geometric spaces, and then incrementally learning complex geometric transformations that map one space to another(spaces of sufficiently high dimensionality)
		- everything is a point in a geometric space
		- Model inputs (text, images, and so on) and targets are first vectorized
		- Each layer in a deep-learning model operates one simple geometric transformation on the data that goes through it
		- This transformation is parameterized by the weights of the layers, which are iteratively updated based on how well the model is currently performing
		- A key characteristic of this geometric transfor- mation is that it must be differentiable, which is required in order for us to be able to learn its parameters via gradient descent(Intuitively, this means the geometric morphing from inputs to outputs must be smooth and continuous — a significant constraint.)
		- meaning is derived from the pairwise relationship between things (between words in a language, between pixels in an image, and so on) and that these relationships can be captured by a distance function(brain might implement meaning in a different way, that we are not exactly sure yet)
	* why DL took off
		- Incremental algorithmic innovations
		- The availability of large amounts of perceptual data
		- The availability of fast, highly parallel computation hardware at a low price, esp. GPU
		- A complex stack of software layers that makes this computational power available to humans, e.g tensorflow, pytorch, keras
	* ML workflow
		- Define the problem
		- Identify a way to reliably measure success on your goal
		- Prepare the validation process that you’ll use to evaluate your models
		- Vectorize the data by turning it into vectors and preprocessing it in a way that makes it more easily approachable by a neural network (normalization, and so on) (it might not be a must, but this preprocessing can boost the learning speed)
		- Develop a first model that beats a trivial common-sense baseline, thus demonstrating that machine learning can work on your problem
		- Gradually refine your model architecture by tuning hyperparameters and add- ing regularization
		- Be aware of validation-set overfitting when turning hyperparameters
	* key network architectures(Densely connected networks, convolutional networks and recurrent networks)
		- Vector data —Densely connected network (Dense layers)
		- Image data - 2D convnets
		- sound data - either 1d convnets(preferred) or RNNs
		- text data - either RNNs(preferred) or 1d convnets
		- timeseries data - either RNNs(preferred) or 1d convnets
		- other types of sequence data - Either RNNs or 1D convnets. Prefer RNNs if data ordering is strongly meaningful (for example, for timeseries, but not for text)
		- Video data—Either 3D convnets (if you need to capture motion effects) or a combination of a frame-level 2D convnet for feature extraction followed by either an RNN or a 1D convnet to process the resulting sequences
		- Volumetric data—3D convnets
2. the limitations of DL
	* In general, anything that requires reasoning—like programming or applying the scientific method—long-term planning, and algorithmic data manipulation is out of reach for deep-learning models, no matter how much data you throw at them
	* All it can do is map one data manifold X into another manifold Y, assuming the existence of a learnable continuous transform from X to Y
	* never fall into the trap of believing that neural networks understand the task they perform—they don’t, at least not in a way that would make sense to us
	* Local generalization vs. extreme generalization(reasoning and abstraction is very hard)
3. the future of DL
	* Models closer to general-purpose computer programs, built on top of far richer primitives than the current differentiable layers
	* New forms of learning that make the previous point possible, allowing models to move away from differentiable transforms
	* Models that require less involvement from human engineers
	* Greater, systematic reuse of previously learned features and architectures, such as meta-learning systems using reusable and modular program subroutines
	* Beyond backpropagation and differentiable layers(when models are used as program, maybe the whole program will not be differentiable)
	* Automated machine learning(AutoML)

![dim mirrow](/images/dim-mirror.png)

![models as program](/images/model-prog.png)

![meta learner](/images/abstraction-learner.png)
		
