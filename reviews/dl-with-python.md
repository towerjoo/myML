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
