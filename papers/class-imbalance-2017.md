## A systematic study of the class imbalance problem in convolutional neural networks


https://arxiv.org/abs/1710.05381


### 要点

1. 作者的结论
	* class imbalance会对classification performance造成不好的影响
	* 使用oversampling可以很大程度上解决class imbalance的问题
	* oversampling可以从根本上解决class imbalance的问题，而downsampling只能在一定程度上解决
	* oversampling不会引起CNN overfitting
	* 有个threshold需要应用到关注的class上（也就是sampling是有数量要求的）
2. 解决class imbalance的方法
	* data level methods: 对training set进行调整，而不对algorithm(model)进行调整
	* algorithm/model level methods: 对algorithm来调整而不对training set进行调整
	* data level和algorithm level的混合
3. data level methods
	* oversampling: 
		* random minority oversampling: 简单地对minority的class进行replicate，简单有效，但会造成overfitting
		* SMOTE: 通过neighboring data points来生成minority class sample
		* cluster-based oversampling: cluster the dataset, oversample each cluster separately
	* undersampling
		* random: randomly remove the data sample in majority classes to achieve the balance
		* 这样会丢弃一些有效信息
4. algorithm level methods
	* thresholding
	* cost sensitive learning: train against misclassified cost instead of standard cost function, 作者说效果和oversampling是一样的
	* one-class classification
	* hybrid of methods
5. Experiments
	* step imbalance: same number of samples within minority class and same number within majority class
	* linear imbalance: different classes' sample number is in a linear distribution


![Class imbalance](/images/class_imbalance.png)

### 个人点评

1. 之前没有考虑过class imbalance的问题，也没有研究过imbalance对于结果的影响，这个paper给出了一些思路和结论
2. oversampling中的random minority oversampling的方法直接看感觉很傻，因为并没有增加任何新的有效信息（knowledge）
   而只是replicate已有的data samples，但是作者说如此会带来很好的性能改善，倒是出乎意料，因为这个方法的操作性
   很强，很好实现
3. undersampling的性能不好也是可以想像的，因为drop掉了不少的有效信息，这自然会对结果打折
4. oversampling+thresholding似乎是个不错的方案，而且实践起来也比较容易，或许在后续的项目中可以采用
