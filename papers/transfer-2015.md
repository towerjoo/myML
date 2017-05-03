## Transfer learning for visual categorization: A survey

http://ieeexplore.ieee.org/abstract/document/6847217/

### 要点

1. 3 questions about transfer learning
    * when to transfer
    * what to transfer
        - inductive transfer learning: 所有的source domain的example和label都在target domain中使用
        - instance transfer learning: 只有source domain的example在target domain中使用，label不使用
        - parameter transfer learning: 除了example和label，source domain的Model parameter也会在target domain中使用
    * how to transfer
2. 衡量source domain和target domain的distribution similarity是非常重要的，只有similarity比较高时，这种transfer才是有意义的。
3. maximum mean discrepancy(MMD)用来衡量source/target domain的分布相似度
4. 在Visual Categorization领域作者使用： feature representation level knowledge transfer和classifier level knowledge transfer
    * feature level: 需要详细的feature engineering来减少source/target domain的数据差别，来提高transfer learning的效果
        - cross-domain: 作者提到了一些lower dimension feature space通常更可能在不同的domain得以共享，例如edges等。也就是说，不需要source/target domain完全相似，而是在lower dimension有相似的feature就可以进行有效的transfer learning。类比人类的学习，我们学习过猫狗后，会对动物有一定的feature识别，那么相应地在学习长颈鹿时这些Feature的识别会有帮助。
        - cross-view: 也就是不同的view-point，这是需要使用一些几何的推理（平移，旋转，投影等），或者更高维的抽象（例如3D）来得到相同的feature
    * classifier level: 不只source domain的instance/label，而且使用学得的source domain的model作为target domain的prior knowledge
        - SVM based
        - TrAdaBoost
        - generative models: Fei-Fei's Bayesian based unsupervised one-shot learning
        - fuzzy system-based models
5. model selection: 当source model不止一个时，如何选择合适的Model来进行transfer learning, 通常可以通过计算source/target similarity来进行model选择
6. 分析
    * transfer learning好处
        - higher start: improved performance at the initial starts
        - higher slope: more rapid growth of performance
        - higher asymtote: improved final performance
    * 作者的一些实验对比不同的方法和模型

7. 结论
    * the feature representation level knowledge transfer aims to unify the mismatched data in different visual domains to the same feature space and the classifier level knowledge transfer aims to learn a target classifier based on the parameters of prelearned source domain models, while considering the data smoothness in the target domain.


### 个人点评

> Transfer learning is a tool for improving the performance of the target domain model only in the case that the target domain labeled data are not sufficient, otherwise the knowledge trans- fer is meaningless.

作者有上面的说法很奇怪，transfer learning当然不止在target model的labeled data不足时有意义，而且即使有target model有足够的labeled data，transfer learning也可以加速学习过程（作者提到的3个higher)，这就像是说我们的计算机性能足够高时，我们就没有必要优化算法一样的荒谬。

(给作者写了一封请教这个问题的邮件希望有回复）
