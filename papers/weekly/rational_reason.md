## A simple neural network module for relational reasoning

https://arxiv.org/pdf/1706.01427


By DeepMind


### 要点

1. 作者提出了Rational Networks(RN)作为rational reasoning的一个通用解决方案
2. Rational Networks好处
    * learn to infer relations
        - 一般情况下考虑所有对象中任意2个（1对）对象的relation
        - 也可以根据prior knowledge来考虑部分pairs
    * data efficient
    * operate on a set of objects
3. CLEVER dataset
    * 作者使用2D pixel version或者state description version(例如3D坐标，颜色，材料等）,作者的模型中根据不同任务使用其中一个version，而没有同时使用2种version
4. models
    * 使用CNN来parse pixel inputs into a set of objects 
    * 问题与object-object关系是相关的，所以需要把问题的embeding也加入到RN模型中
5. 最终在CLEVER上取得了SOTA的成绩，甚至超过了人类
6. 在总结里提到了RN取得如此大成绩跨越的可能原因，作者认为是将processing和reasoning分开是很重要的，也就是说让CNN来进行视觉的处理，而RN做reasoning


![RN](/images/rn.png)

RN的最简单形式，是一个复合的函数, f, g是MLPs, 所以RNs是end-to-end differentiable. g的输出成为"relation", 所以g用来推理出2个对象是否相关以及如何相关，或者根本无关

![RN](/images/rn_question.png)

Visual QA的架构

### 个人点评

1. 不得不说，看到这样的结果还是非常震惊的
1. 作者抽象层次较高的object，不止局限于visual context,也可以是其它的context，这就使得这样的模型具有了更一般的适应性
2. 有意思的是作者提出的RN可以有效地处理之前更多是NLP领域的text based question answering，也就再一次证明了object不止是visual层面的，也可以是word embeding
3. 相比于之前刚刚阅读的Berkeley的[NMN(Nerual Module Networks)](https://github.com/towerjoo/myML/blob/master/papers/weekly/nmn.md) 在效果上更加让人惊奇（超过人类）

### RNs vs NMN

[NMN](https://github.com/towerjoo/myML/blob/master/papers/weekly/nmn.md)是Berkeley最近出的一篇关于reasoning的paper，和RNs在场景上有一些类似的地方，下面我说下2者的差别

1. NMN是训练出一些关系的module（类似于编程语言的标准函数），然后NN来根据问题来选择所要应用的modules，最终得出结果, 这里关键的就是这些module的确定（这也是局限所在，很多问题无法表达成简单的module的组合）以及NN可以正确选择module
2. RNs则是end to end的，确定object的representation（例如CNN，或者word embeding或者其它的representation），形成object pair with question，再通过g(MLP)等RN单元来得出结果. 显然这里关键的就是object representation
3. 直观地看，感觉NMN更像是传统的编程方式的映射，也就是问题有比较好的定义，我们可以定义出这些Module，而RNs则更加ML范，让DL来自己做reasoning

### Resources

1. https://hackernoon.com/deepmind-relational-networks-demystified-b593e408b643
2. https://deepmind.com/blog/neural-approach-relational-reasoning/
3. https://news.ycombinator.com/item?id=14494935
4. https://github.com/kimhc6028/relational-networks
