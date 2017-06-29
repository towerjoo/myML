## Neural Module Networks

http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Andreas_Neural_Module_Networks_CVPR_2016_paper.html

By Berkeley


### 要点

1. contributions
    * NMN, a general architecture for discretely com- posing heterogeneous, jointly-trained neural modules into deep networks
    * construct NMNs based on the output of a semantic parser, and use these to successfully complete established visual question answering tasks
    * introduce a new dataset of challenging, highly compositional questions about abstract shapes, and show that our model again outperforms previous approaches
1. Rather than thinking of question answering as a problem of learning a single function to map from questions and images to answers, it is perhaps useful to think of it as a highly-multitask learning setting, where **each problem instance is associated with a novel task**, and the identity of that task is expressed only noisily in language
    * 黑体部分的视角还是不错的
    * 不是训练出一个question->answer的function，而是每个问题是一个全新的任务
1. 若干个submodule，使用language parser将自然语言处理为layout，然后映射到某些submodule组成一个树，evaluate这棵树就是最终的结果
    * parser: Stanford Parser
    * submodules
        * find: image-> attention, find[c]
        * transform: attention -> attention, transform[c]
        * combine: attention X attention -> attention, combine[c]
        * describe: image X attention -> label
        * measure: attention -> label
    * e.g where color is the cat? -> `describe[color](find[cat])`
1. 网络结构和不同dataset的效果

![NMN](/images/nmn.png)

big picture of the network


### 个人点评

1. 这种方法还是很有意思的，就像编程语言的library一样，后续的reasoning可以是一些原语的组合(submodule），当然submodule本身可能不容易描述，而且没有一个fit all的原语。就像排序，可能需要写一个针对数字的排序，和一个针对字符串的排序，同样可能需要对🐱和🐶的不同的submodule
2. 因为是jointly learning(piped)，某一个环节的准确度直接影响后一阶段的结果，这也是很难避免的，当然如果是end to end的网络，而不是2个就可以在一定程度上避免这个问题
3. 总之，这个方法还是很喜欢的，作者也有一个follow up的paper值得一读


### Resources

1. https://github.com/jacobandreas/nmn2
