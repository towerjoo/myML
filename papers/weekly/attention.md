## Attention is ALL You Need

http://arxiv.org/abs/1706.03762

By Google

**第一个完全基于attention的sequence transduction model，具有更高的训练效率（并行）和更高的准确度**


### 要点

1. RNN等sequence model受限于需要t-1的输出作为t的输入，所以只能串行，而不能并行，所以限制了整个的训练效率。这也是为什么之前Facebook的组提出了用CNN来做sequence model的原因
2. 此paper提出的Transformer，丢弃不能并行的RNN部分，而直接使用attention来做sequence model，减少了训练时间也提高了准确度
3. model architecture: see the image below
3. An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. 
4. Self Attention(compare to RNN, CNN)
    * computational complexity of each layer
    * amount of computation which can be parallelized
    * path length between long-range dependencies of network


![transformer](/images/transformer.png)

model architecture 


### 个人点评

1. 并行化是很重要的一个研究方向，因为目前制约DL的很大一个因素就是training time和evaluation time，这也是为什么之前Facebook将CNN引入机器翻译的原因
2. RNN等类似的时序Model因为本身的限制（当前操作依赖于前一时序的输出）所以很难并行化，这也是Google的这个论文的价值所在
3. 更重要的是，此论文提出的model不止提高了并行性（训练效率），并且保持了更好的准确度（事实上二者很多时候是不可兼得的，Google这次做的很棒）
4. 当然像这样novel和相对比较复杂的model还是需要长时间在领域里面的钻研



### Resources

1. https://github.com/tensorflow/tensor2tensor
2. https://github.com/hoangcuong2011/Good-Papers/blob/master/Attention%20Is%20All%20You%20Need.md (also a reading note about this paper)
