## Annotating Object Instances with a Polygon-RNN

http://arxiv.org/abs/1704.05548

### 要点

1. 选中图片中的一个区域（长方形），Polygon-RNN可以给出一个多边形的语义segmentation proposal，来方便对象的标注
2. Polygon-RNN
    * 一个customized VGG16
    * 使用Conv LSTM作为decoder来propose最终多边形的顶点
3. 标注人可以调整某个顶点，那么后续的顶点生成则是以调整的顶点为输入，而不是预测的顶点（有点active learning的意思）
4. 实验和比较

![polyrnn](/images/plyrnn.png)

### 个人点评

1. 这个工作正好可以连接detection和segmentation的工作，无论是作为pipeline的一部分（detection后接一个segmentation）还是作为annotation的辅助工具都是很有意义的
2. 引入active learning的思路也是非常值得学习的（当然也是很直观的）
3. 所谓的基于多边形(polygon)和之前的基于pixel(superpixel等）的思路的差别也是值得学习的


### Resources

1. http://www.cs.toronto.edu/polyrnn/
