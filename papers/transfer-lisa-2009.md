## Transfer learning

https://books.google.co.kr/books?hl=en&lr=&id=gFpKXO8H_6YC&oi=fnd&pg=PA242&dq=Transfer+Learning+lisa+torrey&ots=6LUbPSmxip&sig=SoQAX8UvQK4s_ggeQuT_fyapYVY

### 要点

1. 3个可能的好处：
    * 直接使用source model的参数应用在target上
    * 使用source model的知识来学习target model的时间与从零学习target model的时间比较
    * 使用source model的知识来习得的target model与不使用source model习得的target model的准确度对比
2. 如果上面3个指标出现不及预期（transfer learning应该会帮助target model的学习）,则称为negative transfer
3. inductive transfer: target task inductive bias is chosen or adjusted based on the source task knowledge
4. Bayesisan Transfer
5. hierarchical transfer:
6. avoid negative transfer
    * rejecting bad information: option based transfer(partly choose information based on the effect)
    * choose a source task
7. mapping task: sometimes source/target task might need to do the task translation to make target understand the knowledge from source task


### 个人点评

1. 事实上，人类无时无刻不在使用transfer learning来认识世界，那么人类是如何决定, when to transfer/what to transfer/how to transfer都是一些未解之谜，而对于机器这些问题就更难了
2. 定义2个task的相似度是个非常核心的问题，这样我们就可以迅速地在source task sets里面搜索到最相关的source task来transfer knowledge
3. 当然我们人类或许不只是选择一个最相关的source task，而是综合不同的source task来进行部分的target task相关的transfer，这就对我们的ML提出更高的要求
4. 我们人类对于已有模型显然不止存储一个维度的feature space，而是多个维度的feature space，我们在transfer learning时可以从不同的source task中选择，从同一source task的不同维度的feature space来选择transfer的知识，从而帮助target task的学习，这是多么美妙的学习过程，机器还是差的太远
5. 当然这篇survey比较久远，新近的进展还是值得关注的

