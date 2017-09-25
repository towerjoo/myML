## ZhuSuan: A Library for Bayesian Deep Learning

http://arxiv.org/abs/1709.05870

By **Tsinghua Univ.**

**A Bayesian DL library built upon TensforFlow**

### 要点

1. 作者提出了Bayesian Deep Learning, i.e DL+Bayesian methods, for unsupervised/semi-supervised learning
2. Zhusuan相比其它类似library的优势(e.g Edward by Tran, PyMC3 by Salvatier)
    * flexibility
    * DL paradigms
    * model reuse
3. DL易于被攻击，因为没有model不确定性，而Bayesian methods可以model不确定性，所以2者的结合可以有更好的结果
4. Zhusuan's Design
    * modeling
        - model primitives: basic structure is DAG, so node primitive(确定的和随机的), graph context
    * inference



### 个人点评

1. 这个清华做的Library从paper来看确实不错，值得仔细看下
2. 准备研究下代码，再看看有没有新的认识和发现



### Resources

1. https://github.com/thu-ml/zhusuan

