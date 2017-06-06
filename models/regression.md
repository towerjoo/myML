# 回归模型

## 基本信息

回归通常用于连续变量的预测，例如房价预测，股市预测等。

## How to evaluate

### 残值分析

![res](/images/res.png)

简单地说就是上面的公式, 对于预估值和实际值的差值进行分析。

### Coefficient of determination R^2 

https://en.wikipedia.org/wiki/Coefficient_of_determination

![def](/images/r2.png)

也就是这个公式：

![formula](/images/r2_formula.png)

在线性回归中，整个训练过程就是为了最小化SS_{res} 残差, 使得预测值和
实际值更加接近，作为表征model好坏的标准。

这里R^2有个问题是当参数增加时，R^2通常会增加，而这个不是我们所期待的，所以
引入了adjusted R^2来解决这个问题。

在常用的scikit-learn library中，对于regression通常使用的是R^2
