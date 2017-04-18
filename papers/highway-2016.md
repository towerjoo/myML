## Highway Networks

https://arxiv.org/abs/1505.00387

highway network就是在普通变换中加入transform gate和carry gate(主要是transform gate)使得输入可以不经改变流入到下一层。

### 要点

1. 为了解决深度网络难于优化的问题
1. 受LSTM启发，使用一个learned gating mechanism来作用于信息，使得一些信息可以不经过改变而流入下一层（或者下几层）, 作者把信息可以不经改变流入成为highway
1. 传统的每层的变换可以表示为: y=H(x,W_H) W_H是H变换的参数，H通常是affine transform(仿射变换, 例如平移，缩放，翻转，旋转和错切等）接一个非线性激活函数
1. 作者提出的变换为: y=H(x, W_H)T(x, W_T)+xC(x, W_C), 这里T是transform gate, C是carry gate, 文中作者使用C=1-T, 于是可以看到T=0 => y=x 在输出保留了输入； T=1 => y=H(x, W_H) 成为了普通网络的变换
1. highway network 通过控制T/C可以平滑控制每层网络在保持输入不变和普通网络变换之间的行为
1. highway network变换中的向量都必须是相同维度的，对于维度的不同，可以通过padding或者plain layer来变换得到
1. 作者提到了初始化对于普通网络的重要性,而这个初始化过程依赖于H，highway network作者使用T(x)=sigmoid(W_Tx+b)作为transform gate(值介于0，1之间），那么highway network的初始化不再受限于H，并且作者建议b初始化为负值，使得初始的训练偏好于carry behavior 
1. 保留上一层的输入信息对于整个优化过程是非常重要的，否则更深的层会出现梯度消失或者爆炸的问题


### 个人点评

1. 是在看到ResNet的评价是回过头来看这篇论文的，这篇是早于ResNet那篇的，就像Schmidhuber对于ResNet的评价：是highway network的特例，是没有gate的feedforward LSTM
1. 所有的研究都指向越深的网络会有越好的结果，但是一个直接的问题就是每层网络都会对输入进行一些处理，从而导致越深层得到的输入variance(导致当前层的输入过大或过小，从而使得激活函数失去非线性，例如sigmoid函数在输入过大或过小时都更接近于线性变换）更大使得优化会更加困难
1. RNN/LSTM的借鉴是很重要的，无论是从时间的角度，还是从保持输入的角度，越深的网络会造成variance的增加，所以gate的思路（放过一些施加的处理）还是很值得借鉴的
