## Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks


CycleGAN

https://arxiv.org/abs/1703.10593

### 要点

1. unpaired是比较大的一个创新，也就是不要求有paired的数据，这样dataset的要求就不高
2. cycle这里说的是2组(不同的demain)的D和G，从而形成了一个环状的结构，也就是Dy和G进行对抗，
   Dx和F进行对抗，x,y分属于X和Y domain. 那么Lgan(G, Dy, X, Y)就是第一组的loss function, 
   Lgan(F, Dx, X, Y)是第二组的loss function，再根据cycle consistence，也就是F(G(x))=x(尽可能）以及
   G(F(y))=y(尽可能）来训练模型
3. 具体训练：作者使用batch size=1, learning rate=0.0002来训练，使用least square loss来替代negative log
   likelihood(作者说LS更稳定）,另外在更新Dx, Dy时使用的是历史生成的图片，而不是最近生成的图片(内存cache
   50张最近生成的图片，作者说是为了减少model oscillation)


### 个人点评

1. 生成网络相比判定网络在unsupervised方向又进了很大一步，而这种unpaired的img2img translation更向前推进了一步
2. 目前使用人工的AMT来评估GAN模型显然在reproduce/scale等方面都差太远，期待更加客观的评估标准和方法出来


### References

1. mode collape: http://aiden.nibali.org/blog/2017-01-18-mode-collapse-gans/

