## On the Expressive Power of Deep Architectures

http://www.iro.umontreal.ca/~lisa/bib/pub_subject/finance/pointeurs/ALT2011.pdf

By **Bengio**


### 要点

1. knowledge is hard to describe
2. learning alg: dataset -> function
3. depth of architecture
    * fixed-kernel SVM: depth 2
    * boosted decision trees: depth 3
4. how a alg generalize from training examples to new cases? 
    * target distribution的一些prior（先验假设）
    * 最常见的prior: local generalization(smoothness assumption), 也就是说target function is smooth
5. expressive power of Deep architecture
6. 作者以sum-product network为例，说明了deep network比shallow network要高效很多(node的数量等，shallow的节点数是deep的几何倍数)
7. some algorithms
    * Boltzmann Machines
    * the Restricted Boltzmann Machines(RBMs), stacked to form a Deep Belief Network(DBN)
    * Auto-encoders
    * sparse coding
    * score matching
    * Denoising Auto-Encoders
    * Noise-Contrastive Estimation
    * Semi-Supervised Embedding
    * Contractive autoencoders
8. 一些想法: 
    * some models may work well in practice because the optimization is easier
    * optimization difficulty means that the optimization problem is not cleanly decoupled from the modeling choices
9. some tricks:
    * independence
    * sparsity
    * grouping
    * slowness
        

### 个人点评

1. Bengio大神的Paper还是很值得一读的，因为他严谨的治学态度，记得看过其在某篇博文下的严谨而耐心的回复
2. 这样的review会让我们知道其实很多想法在20，30年前已经被研究了，例如表达能力，autoencoders等等，而当下我们很多最新的model也是基于这些想法的改善，或者只是结合新的技术，如VAE(GAN)等
3. 文中说明depth相比shallow的network可以很高效（node的数量几何级节约）,但是遗憾的是只是针对线性的变换，而实际中的deep network都是非线性的（各种非线性的activation function）

