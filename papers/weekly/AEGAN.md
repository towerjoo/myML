## Variational Approaches for Auto-Encoding Generative Adversarial Networks

http://arxiv.org/abs/1706.04987

By DeepMind


### 要点

![objective function](images/vaegan.png)

1. mode-collapse: generated data does not reflect the diversity of the underlying data distribution
2. AE-GAN
    * AE as the discriminator, e.g energy based GANs, or boundary-equilibrium GANs
    * denoising AE to derive an auxiliary loss for generator, e.g denoising feature matching GANs
    * combine the ideas from VAE and GAN, e.g VAE-GAN
3. 作者也是将VAE和GAN结合起来，因为
    * VAE经常产生模糊的图片，they don't suffer from the mode collapse
    * GAN allow few distributional assumptions to be made about the model
    * 也就是一个使得更加泛化，一个使得更加精确
4. network结构
    * classifier D: 用来区分来自AE生成的和实际的数据
    * classifer: 用来区分来自encoder生成的latent sample和标准高斯的sample
    * deep generative model
    * encoder network
5. evaluation metrics
    * Inception score: uses a pre-trained neural network classifier to capture to two desirable properties of generated samples: highly classifiable and diverse with respect to class labels.
    * Multi-scale structural similarity(MS-SSIM): catch the mode collapse
    * Independent Wassertein critic: training an independent Wasserstein GAN critic to distinguish between held out validation data and generated samples
6. 数据对比
7. experimental insights
    * the network architectures: given enough capacity, DCGAN can be very robust, does not suffer obvious mode collapse
    * number of updates  for each model components: 作者发现更新多次generator然后更新一次discriminator效果更好（而非GAN theory中更新多次discriminator然后一次generator）

![comparationfunction](images/gancomp.png)


### 个人点评

1. 作者吐槽X-GAN中X基本已经被用光了, 之好选择了\alpha GAN
2. 结合VAE和GAN作者认为的好处是
    * VAE的角度，可以解决模糊的问题
    * GAN的角度，可以解决mode collapse的问题
3. 当然数学的近似是很重要的一环，也就是lower bound on the data likelihood with the density ratio trick，这个是见功力的部分


### Resources

1. https://github.com/carpedm20/DCGAN-tensorflow (DCGAN)
