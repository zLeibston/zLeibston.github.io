---
title: "变分推断"
math: true
date: 2026-01-13
draft: false
cover:
    # 这是一个自动生成风景图的网站，每次刷新都不一样
    image: "https://picsum.photos/id/2/800/400" 
    alt: "随机封面"
    caption: " "
---
## 变分推断（Variational Inference,VI）
#### 1.Introduction
变分推断是机器学习和贝叶斯统计中一种极其重要的近似推断方法。它的核心思想是：**将一个复杂的“概率推断问题”转化为一个“优化问题”来求解**。在贝叶斯推断中，我们通常已知观测数据 $X$，想要计算隐藏变量（或参数）$Z$ 的**后验概率** $P(Z|X)$。根据贝叶斯公式：
$$P(Z|X) = \frac{P(X, Z)}{P(X)} = \frac{P(X|Z)P(Z)}{\int P(X, Z) dZ}$$

**问题在于：** 那个分母 $P(X)$（证据，Evidence）通常涉及对高维空间进行积分。对于大多数复杂的概率模型，这个积分是没有解析解的（即不可积），这导致我们无法直接得到准确的后验分布。

**变分推断的解决方案：**
我们不再死磕那个不可积的真实后验分布 $P(Z|X)$，而是找一个简单的分布族$\mathcal{Q}$（例如高斯分布族）。 我们在这个简单的分布族里找出一个分布 $q(Z)$，让它**尽可能地接近**那个复杂的真实分布 $P(Z|X)$。 “寻找最接近分布”的过程，就变成了一个优化问题。

#### 2. VI的一般范式
首先，我们选择变分分布族，即定义 $q(Z)$ 的形式。为了简化计算，最常用的假设是**平均场变分族（Mean Field Variational Family）**。它假设所有隐藏变量之间是相互独立的（就是说若假定的近似分布是正态分布，则均值与方差独立）。
接着，我们使用KL散度来衡量两个分布 $q(Z)$ 和 $P(Z|X)$ 之间的差异。我们的目标是最小化这个差异：
$$ \begin{aligned} KL(q(Z) || P(Z|X)) &= \int q(Z) \log \frac{q(Z)}{P(Z|X)} dZ\\ 
&=  E_{q(Z)}[\log q(Z) - \log P(Z|X)]\\
&=  E_{q(Z)}[\log q(Z) - \log\frac{P(X,Z)}{P(X)}]\\
&=    E_{q(Z)} [\log q(Z) - \log P(X, Z) + \log P(X)]\\
&= E_{q(Z)} [\log q(Z) - \log P(X, Z)] + \log P(X)
\end{aligned}$$
将上式重新排列得到：$$\log P(X) = KL(q(Z) || P(Z|X)) + \underbrace{E_{q(Z)} [\log P(X, Z) - \log q(Z)]}_{\text{这就是 ELBO}}$$
由于$\log P(X)$是常数，为了最小化$KL(q(Z) || P(Z|X))$，只需最大化ELBO。观察ELBO:$$\begin{aligned}\text{ELBO}&=E_{q(Z)} [\log P(X, Z) - \log q(Z)]\\&= E_{q(Z)} [\log P(X|Z) + \log P(Z) - \log q(Z)]\\
&= E_{q(Z)} [\log P(X|Z)] - E_{q(Z)} [\log q(Z) - \log P(Z)]\\
&=  E_{q(Z)} [\log P(X|Z)] - KL(q(Z) || P(Z))
\end{aligned}$$
在针对观测数据 $X$ 进行推断的实际场景中（如 VAE），变分分布 $q(Z)$ 是由 $X$ 决定的（即 $q$ 是 $X$ 的函数）。为了体现这种依赖关系，我们将符号 $q(Z)$ 改写为 $q(Z|X)$。代入符号修改后，得到最终目标形式：
$$\text{ELBO} = \underbrace{E_{q(Z|X)}[\log P(X|Z)]}_{\text{重构准确度}} - \underbrace{KL(q(Z|X) || P(Z))}_{\text{正则化项}}$$
#### 3.VAE
在以上基础上，介绍变分自编码器(VAE)。输入数据$x$，潜变量（隐藏变量）$z$。用$q_{\phi}(z|x)$表示encoder，输入x，输出z的分布的后验概率分布的参数（通常用正态分布，则输出为均值与方差）。用$p_{\theta}(x|z)$表示decoder，输入z后反推x的分布的参数。而$p_{\theta}(z|x)$表示decoder输出x后反推$z$的分布，定义为$p_{\theta}(z|x)=\frac{p_{\theta}(x|z)p(z)}{p_{\theta}(x)}$,而分母$p_{\theta}(x)$难以计算，需用变分推断。VAE是生成模型，我们的目标是从前空间中采样得到$z$，然后用decoder重建x。同样的有：
$$\begin{aligned}\log p_\theta(x)&=KL(q_{\phi}(z|x)||p_\theta(z|x))+\mathbb{E}_{z\sim q_{\phi}(\cdot|x)}[\log p_{\theta}(x,z)-\log q_\phi(z|x)]\\
&=KL+\underbrace{\mathbb{E}_{z\sim q_\phi(\cdot|x)}\log p_\theta(x|z)-KL(q_\phi(z|x)||p(z))}_{ELBO}\end{aligned}
$$
![](/imgs/img4ML/vae.png)如图，VAE是端到端训练。为了实现端到端的反向传播，VAE 引入了重参数化技巧，将采样的随机性转移到外部噪声 $\epsilon$上，从而使得梯度能够穿过随机采样层,这里不再详细说。损失函数包含两个部分：$$\mathcal{L}(\phi,\theta)=\underbrace{\frac{1}{\text{batchsize}}||x-\hat x||^2}_{\text{重建损失}}+\underbrace{KL(q_{\phi}(z|x)||p(z))}_{\text{正则项}}$$当假设 Decoder 的输出服从高斯分布时，最大化对数似然 $\mathbb{E}[\log p(x|z)]$ 就等价于最小化均方误差（MSE）。生成时直接从先验的潜空间$p(z)$采样，输入decoder即可。