---
title: "EM算法"
math: true
date: 2026-02-08
draft: false
cover:
    # 这是一个自动生成风景图的网站，每次刷新都不一样
    image: "https://picsum.photos/id/2/800/400" 
    alt: "随机封面"
    caption: " "
---
## EM算法(Expectation-Maximization)

### 1. 问题阐述
假设我们有观测数据集合 $X = \{x_1, x_2, \dots, x_N\}$，隐含变量集合 $Z = \{z_1, z_2, \dots, z_N\}$（可以理解为不可观测的中间变量,会随样本而变化，是每个样本特有的、不可直接观测的局部属性/状态），要估计参数 $\theta$。我们用对数极大似然估计：$$L(\theta) = \sum_{i=1}^N \log p(x_i | \theta) = \sum_{i=1}^N \log \sum_{z_i} p(x_i, z_i | \theta)$$
### 2.推导
为了简化，我们考虑一个样本的情况，即最大化$\log p(x|\theta)$。参考VI中的推导，我们得到：$$\log p(x|\theta) = \underbrace{\sum_z q(z) \log \frac{p(x, z|\theta)}{q(z)}}_{\text{ELBO}(q, \theta)} + \underbrace{KL(q(z) || p(z|x, \theta))}_{\text{KL 散度}}$$其中$q(z)$是隐变量的分布函数，随样本变化。 $KL>0$,而利用琴生不等式（对于凹函数 $\log(x)$，有 $E[\log X] \le \log E[X]$，当且仅当随机变量$X$为常数取等）放缩得到：$$\log p(x|\theta)=\log \sum_z q(z) \frac{p(x, z | \theta)}{q(z)} \ge \sum_z q(z) \log \frac{p(x, z | \theta)}{q(z)}$$两种方法都可得到ELBO是$\log p(x|\theta)$的下界。只需最大化$ELBO=\mathbb{E}_{z\sim q(\cdot)}\frac{p(x,z|\theta)}{q(z)}$，

### 3.算法
EM算法是一个迭代算法。首先，随机初始化带估计参数$\theta^{(0)}$。我们的最终目标是就是最大化似然函数$L(\theta)$，但是你会发现，如果你想估算参数$\theta$,你必须先知道每个点属于哪个簇（即隐变量$Z$）。如果你想估算隐变量 $Z$（即点属于哪个簇）：你必须先知道每个簇的参数$\theta$。
#### 1.E步(先假设参数$\theta^{(t)}$是对的，根据它去估算隐变量$Z$的期望)
调整$q(z)$来最大化ELBO，由琴生不等式的取等条件，得此时：$q_i(z)\propto p(x_i,z|\theta)$,又因为对样本$x_i$有$\sum_{z}q_i(z)=1$，我们可以得到$$q_i(z_i=k)=p(z_i=k|x_i,\theta^{(t)})=\frac{p(x_i | z_i = k, \theta^{(t)}) \cdot p(z_i = k | \theta^{(t)})}{\sum_{j=1}^K p(x_i | z_i = j, \theta^{(t)}) \cdot p(z_i = j | \theta^{(t)})}$$其中$K$为隐变量空间大小，$N$为样本数。如此我们实际上得到了一个$N\times K$的矩阵，可以刻画$\theta^{(t)}$情况下每个样本的隐变量分布。
#### 2.M步（假设隐变量分布是对的，更新参数$\theta$）
我们引入一个$Q$函数：$$\begin{aligned}Q(\theta, \theta^{(t)}) &= E_{Z \sim p(\cdot|X, \theta^{(t)})} [\log p(X, Z | \theta)]\\
&= \sum_{i=1}^N \left[ \sum_{k=1}^K p(z_i=k | x_i, \theta^{(t)}) \log p(x_i, z_i=k | \theta) \right]\\
&= \sum_{i=1}^N \left[ \sum_{k=1}^K q_i(z_i=k) \log p(x_i, z_i=k | \theta) \right]
\end{aligned}$$其中$q_i(z_i=k)$在E步中已求出，将 $Q(\theta, \theta^{(t)})$ 视为一个仅关于变量 $\theta$ 的函数（因为 $\theta^{(t)}$ 已经是一个固定的数值），然后找到能使这个函数最大化的 $\theta$ 值。
$$\theta^{(t+1)} = \arg \max_\theta Q(\theta, \theta^{(t)})$$
关于这一步，也有讨论：

- 若$\nabla_\theta Q(\theta, \theta^{(t)})=0$有闭式解，直接带入即可
- 若如果没有闭式解，但 $Q(\theta, \theta^{(t)})$ 是凸函数，可选用梯度上升、牛顿法、拟牛顿法、坐标上升法等。
- 若$Q(\theta, \theta^{(t)})$不是凸函数，可用梯度上升找局部最优，只要保证$Q(\theta^{(t+1)},\theta^{(t)})\ge Q(\theta^{(t)},\theta^{(t)})$即可。
最后，E步M步交替进行即可。
### 4.说明之一：为什么M步要优化Q函数而不是似然函数，即为什么优化Q函数等价于优化似然函数
似然函数:$$\begin{aligned}L(\theta) &=\sum_{i=1}^N \log \sum_{z_i} p(x_i, z_i | \theta)\\
&=\sum_{i=1}^N\log\sum_{z_i} q_i(z_i)\frac{p(x_i, z_i | \theta)}{q_i(z_i)}\\
&\ge \sum_{i=1}^N\sum_{z_i} q_i(z_i)\log \frac{p(x_i, z_i | \theta)}{q_i(z_i)}\text{（琴生不等式）}\\
&=\underbrace{ \sum_{i=1}^N \sum_{z_i} p(z_i|x_i, \theta^{(t)}) \log p(x_i, z_i|\theta)}_{Q(\theta, \theta^{(t)})}\underbrace{-\sum_{i=1}^N \sum_{z_i} p(z_i|x_i, \theta^{(t)}) \log p(z_i|x_i, \theta^{(t)})}_{H(\theta^{(t)}) }\\
&=Q(\theta, \theta^{(t)})+H(\theta^{(t)}) 
\end{aligned}$$
显然$H(\theta^{(t)}) $与优化目标$\theta$无关。若有$Q(\theta^{(t+1)},\theta^{(t)})\ge Q(\theta^{(t)},\theta^{(t)})$，则$$\begin{aligned}
L(\theta^{(t+1)})&=Q(\theta^{(t+1)}, \theta^{(t)})+H(\theta^{(t)})\\
&\ge Q(\theta^{(t)}, \theta^{(t)})+H(\theta^{(t)})\\
&=L(\theta^{(t)})
\end{aligned}$$
保证似然函数增大。
### 5.说明之二：为什么不直接对似然函数做梯度上升来优化，反正似然函数也可导
似然函数反向传播会遍历隐变量集合空间，复杂度太大。若隐变量是连续的，还需要数值积分。时间复杂度大，数值不稳定。
### 6.说明之三：EM算法和VI的对比
VI:
![](/imgs/img4ML/vi.png)
EM：
![](/imgs/img4ML/em.png)

### 6.说明之四：若隐变量空间连续，如何处理
采用蒙特卡洛近似。从后验分布采样$S$个样本：
 $$z_i^{(1)}, z_i^{(2)}, ..., z_i^{(S)} \sim p(z_i|x_i, \theta^{(t)})$$然后用样本均值近似期望：

 $$Q(\theta, \theta^{(t)}) \approx \hat{Q}(\theta, \theta^{(t)}) = \sum_{i=1}^N \frac{1}{S} \sum_{s=1}^S \log p(x_i, z_i^{(s)}|\theta)$$
 随后优化近似的$Q$函数即可：
$$\theta^{(t+1)} = \arg \max_\theta \hat{Q}(\theta, \theta^{(t)})$$
