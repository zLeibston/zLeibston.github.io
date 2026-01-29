---
title: "离线强化学习（offLine-RL）"
math:true
date: 2026-01-13
draft: false
cover:
    # 这是一个自动生成风景图的网站，每次刷新都不一样
    image: "https://picsum.photos/800/400" 
    alt: "随机封面"
    caption: " "
---
## offLineRL(离线强化学习)
1. 定义：与在线（online）RL在训练时同步收集数据不同，离线强化学习旨在通过给定数据集惊醒训练，中途不再自己搜集数据。为了以后叙述饭方便，我们规定$\pi_{\beta}$是搜集数据的策略（也有可能是多个策略的平均）.
2. 直接采用off-policy方法（DQN、SAC等）来解决offline问题，会产生一系列问题。offline的给定数据集不可能涵盖所有动作，必然会有大量未涵盖的OOD（分布外）数据。初始的神经网络$\hat{Q}_{\phi}$对于没有数据监督的 OOD 动作，其输出值接近于随机初始化，充满了随机的、不可靠的噪声。Q-learning中max操作会系统性的抓住较大的噪声不放，而SAC中若Critic 网络$Q_{\phi}$ 对一个 OOD 动作 A 错误地给出了高分。则Actor 网络$\pi_{\theta}$在进行梯度上升时，会把参数朝着能增加动作 A 概率的方向更新。这都会造成系统性的**过估计**。
3. offline的一个基线方法：用简单的模仿学习，但在给定的数据集中剔除一定比例的最终价值太小的轨迹（如只保留最终价值在前30%的轨迹），在剩余数据上做模仿学习（行为克隆）,而模仿学习不会有系统性的过估计。如图：![](/imgs/imgs4rl-note/22.png)
4. 一个稍好的方法：Advantage-weighted regression(AWR)：简单来说就是拟合价值函数$\hat{V}^{\pi_{\beta}}_{\phi}$，进而估计出优势函数$\hat{A}^{\pi_{\beta}}_{\phi}$，以此作为自监督学习更新的权重，详情见下图：![](/imgs/imgs4rl-note/23.png)这个方法的好处是比较简单而实用。但缺点是估计的价值函数是基于搜集数据策略$\pi_{\beta}$的。所以$\hat{A}^{\pi_{\beta}}_{\phi}$回比$\hat{A}^{\pi_{\theta}}_{\phi}$弱，特别是$\pi_{\beta}$本身就很菜的时候。且蒙特卡洛估计本就方差很大。
5. IQL（隐式Q学习）：
   1. 分位损失：在很多情况下，我们采样数据所用的策略都是很平庸的。在优势函数$A^{\pi_\beta}$中，用$V^{\pi_\beta}(s) = \mathbb{E}_{a \sim \pi_\beta(\cdot|s)}[Q(s, a)]$做基线要求太低了，我们适当拔高这个基线。具体的，我们不采用L2损失这种对称的误差方式，而用一种非对称的方式来加大对低于基线价值的估计惩罚，减少对高于基线价值的估计的惩罚。具体损失函数如下:$$ L_V(\psi) = \mathbb{E}_{(s,a) \sim \mathcal{D}} [ L_2^\tau (\hat{Q}_{\hat{\phi}}(s, a) - \hat{V}_\psi(s)) ] $$其中，$L_2^\tau(u)$ 是非对称损失：$$ L_2^\tau(u) = \begin{cases} \tau u^2 & \text{if } u \ge 0 \\ (1-\tau) u^2 & \text{if } u < 0 \end{cases} $$  $\tau$是一个超参数，$\tau$大于0.5时，算法会更严厉地惩罚低估$\hat{V}_{\psi}$。为了最小化损失，$\hat{V}_{\psi}$不得不“抬高”自己的值，去逼近 Q 值的上-$\tau$分位数，也就是我们想要的“乐观”价值。这个trick叫**分位损失**。
   2. IQL步骤：![](/imgs/imgs4rl-note/24.png)几点注意：
        1. 你没看错，这里面要维护3个神经网络。
        2. 上面我自己写的损失函数是$Q-V$，所以是要设置$\tau>0.5$，而图中损失函数是$V-Q$，所以要设置$\tau<0.5$才能达到相同的效果。
6. CQL(保守Q学习):还有一个方法,为了缓解offline中Q函数过高估计OOD动作的价值，我们可以在训练Q网络的损失函数上做文章。具体的，我们可以如下设置损失函数：$$\hat{Q}^\pi_{\phi} = \arg\min_{\hat{Q}^\pi_{\phi}} \max_\mu \mathbb{E}_{(s,a,s')\sim\mathcal{D}} \left[ \left( \hat{Q}^\pi_{\phi}(s,a) - (r(s,a) + \gamma \mathbb{E}_{a'\sim\pi}[\hat{Q}^\pi_{\phi}(s',a')]) \right)^2 \right] + \alpha \mathbb{E}_{s\sim\mathcal{D}, a\sim\mu(\cdot|s)}[\hat{Q}^\pi_{\phi}(s,a)] - \alpha \mathbb{E}_{(s,a)\sim\mathcal{D}}[\hat{Q}^\pi_{\phi}(s,a)]$$其中第一项称为贝尔曼误差，是要最小化的，使其满足贝尔曼方程。第二项是抑制所有Q值较大的动作的价值，这样固然会减小OOD动作的价值，但会误伤分布内的动作的价值。所以第三项作为补偿，统一增大分布内的动作的价值。$\alpha$是一个需要调节的超参数。
   1. 策略$\mu$的一个选择是：不用使用神经网络计算，只需用$$\mu^*(a|s) = \begin{cases} 1 & \text{if } a = \arg\max_{a'} Q(s, a') \\ 0 & \text{otherwise} \end{cases} $$即可.
   2. 上面的$\mu$的选择太贪婪，会使这种抑制的效果不够广泛。我们要让策略$\mu$“随机”一点，不能过于集中。具体的，我们要用熵加一个正则项，则损失函数变为：$$\hat{Q}^\pi_{\phi} = \arg\min_{\hat{Q}^\pi_{\phi}} \max_\mu \mathbb{E}_{(s,a,s')\sim\mathcal{D}} \left[ \left( \hat{Q}^\pi_{\phi}(s,a) - (r(s,a) + \gamma \mathbb{E}_{a'\sim\pi}[\hat{Q}^\pi_{\phi}(s',a')]) \right)^2 \right] +\\ \alpha \mathbb{E}_{s\sim\mathcal{D}, a\sim\mu(\cdot|s)}[\hat{Q}^\pi_{\phi}(s,a)] - \alpha \mathbb{E}_{(s,a)\sim\mathcal{D}}[\hat{Q}^\pi_{\phi}(s,a)]+\mathbb{E}_{s\sim\mathcal{D}}[\mathcal{H}(\mu(\cdot|s))]$$保证策略不能太贪婪。则与策略$\mu$有关的项为:$\max_\mu  \mathbb{E}_{s\sim\mathcal{D}, a\sim\mu(\cdot|s)}[Q(s,a)] + \mathbb{E}_{s\sim\mathcal{D}}[\mathcal{H}(\mu(\cdot|s))]= \max_\mu  \mathbb{E}_{s\sim\mathcal{D}, a\sim\mu(\cdot|s)}[Q(s,a)]-\mathbb{E}_{s\sim\mathcal{D}}[ \sum_a \mu(a|s) \log \mu(a|s)]$,同时还需满足约束条件$\sum_{a}\mu(a|s)=1$。这就是一个优化问题。可以证明，使上式最优的策略$\mu^*(a|s)=\text{softmax}(Q)=\frac{\exp(Q(s,a))}{\sum_{a'}\exp(Q(s,a'))}$，因而代入得$$\begin{aligned}\max_\mu  [\mathbb{E}_{ a\sim\mu(\cdot|s)}[Q(s,a)] + \mathcal{H}(\mu(\cdot|s))]&=\sum_a\mu^*(a|s)Q(s,a)-\sum_a\mu^*(a|s)\log\mu^*(a|s)\\&=\sum_a\left[\frac{\exp(Q(s,a))}{\sum_{a'}\exp(Q(s,a'))}\left(Q(s,a)-Q(s,a)+\log\sum_{a'}\exp(Q(s,a'))\right)\right]\\&= \log\sum_{a}\exp(Q(s,a))\end{aligned}$$因而总的损失函数为：$$\mathcal{L}_{CQL}(\phi)= \mathbb{E}_{(s,a,s')\sim\mathcal{D}} \left[ \left( \hat{Q}^\pi_{\phi}(s,a) - (r(s,a) + \gamma \mathbb{E}_{a'\sim\pi}[\hat{Q}^\pi_{\phi}(s',a')]) \right)^2 \right] + \alpha\mathbb{E}_{s\sim\mathcal{D}}\log\sum_a\exp(\hat{Q}_{\phi}^{\pi}) - \alpha \mathbb{E}_{(s,a)\sim\mathcal{D}}[\hat{Q}^\pi_{\phi}(s,a)]$$
   3. 最终的算法步骤：![](/imgs/imgs4rl-note/25.png)