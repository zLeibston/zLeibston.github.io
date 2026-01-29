---
title: "Q-learning"
math:true
date: 2026-01-15
draft: false
cover:
    # 这是一个自动生成风景图的网站，每次刷新都不一样
    image: "https://picsum.photos/id/4/800/400" 
    alt: "随机封面"
    caption: " "
---

## Q-learning(通常是off-policy的)
1. **核心思想**
用神经网络估计是估计在状态$s$下做动作$a$，然后**遵循最优策略**所能取得的期望总价值。
2. 由贝尔曼最优方程:$$\begin{aligned}Q^{\pi^*}(\mathbf{s}, \mathbf{a}) &= r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_{\mathbf{s}' \sim p(\cdot|\mathbf{s}, \mathbf{a})} \left[ \max_{\bar{\mathbf{a}}'} Q^{\pi^*}(\mathbf{s}', \bar{\mathbf{a}}') \right] \quad \text{for all } (\mathbf{s}, \mathbf{a})\\
&\approx r(s,a)+\gamma \max_{a'}\hat{Q}_{\phi}(s'_{i},a')
\end{aligned}$$故，采样出$(s,a,s',r)$后，只需用$\{(s,r(s,a)+\gamma \max_{a'}\hat{Q}_{\phi}(s'_{i},a'))\}$做神经网络$\hat{Q}_{\phi}$的训练样本。然后，有了函数$\hat{Q}_{\phi}$，我们的策略就是贪心得求出给定状态下的最优策略。
3. DQN基本算法流程：![](/imgs/imgs4rl-note/19.png)
需要指出的是，由于冷启动的局限，刚开始$Q$函数是乱算的，如果直接完全贪心取$Q$最大的策略，容易出不来。我们需要在采样的时候迫使策略做大量尝试，有两种方法：
    1. 以一定概率随机采样。具体的说，我们采样的策略是：$$\pi(\mathbf{a}_t|\mathbf{s}_t) = 
    \begin{cases}
        1 - \epsilon & \text{if } \mathbf{a}_t = \arg\max_{\mathbf{a}_t} Q_\phi(\mathbf{s}_t, \mathbf{a}_t) \\
        \epsilon / (|\mathcal{A}| - 1) & \text{otherwise}
    \end{cases}$$这里$\epsilon$是一个超参数。
    2. 从$Q$输出的概率分布中采样，而不是简单的取$Q$值最大的策略,即$\pi(\mathbf{a}_t|\mathbf{s}_t) \propto \exp(Q_\phi(\mathbf{s}_t, \mathbf{a}_t))$。
由此得到最终的Q-learning流程：![](/imgs/imgs4rl-note/20.png)
4. DQN的改进：以上的基础Q-learning算法中，用自举法造训练数据会是训练目标不断变化，会导致训练不稳定。我们引入一个$\hat{Q}_{\phi}$网络的副本$\hat{Q}_{\phi'}$来做采样，即训练目标变为$\{(s,r(s+a)+\gamma\max_{a'}\hat{Q}_{\phi'}(s',a'))\}$，这个副本隔固定时间步长变化，使训练目标的变化不那么剧烈。其他流程不变，则算法如图：![](/imgs/imgs4rl-note/21.png)
5. double Q-learning:有我们的目标值公式$ y_j = r_j + \gamma \max_{a'} Q_\phi(s'_j, a'_j) $可以看出：首先，我们的 Q 网络是不完美的，而我们选择$a'$和计算$Q(s', a')$都是用的同一个有噪声的神经网络 $Q_\phi$。结果就是双重高估，导致**过估计（overestimate）**。这种正向的误差会通过贝尔曼更新不断地向后传播，导致整个 Q 网络的估值系统性地偏高。为了解决，我们要将“动作选择器”和“价值评估器"分开。由于在4点的改进版DQN中，我们已经有$y=r(s,a)+\gamma\max_{a'}\hat{Q}_{\phi'}(s',a')=r(s,a)+\gamma\max_{a'}\hat{Q}_{\phi'}(s',\argmax_{a'}\hat{Q}_{\phi'}(s',a'))$,将其改进为$y=r(s+a)+\gamma\max_{a'}\hat{Q}_{\phi'}(s',\argmax_{a'}\hat{Q}_{\phi}(s',a'))$,用在线网络$\hat{Q}_{\phi}$来选动作，用$\hat{Q}_{\phi'}$来估计价值。
6. 在刚开始迭代时，我们的$\hat{Q}_{\phi’}$一般较差，这个时候我们的目标中$ y_{j,t} = r_{j,t} + \gamma \max_{a_{j,t+1}} Q_{\phi'}(s_{j,t+1}, a_{j,t+1}) $只有第一项$r$是有效的。我们要增加有效的目标的占比，就可以：$y_{j,t} = \left( \sum_{t'=t}^{t+N-1} \gamma^{t'-t} r_{j,t'} \right) + \gamma^N \max_{a_{j,t+N}} Q_{\phi'}(s_{j,t+N}, a_{j,t+N})$。但是又会产生2个问题：
    1. 我们的神经网络$Q_{\phi}$估计的是最优价值，但是replay buffer里面选出来组合起来的不一定是最优策略的选择结果。这会产生一定的分布迁移。我们的解决方案很简单：忽略它。
    2. 所有的数据我们都是从replay buffer里面采样的，很难收集到正规的一条链上的数据（比如$s_{t+1}$大概率不会是$s_t$执行$a_t$后的状态）。对此，我们只需改变存入replay buffer中的数据内容，从$ (s, a, r, s')$变成$(s_t, a_t, y_t, s_{t+N})$。或者，更常见的是，在训练时动态计算。