---
title: "RL定义与标记规定"
date: 2026-01-15
draft: false
cover:
    # 这是一个自动生成风景图的网站，每次刷新都不一样
    image: "https://picsum.photos/800/400" 
    alt: "随机封面"
    caption: " "
---
## RL定义与标记规定
1. state(s):agent在某一时刻决策所需要的信息的完整描述，是上帝视角，在RL数学框架下，state必须满足马尔可夫性质。作为奖励函数的输入
2. observation(o):agent对当前状态的观察，作为策略网络的输入。通常为state的一部分
3. action(a):agent所采取的改变state的行为
4. 奖励return(r):表示为$r(s,a)$（但也有$r(s)$，仅为s的函数的情况）,为a和s的函数。用于评估在状态s下，只要执行了动作a，平均能得到多少奖励。
5. 价值函数：从当前状态出发，遵循某个策略$\pi$，预期能够获得的累计折扣回报。$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1} $$价值函数就是 $G_t$ 的期望值：$\mathbb{E}[G_t]$。
价值函数的值由四个因素决定，s、a、环境动力因素、奖励函数的设定。
以下是价值函数的分类：
    1. 状态价值函数$(V_{\pi}(s))$:若遵循策略$\pi$，在s状态有多好
    $$ V_\pi(s) = \mathbb{E}_\pi [ G_t | S_t = s ] $$
    2. 动作价值函数$(Q_{\pi}(s,a))$:在状态s下，做动作a有多好
        $$ Q_\pi(s, a) = \mathbb{E}_\pi [ G_t | S_t = s, A_t = a ] $$
6. 一个概率图诠释几个量的关系：![](/imgs/imgs4rl-note/7.png)
7. RL的情景分两类：
    1. 若agent的输入是历史的充分统计量，称为完全观测（fully observation）,此时策略函数可写作$\pi_{\theta}(a_t|s_t)$。此时$o_t$也满足马尔可夫性质。
    2. 反之，agent的输入不是历史的充分统计量，称为部分观测，此时策略函数写作$\pi_{\theta}(a_t|o_t)$。$o_t$不满足马尔可夫性质，必须区分$o_t$和$s_t$。
8. 理论上，RL的框架下，state必循满足马尔可夫性质。但实践中并非完全。我们可以通过特征工程（比如不止输入当前帧，输入前4帧）、模型记忆（如RNN,维护隐藏状态。输入不马尔可夫，但隐藏状态是马尔可夫的）来近似获得马尔可夫量。
9. 强化学习的目标：包含对轨迹的定义：![](/imgs/imgs4rl-note/8.png)
10. 马尔可夫链的状态转移矩阵
    1. $\mathbf{P}_{ij}$是一个方阵，行数n等于state空间的大小。i行j列表示从状态i转移到状态j的概率。满足对任意$i$,有$\sum_{k=1}^n \mathbf{P}_{ik}=1$。
    2. 若$\vec{\mu_t}$代表当前状态的概率分布，其维数为state空间大小n，且$\sum_{k=1}^n\mu_k=1$，则有:$\vec\mu_{t+1}=\vec\mu_t\mathbf{P}$。
11. RL的方法分为model-free和model-base。意思分别是：
    1. model-free：智能体**不尝试去学习**环境是怎么运作的（不学 $P(s'|s,a)$）。它直接根据**过往的经验**，学习哪个动作好，哪个动作坏。
    2. model-base:智能体**先构建一个环境的模型**（可能是已知的，也可能是学出来的），然后在脑子里**推演**未来，基于推演结果来做决策。
    3. 二者的优缺点：
    
    | 维度 | Model-free | Model-based |
    | :--- | :--- | :--- |
    | **真实世界交互** | 需要**海量**交互（数百万步）。 | 需要**少量**交互（几千步可能就行）。 |
    | **算力消耗** | 训练慢，**推断极快**（过一下网路就行）。 | 训练还要练模型，**推断很慢**（因为要规划/搜索）。 |
    | **天花板** | **高**。只要数据够多，它是渐进最优的。 | **受限于模型精度**。如果模型学得不准，策略就永远无法完美。 |
    | **适用场景** | 游戏（Atari）、模拟器环境（数据便宜）。 | 实体机器人（摔不起）、复杂逻辑推理（棋类）。 |
12.RL的方法分为on-policy和off-policy：他们的比较如下：
| 特性 | **On-policy (同策略)** | **Off-policy (异策略)** |
| :--- | :--- | :--- |
| **行为** | 产生数据的策略和作为优化对象的策略必须一致 ($\pi = \mu$) | 产生数据的策略和作为优化对象的策略可以不同 ($\pi \neq \mu$) |
| **数据有效期** | **一次性** (阅后即焚，所以数据复用率低，成本高) | **永久** (放入 Replay Buffer，可进行数据复用) |
| **采样效率** | **低** (Sample Inefficient) | **高** (Sample Efficient) |
| **收敛稳定性** | **高** (稳扎稳打) | **低** (容易发散/震荡) |
| **代表算法** | SARSA, PPO, TRPO, A3C | Q-Learning, DQN, DDPG, SAC |
| **适用场景** | 模拟器环境 (数据便宜)、追求稳定 | 实体机器人 (数据贵)、追求效率 |

13.RL的方法还分为online（在线学习）和offline（离线学习），他们的比较如下：![](/imgs/imgs4rl-note/13.png)
14.优势函数($A^{\pi}(s_t,a_t)$)：衡量在状态$s_t$下，采取$a_t$与平均情况相比的好坏。
$$\begin{aligned}A^{\pi}(s_t,a_t) &= Q^{\pi}(s_t,a_t)-V^{\pi}(s_t)\\
&=\sum_{k=t}^T\mathbb{E}_{\pi_{\theta}}[r(s_k,a_k)|s_t,a_t]-\mathbb{E}_{a_t\sim\pi_{\theta}(a_t|s_t)}[Q^{\pi}(s_t,a_t)]\end{aligned}
$$
15. 贝尔曼方程：$$ Q^\pi(s_t, a_t) = r(s_t, a_t) + \mathbb{E}_{s_{t+1} \sim p(s_{t+1}|s_t, a_t)}[V^\pi(s_{t+1})] $$而引入折扣因子后，变为$Q^\pi(s_t, a_t) = \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)} [r(s_t, a_t) + \gamma V^\pi(s_{t+1})] =r(s_t,a_t)+\gamma\mathbb{E}_{s_{t+1}\sim p(\cdot|s_t,a_t),a_{t+1}\sim \pi(\cdot|s_[t+1])}[Q^{\pi}(s_{t+1},a_{t+1})]$
16. 贝尔曼最优方程：最优策略$\pi^{*}$下Q函数的表现：$Q^{\pi^*}(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_{\mathbf{s}' \sim p(\cdot|\mathbf{s}, \mathbf{a})} \left[ \max_{\bar{\mathbf{a}}'} Q^{\pi^*}(\mathbf{s}', \bar{\mathbf{a}}') \right] \quad \text{for all } (\mathbf{s}, \mathbf{a})$
17. 方法分类：
![](/imgs/28.png)