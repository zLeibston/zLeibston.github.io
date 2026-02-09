---
title: "多臂老虎机（MAB）"
math: true
date: 2026-02-09
draft: false
cover:
    # 这是一个自动生成风景图的网站，每次刷新都不一样
    image: "https://picsum.photos/id/1/800/400" 
    alt: "随机封面"
    caption: " "
---
## 多臂老虎机（MAB）
### 1.数学形式化
有 $K$ 个摇臂（Arms），即动作集合为 $\mathcal{A} = \{1, 2, \dots, K\}$。

在每个时刻 $t = 1, 2, \dots, T$：
- 智能体选择一个动作 $a_t \in \mathcal{A}$。
- 环境根据该动作对应的未知概率分布 $\mathcal{R}_{a_t}$ 生成一个奖励 $r_t \sim \mathcal{R}_{a_t}$。
- 智能体的目标是最大化 $T$ 时间步内的累积奖励：$\sum_{t=1}^T r_t$。

在 MAB 研究中，我们通常不直接看累积奖励，而是看**悔失（Regret）**。
设每个臂的期望奖励为 $\mu_a = \mathbb{E}[r | a]$，最优臂的期望奖励为 $\mu^* = \max_{a \in \mathcal{A}} \mu_a$。
$T$ 时间步内的累积悔失定义为：
$$R_T = \mathbb{E} \left[ \sum_{t=1}^T (\mu^* - \mu_{a_t}) \right]$$
我们的目标是使 $R_T$ 随 $T$ 的增长尽可能慢（亚线性增长）。如果 $R_T/T \to 0$，则说明智能体最终找到了最优策略。
### 2.特征
**MAB处理的就是没有状态转移的MDP，这意味着每个动作对应的奖励分布是确定的（也许很难描绘）。矛盾在于，我们愿意花多少成本去探索出期望奖励最大的那个动作**

### 3.$\epsilon$-Greedy算法

该算法是一个启发式算法，具体的，在每个时间步，维护两个变量：
-  $N_t(a)$：到时间 $t$ 为止，动作 $a$ 被选择的次数。
-  $\hat{Q}_t(a)$：到时间 $t$ 为止，动作 $a$ 获得的累积奖励的平均值。
  
算法的决策逻辑是，在每一个时间步 $t$，智能体以概率 $1-\epsilon$ 选择当前看起来最好的动作，即选择 $\arg\max_{a} \hat{Q}_t(a)$。以概率 $\epsilon$ 从所有动作中随机选择一个。前者叫做利用(Exploitation)，后者叫做探索(Exploration)。

### 4.UCB(Upper Confidence Bound)算法
#### 1.Hoeffding 不等式
假设 $X_1, X_2, \dots, X_n$ 是独立同分布的随机变量，取值范围在 $[0, 1]$ 之间。它们的真实均值为 $\mu$，观察到的经验均值为 $\hat{\mu}_n = \frac{1}{n} \sum X_i$。
Hoeffding 不等式告诉我们，经验均值偏离真实均值超过 $\Delta$ 的概率是有界的：
$$P(\mu \geq \hat{\mu}_n + \Delta) \leq e^{-2n\Delta^2}$$
这个公式的意思是：**随着尝试次数 $n$ 的增加，经验均值大于真实均值的概率会以指数级速度衰减。**
#### 2.公式导出
我们的目标是找到一个上限 $U_n$，使得真实均值 $\mu$ 以极高的概率落在 $[0, U_n]$ 之间。令这个“出错”的概率（即真实均值超过我们设定的上限的概率）为一个很小的值 $\delta$，则$e^{-2n\Delta^2} = \delta$，反推得：$\Delta = \sqrt{\frac{\ln(1/\delta)}{2n}}$。所以，对于任何一个臂，它的真实均值 $\mu$ 以 $1-\delta$ 的概率满足：$\mu < \hat{\mu}_n + \sqrt{\frac{\ln(1/\delta)}{2n}}$。若设定 $\delta = t^{-4}$，则代入上式得到：$\Delta_t = \sqrt{\frac{2 \ln t}{n}}$。
因而导出公式：
在每一轮 $t$，算法计算所有臂的 UCB 值，并选择最大者：
$$a_t = \arg\max_{a \in \{1, \dots, K\}} \left[ \underbrace{\hat{\mu}_a}_{\text{利用 (Exploitation)}} + \underbrace{\sqrt{\frac{2 \ln t}{N_t(a)}}}_{\text{探索 (Exploration)}} \right]$$

*   第一项（经验均值）：代表该臂目前表现有多好。
*   第二项（不确定性奖金）：
    *   分母 $N_t(a)$：该臂被试次数越多，这一项越小，代表不确定性降低。
    *   分子 $\ln t$：随着总时间 $t$ 的增加，这一项缓慢增大。这意味着**如果一个臂很久没被尝试了，它的不确定性会随时间增加，直到再次被选中。**
#### 3.悔失上界证明
下证： UCB 算法的累积悔失上界为 $O(\ln T)$。
设最优臂为 $a^*$，其期望奖励为 $\mu^*$。对于任意次优臂 $a$，其奖励间隙（Gap）为 $\Delta_a = \mu^* - \mu_a$。
总悔失可以表示为所有次优臂被拉动次数的加权和：
$$R_T = \mathbb{E}\left[\sum_{t=1}^T (\mu^* - \mu_{a_t})\right] = \sum_{a: \Delta_a > 0} \Delta_a \mathbb{E}[N_T(a)]$$
因此，证明的关键在于：**证明次优臂被拉动的预期次数 $\mathbb{E}[N_T(a)]$ 是 $O(\ln T)$ 级别的。**

在时刻 $t$，次优臂 $a$ 被选中（即 $a_t = a$），必然是因为它的 UCB 值超过了最优臂 $a^*$ 的 UCB 值：
$$\hat{\mu}_a(t) + r_a(t) \geq \hat{\mu}^*(t) + r^*(t)$$
其中 $r_a(t) = \sqrt{\frac{2 \ln t}{N_t(a)}}$ 是置信半径。

为了让上述不等式成立，至少以下三个坏事件之一必须发生：
1.  最优臂被低估：$\hat{\mu}^*(t) \leq \mu^* - r^*(t)$
2.  次优臂被高估：$\hat{\mu}_a(t) \geq \mu_a + r_a(t)$
3.  置信区间仍然太宽：$\mu^* < \mu_a + 2r_a(t)$
   
如果这三个都不发生，则 $UCB^*(t) > \mu^* > \mu_a + 2r_a(t) > UCB_a(t)$，次优臂 $a$ 绝不可能被选中。

- 对事件 3。由于 $\mu^* - \mu_a = \Delta_a$，事件 3 意味着：$\Delta_a < 2 \sqrt{\frac{2 \ln t}{N_t(a)}}$反解 $N_t(a)$，得到一个临界值：$N_t(a) < \frac{8 \ln t}{\Delta_a^2}$。这意味着：**一旦次优臂 $a$ 被拉动的次数超过 $n_a = \frac{8 \ln T}{\Delta_a^2}$，事件 3 就永远不会再发生。**
- 对于**事件 1** 和**事件 2**，根据我们在 UCB 推导中使用的 Hoeffding 不等式：
  $$P(\hat{\mu}^*(t) \leq \mu^* - r^*(t)) \leq e^{-2 N_t(a^*) (\sqrt{2\ln t / N_t(a^*)})^2} = e^{-4\ln t} = t^{-4}$$
同理，$$P(\hat{\mu}_a(t) \geq \mu_a + r_a(t)) \leq t^{-4}$$

则计算次优臂被拉动期望：
$$\mathbb{E}[N_T(a)] \leq \underbrace{n_a}_{前 n_a次拉动,之后事件3永不发生} + \sum_{t=n_a+1}^T P(\text{事件1 或 事件2 发生})$$
$$\mathbb{E}[N_T(a)] \leq \frac{8 \ln T}{\Delta_a^2} + \sum_{t=1}^T (t^{-4} + t^{-4})$$

由于无穷级数 $\sum_{t=1}^\infty 2t^{-4}$ 是收敛的（等于一个很小的常数，约为 $2.16$），所以：
$$\mathbb{E}[N_T(a)] \leq \frac{8 \ln T}{\Delta_a^2} + \text{constant}$$
**证毕！**