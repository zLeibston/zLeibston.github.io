---
title: "策略梯度（policy gradient）"
math:true
date: 2026-01-15
draft: false
cover:
    # 这是一个自动生成风景图的网站，每次刷新都不一样
    image: "https://picsum.photos/800/400" 
    alt: "随机封面"
    caption: " "
---

## 策略梯度（通常是model-free）
1. 定义：策略梯度是训练一个神经网络，输入状态s（或观测 o），输出动作的概率分布（或参数），通过奖励信号的反向传播，使得能获得高回报的动作出现的概率越来越大。
2. 最朴素：REINFORCE
   1. 方法简述：基于当前策略走N个轨迹并算出平均得分，随后利用已知信息更新神经网络$\pi_{\theta}(a_t|s_t)$的参数。
   2. 推导：
   我们的目标是：$$\theta^*=arg \max \mathbb{E}_{\tau\sim p_{\theta(\tau)}}\left[\sum _t r(s_t,a_t)\right]=arg \max \mathbb{E}_{\tau\sim p_{\theta(\tau)}}\left[ r(\tau)\right]$$其中$\tau$代表轨迹$\{(s_1,a_1)、(s_2,a_2).......\}$
   令$J(\theta)=\mathbb{E}_{\tau\sim p_{\theta(\tau)}}\left[ r(\tau)\right]=\int p_{\theta}(\tau)r(\tau)d\tau$,则有：$$\nabla_{\theta} J(\theta)=\int\nabla_{\theta} p_{\theta}(\tau)r(\tau)d\tau$$因为$$p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)=p_{\theta}(\tau)\frac{\nabla_{\theta} p_{\theta}(\tau)}{p_{\theta}(\tau)}=\nabla_{\theta} p_{\theta}(\tau)$$所以有：$$\begin{aligned}\nabla_{\theta} J(\theta)=&\int p_{\theta}(\tau)\nabla_{\theta}\log p_{\theta}(\tau)r(\tau)d\tau\\=&\mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\nabla_{\theta}\log p_{\theta}(\tau)r(\tau)\right]\end{aligned}$$
   因为有：$$p_{\theta}(\tau)=p_{\theta}(s_1,a_1......s_T,a_T)=p(s_1)\prod_{t=1}^T\left(\pi_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t)\right)$$而两边取对数并求导：
   $$\nabla_{\theta} \log p_{\theta}(\tau)=\sum_{t=1}^T\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)$$
   所以：$$\begin{aligned}\nabla_{\theta} J(\theta)=&\mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left(\left(\sum_{t=1}^T\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)\right)\left(\sum_{t=1}^T r(s_t,a_t)\right)\right)\\\approx&\frac{1}{N}\sum_{i=1}^N\left(\sum_{t=1}^T\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)\right)\left(\sum_{t=1}^T r(s_t,a_t)\right)\end{aligned}$$
   更新参数：$\theta^{'}=\theta+\alpha \nabla_{\theta} J(\theta)$
   3. algorithm:![](/imgs/imgs4rl-note/9.png)
   4. 诠释：
      1. 当action空间离散时，神经网络$\pi_{\theta}$的输出是所有a的概率分布。
      2. 当action空间连续，看成高斯分布，神经网络输出高斯分布的均值，如图![](/imgs/imgs4rl-note/10.png)
   5. 我们以上的推导是在fully observed的情况下发生的，即神经网络的输入是s。但由于我们的推导未利用s的马尔可夫性，所以partly observed的情况下的计算方法如下，简单替换即可：$$\nabla_{\theta} J(\theta)=\frac{1}{N}\sum_{i=1}^N\left(\sum_{t=1}^T\nabla_{\theta}\log \pi_{\theta}(a_t|o_t)\right)\left(\sum_{t=1}^T r(s_t,a_t)\right)$$
   6. 朴素策略梯度（REINFORCE）的缺点之一：能在有限成本内尝试的轨迹数太少，轨迹数太少会导致梯度方向抖动剧烈，难以收敛。为了克服，我们对原来的算$\nabla_{\theta} J(\theta)$的式子稍作修改。首先，原来的式子分配律展开：$$\nabla_{\theta} J(\theta)=\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_{\theta}\log\pi_{\theta}(a_{i,t}|s_{i,t})\left(\sum_{t^{'}=1}^T r(s_{i,t^{'}},a_{i,t^{'}})\right)$$
   做个小变化：
   $$\nabla_{\theta} J^{'}(\theta)=\frac{1}{N}\sum_{i=1}^N\sum_{t=1}^T\nabla_{\theta}\log\pi_{\theta}(a_{i,t}|s_{i,t})\left(\sum_{t^{'}=t}^T r(s_{i,t^{'}},a_{i,t^{'}})\right)$$
   这样一个决策之后的奖励累计才算做他的影响，合理多了，也减少了震荡。这个策略梯度的常见优化方法，叫做**Reward-to-Go**
   7. 朴素策略梯度（REINFORCE）的缺点之二：理想的整个轨迹的总奖励应该有正负之分，即好的轨迹总奖励为正，坏的为负。但事实可能都为正或都为负。解决方法:用若干个轨迹的总奖励的平均值做基线(**Baseline**)：
   $$
    \begin{aligned}
    \nabla_\theta J(\theta) &\approx \frac{1}{N} \sum_{i=1}^{N} \nabla_\theta \log p_\theta(\tau)[r(\tau) - b] \\
    b &= \frac{1}{N} \sum_{i=1}^{N} r(\tau)
    \end{aligned}
    $$
    实际上最优基线可以有理论推出，它可以使不同轨迹造成的梯度的方差最小，从而缓解震荡（但实际中用平均值做基线也不错），如图：![](/imgs/imgs4rl-note/11.png)
3. off-policy的polidy gredients
   1. 一个结论，重要性采样，实现off-policy的核心数学工具：$$\begin{aligned}\mathbb{E}_{x\sim p(x)}f(x)=&\int f(x)p(x)dx\\=&\int\frac{q(x)}{q(x)}f(x)p(x)dx\\=&\mathbb{E}_{x\sim q(x)}\left(\frac{p(x)}{q(x)}f(x)\right)\end{aligned}$$
   2. 推导：令$\theta$为用于采样的策略的参数，$\theta^{'}$为我们优化的对象。则由原来的$J(\theta)=\mathbb{E}_{\tau\sim p_{\theta(\tau)}}\left[ r(\tau)\right]$得$J(\theta^{'})=\mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[ \frac{p_{\theta^{'}}(\tau)}{p_{\theta}(\tau)} r(\tau) \right]$,则：$$\begin{aligned}
   \nabla_{\theta'} J(\theta') &= E_{\tau \sim p_\theta(\tau)} \left[ \frac{\nabla_{\theta'} p_{\theta'}(\tau)}{p_\theta(\tau)} r(\tau) \right] \\
   &= E_{\tau \sim p_\theta(\tau)} \left[ \frac{p_{\theta'}(\tau)}{p_\theta(\tau)} \nabla_{\theta'} \log p_{\theta'}(\tau) r(\tau) \right]
   \end{aligned}$$易得：$\frac{p_{\theta'}(\tau)}{p_\theta(\tau)} = \frac{\prod_{t=1}^T \pi_{\theta'}(\mathbf{a}_t|\mathbf{s}_t)}{\prod_{t=1}^T \pi_\theta(\mathbf{a}_t|\mathbf{s}_t)}$,所以：$$
   \begin{aligned}
   \nabla_{\theta'} J(\theta') &= E_{\tau \sim p_\theta(\tau)} \left[ \frac{p_{\theta'}(\tau)}{p_\theta(\tau)} \nabla_{\theta'} \log \pi_{\theta'}(\tau) r(\tau) \right] \quad \text{when } \theta \neq \theta' \\
   \\
   &= E_{\tau \sim p_\theta(\tau)} \left[ \left( \prod_{t=1}^T \frac{\pi_{\theta'}(\mathbf{a}_t|\mathbf{s}_t)}{\pi_\theta(\mathbf{a}_t|\mathbf{s}_t)} \right) \left( \sum_{t=1}^T \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_t|\mathbf{s}_t) \right) \left( \sum_{t=1}^T r(\mathbf{s}_t, \mathbf{a}_t) \right) \right] 
   \end{aligned}
   $$
   考虑 $t$ 时刻的梯度项：我们只看 $\nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t)$ 的贡献。
    *   **因果性 I (Reward-to-be)**：$\nabla_t$ 不应该乘以它之前的奖励 ($r_1, \dots, r_{t-1}$)，因为 $a_t$ 无法影响过去。所以，与 $\nabla_t$ 相乘的奖励项变成了 $\sum_{t'=t}^T r_{t'}$ (Reward-to-Go)。
    *   **因果性 II (IS Weight)**：$\nabla_t$ 只在 $(s_t, a_t)$ 处有值。要计算这一项在 Off-policy 下的期望，我们需要修正到达 $(s_t, a_t)$ 的概率。而这个概率只取决于过去的动作（从 1 到 $t$）。未来的动作（从 $t+1$ 到 $T$）与“我们如何到达这里”无关。

    所以，与 $\nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t)$ 相乘的重要性权重，实际上只需要考虑到时刻 $t$ 为止：
    $$ \rho(h_t) = \prod_{t'=1}^t \frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_{\theta}(a_{t'}|s_{t'})} $$
   因而得到：
   $$\nabla_{\theta'} J(\theta') =E_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=1}^T \nabla_{\theta'} \log \pi_{\theta'}(\mathbf{a}_t|\mathbf{s}_t)  \left( \prod_{t'=1}^{t} \frac{\pi_{\theta'}(\mathbf{a}_{t'}|\mathbf{s}_{t'})}{\pi_\theta(\mathbf{a}_{t'}|\mathbf{s}_{t'})} \right) \left( \sum_{t'=t}^T r(\mathbf{s}_{t'}, \mathbf{a}_{t'})  \right)\right]$$
   为了避免以上连乘造成的数值不稳定，用蒙特卡洛采样：$$\begin{aligned}\nabla_{\theta'} J(\theta') =& \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=1}^T \nabla_{\theta'} \log \pi_{\theta'}(a_t|s_t) \left( \prod_{t'=1}^t \frac{\pi_{\theta'}(a_{t'}|s_{t'})}{\pi_{\theta}(a_{t'}|s_{t'})} \right) \hat{Q}_{t} \right]\\
   =&\mathbb{E}_{(s_{t},a_{t})\sim p_{\theta}(s,a)}\left[\sum_{t=1}^T\nabla_{\theta^{'}}\log\pi_{\theta^{'}}(a_t|s_t)\frac{p_{\theta^{'}}(s_{t},a_{t})}{p_{\theta}(s_{i,t},a_{i,t})}\hat{Q}_t\right]\\
   \approx& \frac{1}{N} \sum_{i=1}^N \left[ \sum_{t=1}^T \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t}|s_{i,t})  \frac{p_{\theta'}(s_{i,t}, a_{i,t})}{p_{\theta}(s_{i,t}, a_{i,t})}  \hat{Q}_{i,t} \right]\\
   =& \frac{1}{N} \sum_{i=1}^N  \sum_{t=1}^T \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t}|s_{i,t})  \frac{p_{\theta'}(s_{i,t}, a_{i,t})}{p_{\theta}(s_{i,t}, a_{i,t})}  \hat{Q}_{i,t} \\
   =&  \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \left( \frac{p_{\theta'}(s_{i,t})}{p_{\theta}(s_{i,t})} \cdot \frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}{\pi_{\theta}(a_{i,t}|s_{i,t})} \right) \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t}|s_{i,t}) \hat{Q}_{i,t}\\
   \xlongequal{p_{\theta}(s_{i,t})\approx p_{\theta^{'}}(s_{i,t})}& \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}{\pi_{\theta}(a_{i,t}|s_{i,t})}  \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t}|s_{i,t}) \hat{Q}_{i,t}
   \end{aligned}
   $$
3. actor-critic method
   1. 背景：在策略梯度中，我们使用了baseline，但是对baseline的朴素计算方法类似蒙特卡洛，他想要模拟的是$V^{\pi}(s_t)=\mathbb{E}_{a_t\sim\pi(s_t)}[Q^{\pi}(s_t,a_t)]$。所以实际上利用优势函数$A^{\pi}(s_t,a_t)=Q^{\pi}(s_t,a_t)-V^{\pi}(s_t)$作为策略梯度中的价值预测，即$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t}) A^\pi(\mathbf{s}_{i,t}, \mathbf{a}_{i,t})$是更好的策略，相当于选了更好的beseline。问题只在于$A^{\pi}(s_t,a_t)$的计算。由贝尔曼方程：
   $$Q^\pi(s_t, a_t) = r(s_t, a_t) + \mathbb{E}_{s_{t+1} \sim p(s_{t+1}|s_t, a_t)}[V^\pi(s_{t+1})]\approx r(s_t, a_t) + V^{\pi}(s_{t+1})$$得：
   $$A^{\pi}(s_t,a_t)\approx r(s_t, a_t) + V^{\pi}(s_{t+1})-V^{\pi}(s_{t})$$所以我们只需拟合$V^{\pi}(s)$即可。
   2. 策略估计：估计$V^\pi(\mathbf{s}_t) = \sum_{t'=t}^T E_{\pi_\theta} [r(\mathbf{s}_{t'}, \mathbf{a}_{t'}) | \mathbf{s}_t]$可利用$V^\pi(\mathbf{s}_t) \approx  \frac{1}{N} \sum_{i=1}^N \sum_{t'=t}^T r(\mathbf{s}_{t'}, \mathbf{a}_{t'})$,使用蒙特卡洛采样。但是我们若直接从初始state分布中采样、执行策略，算GT价值做成数据集会陷入分布迁移。除非能直接在执行策略下的state分布中采样，不然这个方法用不用了.
      1. 方法一：利用$V^{\pi}()s_t\approx \sum_{k=t}^Tr(s_k,a_k)$,搜集train data:$\{ ( \mathbf{s}_{i,t}, \underbrace{\sum_{t'=t}^T r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'})}_{y_{i,t}} ) \}$，做监督学习：$\mathcal{L}(\phi) = \frac{1}{2} \sum_i \| \hat{V}^\pi_\phi(\mathbf{s}_i) - y_i \|^2$
      2. 方法二：我们的目标：$y_{i,t} = \sum_{t'=t}^T E_{\pi_\theta} [r(\mathbf{s}_{t'}, \mathbf{a}_{t'}) | \mathbf{s}_{i,t}] \approx r(\mathbf{s}_{i,t}, \mathbf{a}_{i,t}) + V^\pi(\mathbf{s}_{i,t+1}) \approx r(\mathbf{s}_{i,t}, \mathbf{a}_{i,t}) + \hat{V}^\pi_\phi(\mathbf{s}_{i,t+1})$，(其中$\hat{V}^\pi_\phi$代表神经网络的预测)。因此可以:搜集train data:$\{ ( \mathbf{s}_{i,t}, \underbrace{r(\mathbf{s}_{i,t}, \mathbf{a}_{i,t}) + \hat{V}^\pi_\phi(\mathbf{s}_{i,t+1})}_{y_{i,t}} ) \}$，然后用同样的方法做监督学习。这种方法叫自举，又叫时序差分。
      3. 方法一方差大但无偏，方法二方差小，训练比较稳定，但有偏。方法二使用较多。
      4. 方法一二还可以取中，用$\{(s_{i,t},\sum_{m=t}^{t+n-1}r(s_{i,m},a_{i,m})+\hat{V}^{\pi}_{\phi}(s_{i,t+n}))\}$作为training data,其中n的值可以依具体任务特点而定。若考虑折扣因子，则变为$\{(s_{i,t},\sum_{m=t}^{t+n-1}r(s_{i,m},a_{i,m})+\gamma^{n}\hat{V}^{\pi}_{\phi}(s_{i,t+n}))\}$
   3. actor-critic 算法一般步骤：![](/imgs/imgs4rl-note/14.png)
   4. 无限时间步中对价值函数的设定（防止出现无穷大数值）：我们对蒙特卡洛采样进行折扣，有以下两种方式：
   $$\text{option 1: } \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t}) \left( \sum_{t'=t}^{T} \gamma^{t'-t} r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'}) \right)$$
   $$\text{option 2: } \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \gamma^{t-1} \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t}|\mathbf{s}_{i,t}) \left( \sum_{t'=t}^{T} \gamma^{t'-t} r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'}) \right)$$
   这里面的思想是，我们更关心最近的action获得的奖励，因此较近的动作梯度的奖励乘的折扣的指数较小。通常，我们用option1. 在此基础上，我们在做training data时用$y_{i,t}\approx r(s_{i,t},a_{i,t})+\gamma\hat V_{\phi}^{\pi}(s_{i,t+1})$作为GT价值。（依据是贝尔曼方程$Q^\pi(s_t, a_t) = \mathbb{E}_{s_{t+1}\sim p(s_{t+1}|s_t,a_t)} [r(s_t, a_t) + \gamma V^\pi(s_{t+1})]\approx r(s_t,a_t)+\gamma V^{\pi}(s_{t+1})$）
   5. 由以上，得到两种带折扣的算法：
   批量actor-critic：先收集一大批数据，再统一训练
   ![](/imgs/imgs4rl-note/15.png)
   在线actor-critic：每走一步，就学一点
   ![](/imgs/imgs4rl-note/16.png)
   6. off-policy actor critic：（用旧的策略去采样data更新新的策略）
      1. 修正：由于data使用旧策略采样所得，梯度项要用重要性采样。即$\nabla_{\theta'} J(\theta') \approx \sum_{t,i} \frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}{\pi_{\theta}(a_{i,t}|s_{i,t})} \nabla_{\theta'} \log \pi_{\theta'}(a_{i,t}|s_{i,t}) \hat{A}^{\pi_\theta}(s_{i,t}, a_{i,t})$。由$\nabla\pi_{\theta^{'}}(a_{i,t}|s_{i,t})=\pi_{\theta'}(a_{i,t}|s_{i,t})\nabla_{\theta'} \log \pi_{\theta'}(a_{i,t}|s_{i,t})$得到目标函数$J(\theta')=\sum_{t,i}\frac{\pi_{\theta^{'}}(a_{i,t}|s_{i,t})}{\pi_{\theta}(a_{i,t}|s_{i,t})}\hat{A}^{\pi_\theta}(s_{i,t}, a_{i,t})$。
      2. 这又引出第二个矛盾。当你经过多个梯度步后，新旧策略的差距会变大，这样会使重要性权重推导中的$\frac{p_{\theta'}(s_{i,t})}{p_{\theta}(s_{i,t})} \cdot \frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}{\pi_{\theta}(a_{i,t}|s_{i,t})}\approx \frac{\pi_{\theta'}(a_{i,t}|s_{i,t})}{\pi_{\theta}(a_{i,t}|s_{i,t})}$不成立。且容易发生新策略在旧策略产生的旧数据上过拟合。我们有两种解决方法（他们都算是ppo,介于on-policy和off-policy之间）：
         1. PPO：在目标函数中加入KL散度正则化项：$J(\theta')=\sum_{t,i}\frac{\pi_{\theta^{'}}(a_{i,t}|s_{i,t})}{\pi_{\theta}(a_{i,t}|s_{i,t})}\hat{A}^{\pi_\theta}(s_{i,t}, a_{i,t})+\beta \mathbf{D}_{kl}(\pi_{\theta'}(\cdot|s)||\pi_{\theta}(\cdot|s))$,$\beta$是超参数，用这个函数可以防止新策略在旧策略产生的旧数据上过拟合。
         2. PPO:做重要性权重做裁剪$ \text{clip} \left( \frac{\pi_{\theta'}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})}{\pi_{\theta}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})}, 1-\epsilon, 1+\epsilon \right)=\min(\max(\frac{\pi_{\theta'}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})}{\pi_{\theta}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})},1-\epsilon),1+\epsilon)$，则ppo的目标函数可以表示为$\tilde{J}(\theta') \approx \sum_{t,i} \min \left( \frac{\pi_{\theta'}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})}{\pi_{\theta}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})} \hat{A}^{\pi_\theta}(\mathbf{s}_{i,t}, \mathbf{a}_{i,t}), \text{clip} \left( \frac{\pi_{\theta'}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})}{\pi_{\theta}(\mathbf{a}_{i,t}|\mathbf{s}_{i,t})}, 1-\epsilon, 1+\epsilon \right) \hat{A}^{\pi_\theta}(\mathbf{s}_{i,t}, \mathbf{a}_{i,t}) \right)$
         3. PPO：为了更好估计优势函数，我们在蒙特卡洛（准确但高方差）和TD（低方差但有偏）之间做一个平滑的权衡。具体的，我们的优势函数如此估计：$$ \hat{A}_n^{\pi}(s_t, a_t) = \left( \sum_{t'=t}^{t+n} \gamma^{t'-t} r(s_{t'}, a_{t'}) \right) - \hat{V}_\phi^\pi(s_t) + \gamma^n \hat{V}_\phi^\pi(s_{t+n})$$加权平均得到最终的优势函数:$$\hat{A}_{GAE}^{\pi}(s_t, a_t) = \sum_{n=1}^{\infty} w_n \hat{A}_n^{\pi}(s_t, a_t) $$其中$w_n\propto\lambda^{n-1}$
         4. 总结ppo的流程：![](/imgs/imgs4rl-note/17.png)
      3. SAC:一种更加完全的off-policy method。每一轮次都执行策略搜集数据存入replay buffer，然后从replay buffer中采样一批数据$\{(s,a,r,s')\}$，实现数据复用。但若还用$\{s_i,r_i+\gamma\hat{V}^{\pi}_{\phi}(s_i')\}$做training data，则这里的$a$是旧策略下产生的动作，使得由它产生的那个$r$和$s$共同构成的$\text{Target}$ y不再能代表 V 函数所需要的那个“对所有$a\sim \pi$求期望"的平均值了。因此，我们换用神经网络估计Q函数。因为$Q^{\pi_\theta}(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_{\mathbf{s}' \sim p(\cdot|\mathbf{s}, \mathbf{a}), \bar{\mathbf{a}}' \sim \pi_\theta(\cdot|\mathbf{s}')} [Q^{\pi_\theta}(\mathbf{s}', \bar{\mathbf{a}}')]$,故我们从replay buffer里面采样$(s_i,a_i,s_i')$后，从**当前策略**中采样得$a_i'$，然后用$\{((s_i,a_i),r(s_i,a_i)+\gamma Q^{\pi_{\theta}}(s'_i,a'_i))\}$做监督训练的training data。然后目标函数也放弃用基线，将梯度由$\nabla_{\theta} J(\theta)=\frac{1}{N}\sum_i\nabla_{\theta}\log\pi_{\theta}(a_i|s_i)\hat{A}^{\pi}(s_i,a_i)$变为$\nabla_{\theta} J(\theta)=\frac{1}{N}\sum_i\nabla_{\theta}\log\pi_{\theta}(a_i|s_i)\hat{Q}^{\pi}(s_i,a_i)$。因而，SAC得算法流程：![](/imgs/imgs4rl-note/18.png)

   