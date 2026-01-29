---
title: "RL4LLM"
math: true
date: 2026-01-15
draft: false
cover:
    # 这是一个自动生成风景图的网站，每次刷新都不一样
    image: "https://picsum.photos/id/7/800/400" 
    alt: "随机封面"
    caption: " "
---

## 大语言模型的强化学习
1. 指令微调（instruction fineturning）：不算RL，算监督学习
   1. 通过预训练的model只掌握了通用的知识，我们通过微调要让它按我们期望的方式回答问题。
   2. 数据集：$\{(Q_i,A_i)\}$，其中$Q_i$为问题，$A_i$为回答，都为文本形式。
   3. 损失函数：自回归的交叉熵。设$A_i$分词为$(a_{i1},a_{i2}...a_{iL})$，则问答$(Q_i,A_i)$的损失为：$$            \mathcal{L}_i = - \sum_{t=1}^{L} \log P_{\theta}(a_{i,t} | Q_i, a_{i,1}, ..., a_{i, t-1})$$这是一个自回归过程。总损失函数：$\mathcal{L}(\theta)=\sum_{k=1}^{N}\mathcal{L}_k$。
   4. 局限：很多时候学的都是格式，而不是对问题的理解。并且十分依赖数据质量，很容易陷入偏见。并且容易陷入“只有一个正确答案”的误区。
2. RLHF(核心思想和奖励学习差不多)
   1. 奖励模型的训练：我们有一个数据集$\mathcal{D}:\{(x,y_w,y_l)\}$，$x$是问题，$y_w$表示较好的回答，$y_l$表示较差的回答。我们的奖励模型$RM_{\phi}$会对问题的回答进行打分，则训练的损失函数为：$J_{RM}(\phi) = -\mathbb{E}_{(x, y^w, y^l) \sim D}\left[\log \sigma\left(RM_{\phi}(x, y^w) - RM_{\phi}(x, y^l)\right)\right]$，其中$\sigma$是$\text{sigmod}$函数。
   2. 总的RLHF的目标函数是：$$
    \text{Maximize: } \mathbb{E}_{\hat{y} \sim p_{\theta}^{\text{RL}}( \hat{y}|x)} \left[ RM_{\phi}(x, \hat{y}) \right] - \beta \cdot D_{KL} \left( p_{\theta}^{\text{RL}}(\cdot|x) \ || \ p^{\text{PT}}(\cdot|x) \right)
    \\=\mathbb{E}_{\hat{y} \sim p_{\theta}^{RL}(\hat{y}|x)} \left[ RM_{\phi}(x, \hat{y}) - \beta \log \left( \frac{p_{\theta}^{RL}(\hat{y}|x)}{p^{PT}(\hat{y}|x)} \right) \right]
    $$其中$p^{PT}(\cdot|x)$代表预训练模型或经过指令微调的模型。第二项是限制RL微调的模型不要离原来太远，否则会导致模型钻空或模式崩溃。
3. DPO(Direct Preference Optimization)
   1. 动机：在RLHF中，我们搜集了人类偏好的数据集$\mathcal{D}:\{(x,y_w,y_l)\}$，但是还是要训练一个奖励函数，并且在最终微调时还要不断的运行LLM搜集在线数据。我们想要跳过奖励函数直接将偏好注入LLM。
   2. 推导：RLHF的目标函数回顾：$$\max_{\theta} \quad \mathbb{E}_{\hat{y} \sim p_{\theta}^{RL}(\hat{y}|x)} \left[ RM(x, \hat{y}) - \beta \log \left( \frac{p_{\theta}^{RL}(\hat{y}|x)}{p^{PT}(\hat{y}|x)} \right) \right]$$其实，上式是有解析解的，其解析解为：$$p^*(\hat{y}|x) = \frac{1}{Z(x)} p^{PT}(\hat{y}|x) \exp\left(\frac{1}{\beta} RM(x, \hat{y})\right)$$其中$Z(x)$是用来做归一化的，暂时不管。因此，我们可以由最优策略反解出奖励函数：$$RM(x, \hat{y}) = \beta \log\frac{p^*(\hat{y}|x)}{p^{PT}(\hat{y}|x)} + \beta \log Z(x)$$这里揭示了一个很深刻的道理：在预训练模型$p^{PT}(\hat{y}|x)$给定的情况下，奖励函数和最优模型是一一对应的。而正确的奖励函数应满足最小化以下式子：$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y^w, y^l) \sim D}\left[\log \sigma\left(RM(x, y^w) - RM(x, y^l)\right)\right]$$则，最优模型对应的就是正确的奖励函数，也应该最小化上式，带入得到：$$\mathcal{L}_{DPO}(p_{\theta}; p^{PT}) = -\mathbb{E}_{(x, y^w, y^l) \sim \mathcal{D}}\left[\log \sigma\left( \beta \log \frac{p_{\theta}(y^w|x)}{p^{PT}(y^w|x)} - \beta \log \frac{p_{\theta}(y^l|x)}{p^{PT}(y^l|x)} \right)\right]$$归一化项$Z(x)$完美的被抵消了，得到了我们最终的模型训练损失函数。直接在人类偏好数据集$\mathcal{D}$上对模型进行训练，跳过了奖励函数。