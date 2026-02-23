---
title: "影响力函数"
math: true
date: 2026-02-10
draft: false
cover:
    # 这是一个自动生成风景图的网站，每次刷新都不一样
    image: "https://picsum.photos/id/2/800/400" 
    alt: "随机封面"
    caption: " "
---
## 影响力函数（Influence Functions）
目标： 推导出一个数学公式，量化“如果我把训练样本 $z$ 的权重增加 $\epsilon$，模型在测试点 $z_{test}$ 上的 Loss 会怎么变”。

### 1. 符号定义与维度
*   **模型参数**：$\theta \in \mathbb{R}^P$。对于 1.3B 模型，$P \approx 1.3 \times 10^9$。
*   **训练样本**：$z = (x, y)$。
*   **损失函数**：$L(z, \theta)$。通常是交叉熵损失。
*   **经验风险**：$R(\theta) = \frac{1}{N} \sum_{i=1}^N L(z_i, \theta)$。
*   **最优参数**：$\hat{\theta} = \arg\min R(\theta)$。满足 $\nabla_{\theta} R(\hat{\theta})=\nabla_{\theta} R({\theta})|_{\hat{\theta}} = 0$。
### 2.带黑塞矩阵的影响力函数的推导
定义扰动后的风险函数：$R_\epsilon(\theta) = R(\theta) + \epsilon L(z, \theta)$，令扰动后的最优参数为 $\hat{\theta}_\epsilon$，则$ \nabla R_\epsilon(\hat{\theta}_\epsilon) = 0 $，即：$$ \nabla R(\hat{\theta}_\epsilon) + \epsilon \nabla L(z, \hat{\theta}_\epsilon) = 0 $$
在原参数 $\hat{\theta}$ 处对 $\nabla R(\hat{\theta}_\epsilon)$ 进行一阶展开：
$$ \nabla R(\hat{\theta}_\epsilon) \approx \nabla R(\hat{\theta}) + \nabla^2 R(\hat{\theta})(\hat{\theta}_\epsilon - \hat{\theta}) $$
因为$\nabla R(\hat{\theta}_\epsilon)=\nabla_{\theta}R(\hat{\theta}_\epsilon)=0$,且定义 Hessian 矩阵 $H = \nabla^2 R(\hat{\theta}) \in \mathbb{R}^{P \times P}$，则带回方程：$$ H(\hat{\theta}_\epsilon - \hat{\theta}) + \epsilon \nabla L(z, \hat{\theta}) \approx 0 $$所以：

$$ \frac{d\theta}{d\epsilon}|_{\epsilon=0} = \lim_{\epsilon \to 0} \frac{\hat{\theta}_\epsilon - \hat{\theta}}{\epsilon} = -H^{-1} \nabla L(z, \hat{\theta}) $$