---
title: "奖励学习"
math:true
date: 2026-01-17
draft: false
cover:
    # 这是一个自动生成风景图的网站，每次刷新都不一样
    image: "https://picsum.photos/800/400" 
    alt: "随机封面"
    caption: " "
---

## 奖励学习

1. 说明：对于很多现实世界的复杂任务（如模糊任务、用户偏好），我们根本无法手动写出奖励函数$r(s,a)$，只能用神经网络拟合。

2. 目标分类器(Goal Classifiers)
   
   具体做法是，train一个分类器，这个分类器接受一个state作为输入，这个state成功的概率作为输出。有了这个分类器，我们就可以套用REINFORCED等算法进行策略的学习。

   1. 具体的算法步骤如下：![](/imgs/imgs4rl-note/26.png)
   2. 注意之一：图中算法动态地把策略执行中的state也加入数据集了。这是因为我们初始训练分类器的数据是有限的，分类器也有其局限，不能涵盖策略遇到的所有状态。若后面不加数据对分类器更新，策略可能会钻分类器的空子。
   3. 你会发现，我们把策略运行得到的state的标签一律标为负（失败），但同时还有保持数据集中正负样本数量平衡（1：1）。这样可以避免数据集中失败样本太多，导致训练崩溃。
   4. 训练分类器的损失函数可以写为：若分类器：$c_{\psi}(s)\in[0,1]$，则损失函数为$\mathcal{L}(\psi)=\mathbb{E}_{s^+\sim\mathcal{D}^+}[-\log(c_{\psi}(s^+))]+\mathbb{E}_{s^-\sim\mathcal{D}^-}[-\log(1-c_{\psi}(s^-))]$
   5. 只要我们的数据集平衡，则数据集中至少有一半以上的真正的成功状态的分类器输出也是真的，则至少可以保证训练出来的分类器：$\mathbf{P}(\text{Classify}(s)=1|s\text{是成功状态})>0.5$。

3. 从人类反馈中学习偏好
   
   1. 若我们有一个数据集，形式为$\{(\tau_{g},\tau_b)\}$，其中$\tau_g$为较好的轨迹，$\tau_b$为较差的轨迹（人工标注）。则我们的奖励网络应有：$\sum_{(s,a)\sim\tau_g}r_{\theta}(s,a)>\sum_{(s,a)\sim\tau_b}r_{\theta}(s,a)$。因而只需在损失函数$\mathcal{L}(\theta)=\mathbb{E}_{(\tau_g,\tau_b)\sim\mathcal{D}}[\log(\text{sigmod}(r_{\theta}(\tau_g)-r_{\theta}(\tau_b)))]$训练奖励网络即可。
   2. 从以上拓展一下：请人对k条轨迹进行排序，就得到$C_k^2$个比较关系。总体的流程如下：![](/imgs/imgs4rl-note/27.png)