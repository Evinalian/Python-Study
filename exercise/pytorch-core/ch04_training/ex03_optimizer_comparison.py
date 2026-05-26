"""
练习 4-3: 优化器对比

任务：
在同一个简单回归任务上对比 4 种优化器的性能。

要求：
1. 数据生成：
   - X: torch.randn(1000, 20)  (1000 个样本，20 维特征)
   - true_w: shape (20, 1)，从 N(0, 1) 采样
   - y = X @ true_w + noise (noise ~ N(0, 0.1))
   - 验证集: 200 个样本

2. 模型：
   - Linear(20, 128) -> ReLU -> Linear(128, 64) -> ReLU -> Linear(64, 1)
   - criterion: MSELoss

3. 四种优化器配置：
   a) SGD: lr=0.01
   b) SGD + Momentum: lr=0.01, momentum=0.9
   c) Adam: lr=0.001
   d) AdamW: lr=0.001, weight_decay=0.01

4. 训练：
   - 每种优化器训练 50 个 epoch
   - 记录每个 epoch 的训练 loss 和验证 loss

5. 输出：
   - 用一个 (2, 2) 子图或两条折线图，分别展示训练 loss 和验证 loss 随 epoch 变化的曲线
   - 四种优化器用不同颜色 + 图例区分
   - 分析哪种优化器收敛最快、最终 loss 最低

提示：
- 每次训练前需要重新初始化模型（或用 copy.deepcopy 复制初始模型）
- 确保每个优化器的训练起点相同（同一个初始模型）
- 使用 matplotlib 的 subplot 排列多个图
"""

import torch
import torch.nn as nn
import copy
import matplotlib.pyplot as plt


# TODO: 1. 生成回归数据


# TODO: 2. 定义 MLP 模型


# TODO: 3. 定义训练函数（给定 optimizer 类型，训练并返回 loss 历史）


# TODO: 4. 保存初始模型（deepcopy），用四种优化器分别训练


# TODO: 5. 绘制 loss 对比图


