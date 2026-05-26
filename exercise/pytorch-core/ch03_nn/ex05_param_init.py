"""
练习 3-5（进阶）: 参数初始化对比实验

任务：
比较三种初始化方法对同一 MLP 训练的影响。

要求：
1. 构建统一 MLP：input_dim=50, hidden_dim=128, output_dim=1, 共 5 个隐藏层

2. 三种初始化：
   a) 零初始化：所有权重和 bias 初始化为 0
   b) 大随机初始化：权重 ~ N(0, 1.0)，bias = 0
   c) Kaiming 初始化：用 init.kaiming_normal_(w, nonlinearity='relu')，bias = 0

3. 数据：
   - X: torch.randn(500, 50)，y: 某个非线性函数（如 X 前三列的加权和 + 非线性变换）
   - 建议: y = torch.sin(X[:, 0:1] * 2) + 0.5 * X[:, 1:2]**2 + X[:, 2:3] * 0.3 + noise

4. 训练配置：
   - MSELoss
   - SGD optimizer, lr=0.01
   - 训练 30 个 epoch，记录每个 epoch 的平均 loss

5. 输出：
   - 用 matplotlib 绘制三条 loss 曲线（不同颜色 + 图例）
   - 用不同 y 轴刻度可能有助于展示（zero init 的 loss 可能很大）
   - 分析哪种初始化收敛最快、最终 loss 最低

提示：
- model.apply(fn) 对模型所有子模块递归调用 fn
- 在 apply 的回调函数中根据 isinstance(m, nn.Linear) 判断层类型
- 零初始化的 loss 可能非常大，但不会完全不下降（因为有 bias）
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt


# TODO: 1. 定义 MLP 类


# TODO: 2. 生成训练数据


# TODO: 3. 定义三种初始化的 apply 函数


# TODO: 4. 分别训练三个模型（zero, large_random, kaiming）并记录 loss


# TODO: 5. 绘制对比图并分析


