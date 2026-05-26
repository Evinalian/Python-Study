"""
练习 3-4（进阶）: 带残差连接的 MLP

任务：
1. 实现 ResidualBlock(nn.Module)：
   - dim：输入/输出维度（保持不变）
   - 结构：Linear(dim, dim) -> ReLU -> Linear(dim, dim)
   - forward: out = self.layers(x); return out + x; (残差连接后再 ReLU)

2. 实现 ResMLP(nn.Module)：
   - input_dim, hidden_dim, output_dim, num_blocks
   - input_proj: Linear(input_dim, hidden_dim) — 维度对齐
   - blocks: 堆叠 num_blocks 个 ResidualBlock(hidden_dim)
   - output_proj: Linear(hidden_dim, output_dim)

3. 实现 PlainMLP(nn.Module) — 同样深度但不带残差：
   - 每层 Linear(hidden_dim, hidden_dim) -> ReLU
   - 同样数量的层

4. 对比实验：
   - 生成简单回归数据：y = f(X) 其中 X 是 20 维，y 是 1 维
   - 同样 hidden_dim=128, num_blocks=8
   - 分别训练 ResMLP 和 PlainMLP 50 个 epoch
   - 记录并打印两者的训练 loss 曲线（可用列表记录每 epoch 的 loss）

5. 分析：
   - 残差版本是否更容易收敛？
   - 残差版本的最终 loss 是否更低？

提示：
- ResidualBlock.forward: identity = x; out = fc2(relu(fc1(x))); return relu(out + identity)
- 使用 MSELoss 和手动梯度下降（或 SGD optimizer）
- 用 matplotlib 绘制两条 loss 曲线做对比
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# TODO: 1. 定义 ResidualBlock


# TODO: 2. 定义 ResMLP


# TODO: 3. 定义 PlainMLP（同样深度，无残差连接）


# TODO: 4. 生成回归数据


# TODO: 5. 训练两个模型，记录 loss 曲线


# TODO: 6. 绘制对比图并分析


