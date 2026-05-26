"""
练习 1-4（进阶）: 手写梯度下降实现多项式回归

任务：
用 autograd 手动梯度下降拟合二次多项式 y = 2x^2 - 3x + 1 + noise。

要求：
1. 数据生成：
   - 在 [-3, 3] 区间均匀采样 100 个点作为 x
   - 生成 y = 2*x^2 - 3*x + 1 + noise（noise 为标准正态分布，标准差 0.5）
   - x 的 shape 为 (100, 1)，y 的 shape 为 (100, 1)

2. 模型：y_pred = a * x^2 + b * x + c（即拟合三个参数 a, b, c）
   - 用 torch.randn 初始化三个参数，均 requires_grad=True
   - 不使用 nn.Module 和 torch.optim

3. 训练：
   - 损失函数：均方误差 MSE
   - 学习率 lr = 0.01
   - 训练 500 个 epoch
   - 每 100 个 epoch 打印一次 loss

4. 结果输出：
   - 打印学习到的参数 a, b, c 与真实值 2, -3, 1 对比
   - 用 matplotlib 绘制数据散点图和拟合曲线在同一张图上

提示：
- 手动更新参数时必须包裹在 `with torch.no_grad():` 中
- 每次 backward 前需要清零梯度 .grad.zero_()
- 拟合曲线绘制：在 [-3, 3] 上密集采样点，用学习到的 a, b, c 计算 y 值
"""

import torch
import matplotlib.pyplot as plt


# TODO: 1. 设置随机种子 torch.manual_seed(42)


# TODO: 2. 生成 x: torch.linspace(-3, 3, 100).reshape(-1, 1)


# TODO: 3. 生成带噪声的 y


# TODO: 4. 初始化参数 a, b, c (requires_grad=True)


# TODO: 5. 设置学习率和 epoch 数


# TODO: 6. 训练循环: forward -> loss -> backward -> 更新参数 -> 清零梯度


# TODO: 7. 打印学习到的参数与真实参数对比


# TODO: 8. 绘制散点图和拟合曲线


