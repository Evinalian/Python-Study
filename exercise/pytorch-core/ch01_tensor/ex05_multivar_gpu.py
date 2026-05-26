"""
练习 1-5（进阶）: 多变量线性回归 + GPU 加速

任务：
生成包含 5 个特征的多变量线性回归数据，将数据和参数都移到 GPU 上，用 autograd 完成训练。

要求：
1. 数据生成：
   - 生成 X: shape (1000, 5)，标准正态分布 N(0, 1)
   - 真实权重 true_w: shape (5, 1)，从标准正态分布采样
   - 真实偏置 true_b: 标量，从标准正态分布采样
   - 生成 y = X @ true_w + true_b + noise（noise ~ N(0, 0.1)）
   - 打印 true_w 和 true_b 的值，以便对比

2. GPU 迁移：
   - 检查 torch.cuda.is_available()，如果可用则用 GPU，否则用 CPU
   - 将 X, y 和参数都移到目标 device 上

3. 训练：
   - 参数 w: shape (5, 1), b: 标量，均 requires_grad=True
   - 损失：MSE
   - lr = 0.1
   - 训练 2000 个 epoch
   - 每 500 epoch 打印 loss 和当前 w 的范数

4. 结果：
   - 打印学习到的 w 与 true_w 对比（逐元素）
   - 打印学习到的 b 与 true_b 对比
   - 计算最终 MSE loss

提示：
- 所有参与计算的 Tensor 必须在同一 device 上
- 用 `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` 统一管理
- 如果 CUDA 不可用，代码应在 CPU 上正常运行
"""

import torch


# TODO: 1. 检查 GPU 可用性，设置 device


# TODO: 2. 生成训练数据 (X, y) 并打印真实参数


# TODO: 3. 将数据移到 device 上


# TODO: 4. 在 device 上初始化参数 w, b (requires_grad=True)


# TODO: 5. 训练循环


# TODO: 6. 对比学习到的参数与真实参数


