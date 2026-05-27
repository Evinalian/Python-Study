"""
练习 4（进阶）：从零实现 BatchNorm
==================================

从零实现 Batch Normalization 层，不依赖任何框架。

你需要完成:
1. 实现训练模式下的 BatchNorm 前向传播:
   - 沿 batch 维度计算均值和方差
   - 归一化: x_hat = (x - mean) / sqrt(var + eps)
   - 缩放和平移: out = gamma * x_hat + beta
2. 维护 running_mean 和 running_var（指数移动平均），用于测试模式
3. 实现 BatchNorm 的反向传播（手动推导梯度）:
   - dL/dx_hat = dout * gamma
   - dL/dvar = sum(dL/dx_hat * (x - mean) * (-0.5) * (var + eps)^(-1.5))
   - dL/dmean = sum(dL/dx_hat * (-1/sqrt(var + eps))) + dL/dvar * (-2/N) * sum(x - mean)
   - dL/dx = dL/dx_hat / sqrt(var + eps) + dL/dvar * 2 * (x - mean) / N + dL/dmean / N
   - dL/dgamma = sum(dout * x_hat)
   - dL/dbeta = sum(dout)
4. 在 MLP 中插入 BatchNorm 层（放在 Linear 之后、ReLU 之前）
5. 比较有无 BatchNorm 对训练速度和最终准确率的影响

数学参考:
    mu = mean(x, axis=0)
    sigma2 = var(x, axis=0)
    x_hat = (x - mu) / sqrt(sigma2 + eps)
    y = gamma * x_hat + beta

思考题:
- 为什么 BatchNorm 能加速训练？（提示: 内部协变量偏移）
- 为什么训练和测试模式不同？
- BatchNorm 为什么不适合 batch_size=1 的场景？
"""

import numpy as np


# TODO: 实现 BatchNorm 层
class BatchNormLayer:
    """
    Batch Normalization 层。

    TODO: 实现 __init__, forward, backward 方法。

    forward 需要支持两种模式:
    - training=True: 用当前 batch 的 mean/var，同时更新 running_mean/running_var
    - training=False: 用 running_mean/running_var

    running_mean 和 running_var 使用指数移动平均更新:
        running_mean = momentum * running_mean + (1 - momentum) * batch_mean
        running_var  = momentum * running_var  + (1 - momentum) * batch_var
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        """
        参数:
            num_features: 特征维度（通道数）
            eps: 防止除零的小常数
            momentum: running mean/var 的动量系数
        """
        # TODO: 初始化 gamma 为 1, beta 为 0
        # TODO: 初始化 running_mean 为 0, running_var 为 1
        pass

    def forward(self, x, training=True):
        """
        参数:
            x: (N, D) - N 是 batch_size, D 是特征维度
            training: 是否在训练模式
        返回:
            y: (N, D) 归一化并缩放后的输出
        """
        # TODO: 区分训练和测试模式
        # 训练时: 计算 batch mean/var, 更新 running, 返回归一化结果
        # 测试时: 用 running mean/var 做归一化
        pass

    def backward(self, dout):
        """
        参数:
            dout: 上游梯度 (N, D)
        返回:
            dx: 对输入 x 的梯度 (N, D)
        """
        # TODO: 手动推导并实现 BatchNorm 的反向传播
        # 需要计算: dgamma, dbeta, dx
        pass


# TODO: 修改教程中的 ThreeLayerMLP，在 Linear 和 ReLU 之间插入 BatchNormLayer


# TODO: 主实验
def compare_with_without_bn():
    """
    TODO:
    1. 训练不带 BatchNorm 的 MLP（基准）
    2. 训练带 BatchNorm 的 MLP
    3. 对比两者:
       - 训练 loss 下降速度
       - 最终验证准确率
       - 是否可以用更大的学习率
    """
    pass


if __name__ == "__main__":
    compare_with_without_bn()
