"""
练习 2：实现 Momentum 优化器
=============================

在第01章教程的 SGD 基础上实现 Momentum 优化器。

Momentum 更新公式:
    v_t = beta * v_{t-1} + lr * gradient
    w = w - v_t

你需要完成:
1. 实现 MomentumOptimizer 类
2. 为每个参数维护一个 velocity 缓存（初始化为零）
3. 比较 Momentum (beta=0.9) 和 SGD 在相同学习率下的收敛速度
4. 实验不同的 beta 值 (0.5, 0.9, 0.99) 对训练的影响

思考题:
- beta=0 时 Momentum 退化成什么？
- beta 接近 1 意味着什么？有什么优缺点？
- 为什么 Momentum 能加速"峡谷地形"中的收敛？
"""

import numpy as np


# TODO: 实现 Momentum 优化器
class MomentumOptimizer:
    """
    Momentum (动量) 优化器。

    公式:
        v = beta * v + lr * grad
        param = param - v

    TODO: 实现 __init__ 和 update 方法
    """

    def __init__(self, params, lr=0.01, beta=0.9):
        """
        参数:
            params: 包含参数的层列表（如 [fc1, fc2, fc3]）
            lr: 学习率
            beta: 动量系数（0 <= beta < 1）
        """
        # TODO: 保存参数
        # TODO: 为每个参数创建 velocity 缓存（与参数形状相同的零矩阵）
        pass

    def update(self):
        """
        TODO: 对所有参数应用 Momentum 更新。
        每个参数的 velocity 需要保存在 self.velocities 中。

        提示: velocity[i] = beta * velocity[i] + lr * param.dW (或 param.db)
               param.W = param.W - velocity_for_W
        """
        pass


# TODO: 复制第01章教程中的 ThreeLayerMLP 和相关函数，
#       修改 train 函数以支持不同的优化器


# TODO: 主实验
def compare_momentum_vs_sgd():
    """
    TODO:
    1. 用 SGD (lr=0.01) 训练模型，记录 train_loss 历史
    2. 用 Momentum (lr=0.01, beta=0.9) 训练模型，记录 train_loss 历史
    3. 用 matplotlib 绘制两条 loss 曲线对比
    4. 实验 beta=0.5, 0.99，分析 beta 对收敛的影响
    """
    pass


# TODO: 加分项 - 实现 Nesterov Momentum
# NAG 公式:
#   lookahead = param - beta * v
#   v = beta * v + lr * gradient_at(lookahead)
#   param = param - v


if __name__ == "__main__":
    compare_momentum_vs_sgd()
