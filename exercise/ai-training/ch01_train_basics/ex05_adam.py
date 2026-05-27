"""
练习 5（进阶）：实现 Adam 优化器
================================

从零实现 Adam 优化器。

Adam 更新公式:
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t          (一阶矩)
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2        (二阶矩)
    m_hat = m_t / (1 - beta1^t)                         (偏差校正)
    v_hat = v_t / (1 - beta2^t)                         (偏差校正)
    theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)

你需要完成:
1. 实现 AdamOptimizer 类
2. 为每个参数维护 m (一阶矩) 和 v (二阶矩) 缓存
3. 实现偏差校正
4. 比较 Adam 和 SGD/Momentum 在相同任务上的:
   - 收敛速度
   - 最终准确率
   - 对学习率的敏感度（尝试 lr = 0.1, 0.01, 0.001, 0.0001）

思考题:
- 为什么需要偏差校正？
- beta1 和 beta2 分别控制什么？
- Adam 为什么对超参数不那么敏感？
"""

import numpy as np


# TODO: 实现 Adam 优化器
class AdamOptimizer:
    """
    Adam 优化器。

    TODO: 实现 __init__ 和 update 方法。

    状态:
        self.m: 一阶矩（动量），字典 {layer_id: {'W': array, 'b': array}}
        self.v: 二阶矩（RMSprop），字典 {layer_id: {'W': array, 'b': array}}
        self.t: 更新步数计数器
    """

    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        参数:
            params: 包含参数的层列表
            lr: 学习率
            beta1: 一阶矩衰减系数
            beta2: 二阶矩衰减系数
            eps: 防止除零
        """
        # TODO: 保存超参数
        # TODO: 初始化 m 和 v 为零
        # TODO: 初始化步数计数器 t = 0
        pass

    def update(self):
        """
        TODO: 对每个参数应用 Adam 更新。

        步骤:
        1. t += 1
        2. 对每层的 W 和 b:
           - 更新 m = beta1 * m + (1 - beta1) * grad
           - 更新 v = beta2 * v + (1 - beta2) * grad^2
           - 偏差校正: m_hat = m / (1 - beta1^t); v_hat = v / (1 - beta2^t)
           - 参数更新: param = param - lr * m_hat / (sqrt(v_hat) + eps)
        """
        pass


# TODO: 主实验 - 对比所有优化器
def compare_optimizers():
    """
    TODO:
    训练 4 个模型，分别使用:
    - SGD (lr=0.01)
    - Momentum (lr=0.01, beta=0.9)
    - Adam (lr=0.001, beta1=0.9, beta2=0.999)
    - 你自己实现的 Adam

    对比:
    - 训练 loss 下降曲线（用 matplotlib 绘制）
    - 最终验证准确率
    - 不同 lr 下的稳定性

    额外: 尝试实现 AdamW（将权重衰减从损失函数中解耦）
    """
    pass


if __name__ == "__main__":
    compare_optimizers()
