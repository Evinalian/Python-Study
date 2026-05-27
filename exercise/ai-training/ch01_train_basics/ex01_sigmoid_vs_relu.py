"""
练习 1：修改激活函数 —— Sigmoid vs ReLU
==========================================

将第01章 NumPy MLP 中的 ReLU 激活函数替换为 Sigmoid，
观察训练效果的变化。

你需要完成:
1. 实现 sigmoid 函数及其导数
2. 在 ThreeLayerMLP 中支持选择激活函数（ReLU 或 Sigmoid）
3. 分别训练两个版本的模型，记录并对比:
   - 训练 loss 的下降速度
   - 最终验证准确率
   - 是否出现梯度消失现象（提示: 监控每层的梯度范数）

Sigmoid 公式:
    sigma(z) = 1 / (1 + exp(-z))
    sigma'(z) = sigma(z) * (1 - sigma(z))

思考题:
- Sigmoid 的梯度在什么情况下趋近于零？
- 为什么 Sigmoid 在深层网络中更容易导致梯度消失？
- 和 ReLU 相比，Sigmoid 的一个优点是什么？（提示: 输出范围）
"""

import numpy as np


# TODO: 实现 sigmoid 函数
def sigmoid(z):
    """
    Sigmoid 激活函数。

    参数:
        z: ndarray，任意形状
    返回:
        ndarray，sigmoid(z)
    """
    pass  # TODO: 你的代码


# TODO: 实现 sigmoid 的导数
def sigmoid_derivative(z):
    """
    Sigmoid 的导数: sigma(z) * (1 - sigma(z))

    参数:
        z: ndarray, sigmoid 的输入
    返回:
        ndarray, sigmoid 在 z 处的导数
    """
    pass  # TODO: 你的代码


# TODO: 修改 ReLU 层使其支持选择激活函数类型
class Activation:
    """
    通用激活层，支持 ReLU 和 Sigmoid。

    TODO: 实现 __init__, forward, backward 方法。
    提示: 在 __init__ 中保存激活类型，在 forward/backward 中根据类型调用不同函数。
    """
    def __init__(self, activation_type='relu'):
        """
        参数:
            activation_type: 'relu' 或 'sigmoid'
        """
        # TODO: 保存 activation_type
        pass

    def forward(self, z):
        # TODO: 根据 activation_type 应用不同的激活函数
        # 同时缓存 z 供 backward 使用
        pass

    def backward(self, dout):
        # TODO: 根据 activation_type 使用不同的导数
        pass


# TODO: 复制第01章教程中的 ThreeLayerMLP、train 等代码，
#       将 ReLU 层替换为 Activation 层（支持类型选择）


# TODO: 主实验 - 对比 ReLU 和 Sigmoid
def compare_activations():
    """
    TODO: 分别用 ReLU 和 Sigmoid 训练两个 MLP。
    记录并打印:
    - 每个 epoch 的训练 loss 和验证 accuracy
    - 训练完成后，可视化两者的 loss 曲线（用 matplotlib）
    - 分析 Sigmoid 是否出现梯度消失
    """
    pass


if __name__ == "__main__":
    compare_activations()
