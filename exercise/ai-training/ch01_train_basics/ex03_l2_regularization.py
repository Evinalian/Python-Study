"""
练习 3：添加 L2 正则化
=======================

在第01章教程中添加 L2 正则化（权重衰减）。

L2 正则化:
    L_total = L_original + (lambda / 2) * sum(W^2)
    dL/dW = dL_original/dW + lambda * W

你需要完成:
1. 修改损失计算，加入 L2 正则化项
2. 修改梯度计算，对每个权重矩阵加上 lambda * W 的梯度贡献
3. 实验不同的 lambda 值（1e-5, 1e-4, 1e-3, 1e-2），观察:
   - 训练 loss vs 验证 loss 的差距（过拟合程度）
   - 权重范数的分布变化

注意:
- L2 正则化只应用于权重矩阵 W，不应用于偏置 b
- 偏置 b 不会导致过拟合，不需要正则化

思考题:
- lambda 太大和太小分别有什么后果？
- 为什么 L2 正则化也叫 Weight Decay？
- 在 Adam 优化器中 L2 正则化与 Weight Decay 有何不同？（引出 AdamW）
"""

import numpy as np


# TODO: 修改交叉熵损失函数以支持 L2 正则化
def cross_entropy_loss_with_l2(probs, labels, params, lambda_reg=0.0):
    """
    带 L2 正则化的交叉熵损失。

    参数:
        probs: softmax 概率, (N, num_classes)
        labels: 整数标签, (N,)
        params: 层列表 [fc1, fc2, fc3]，每个层的 W 参与正则化
        lambda_reg: L2 正则化系数

    返回:
        total_loss: L_ce + (lambda/2) * sum(W^2 for W in weights)

    TODO: 实现原始交叉熵损失 + L2 正则化项
    """
    pass  # TODO: 你的代码


# TODO: 修改梯度计算，加入 L2 正则化的梯度贡献
def apply_l2_gradient(layers, lambda_reg):
    """
    对每个层的 dW 加上 lambda * W 的贡献。

    参数:
        layers: 全连接层列表 (每个有 W, dW 属性)
        lambda_reg: L2 正则化系数

    TODO: 遍历 layers，对每个 layer.dW 加上 lambda_reg * layer.W
    注意: 不修改 db（偏置不参与正则化）
    """
    pass  # TODO: 你的代码


# TODO: 修改 train 函数，集成 L2 正则化


# TODO: 主实验
def experiment_l2_regularization():
    """
    TODO:
    1. 训练 4 个模型，lambda = [0, 1e-5, 1e-4, 1e-3]
    2. 记录每个模型的 train_loss 和 val_loss
    3. 绘制 train_loss vs val_loss 的对比图
    4. 打印每个模型的权重范数 (||W||_F for each layer)
    5. 分析: 哪个 lambda 在"防止过拟合"和"不伤害拟合能力"之间平衡最好？
    """
    pass


if __name__ == "__main__":
    experiment_l2_regularization()
