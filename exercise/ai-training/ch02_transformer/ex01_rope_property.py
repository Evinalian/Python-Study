"""
练习 1：验证 RoPE 的相对位置性质
=================================

RoPE 的关键性质是：应用 RoPE 后，Q 和 K 的点积只依赖于它们之间的相对位置，
而不是绝对位置。即:

    RoPE(q, m) · RoPE(k, n) = f(q·k, m - n)

你需要完成:
1. 从第02章教程中复制 RoPE 的实现
2. 编写测试验证上述性质:
   - 生成随机向量 q, k
   - 计算 (RoPE(q, m) · RoPE(k, n)) 在固定 (m-n) 但不同 (m, n) 下的值
   - 验证它们相等（或在数值误差范围内一致）
3. 对于绝对位置 (m, n) = (5, 10) 和 (15, 20)，计算两者的点积
   如果 RoPE 工作正常，它们的点积应该非常接近（因为 m-n 都是 -5）

思考题:
- 这个性质为什么重要？（提示: 训练时见过的位置和推理时的新位置）
- 如果 RoPE 依赖绝对位置，会出现什么问题？
"""

import torch


# TODO: 从教程中复制 RotaryPositionalEmbedding 类


# TODO: 实现验证函数
def verify_rope_relative_property(rope_fn, dim=64):
    """
    TODO: 验证 RoPE 的相对位置性质。

    步骤:
    1. 生成随机向量 q, k，形状均为 (dim,)
    2. 对于多组 (m, n)，保持 m-n 不变但改变 m 和 n:
       - 例如 (m, n) = (0, 5), (10, 15), (100, 105) —— m-n 都是 -5
       - 对每组分别应用 RoPE 并计算点积
    3. 比较这些点积，计算最大偏差
    4. 打印结果，判断性质是否成立

    返回:
        max_diff: 各组点积之间的最大偏差
    """
    pass  # TODO: 你的代码


# TODO: 可视化
def visualize_rope_encoding(dim=64, max_pos=512):
    """
    TODO: 可视化 RoPE 的编码模式。

    对 position=0..max_pos-1，计算每个位置的位置编码（在 RoPE 中即是旋转角度）
    用 matplotlib 绘制不同频率成分的热力图或线图。
    观察低频和高频成分的行为。
    """
    pass


if __name__ == "__main__":
    # TODO: 创建 RoPE 实例并验证性质
    pass
