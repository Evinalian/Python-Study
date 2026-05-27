"""
练习 3：实现 Post-Norm 版本
============================

将第02章教程中的 TransformerBlock 从 Pre-Norm 修改为 Post-Norm 结构。

Pre-Norm（当前实现）:
    x = x + Attention(Norm(x))
    x = x + FFN(Norm(x))

Post-Norm（原始 Transformer）:
    x = Norm(x + Attention(x))
    x = Norm(x + FFN(x))

你需要完成:
1. 实现 Post-Norm 版本的 TransformerBlock
2. 分别用 Pre-Norm 和 Post-Norm 训练模型（相同超参）
3. 对比两者:
   - 训练稳定性（loss 震荡程度）
   - 是否需要 warmup
   - 最终性能差异
4. 实验分析: 为什么 Pre-Norm 更稳定？

思考题:
- Post-Norm 中残差路径的输出没有经过归一化就进入了下一层，
  这会导致什么问题？
- 为什么原始 Transformer 论文用的是 Post-Norm 但仍能训练成功？
  （提示: 原始 Transformer 用了大量的 warmup 和小初始化）
"""

import torch
import torch.nn as nn


# TODO: 实现 Post-Norm 版本的 TransformerBlock
class TransformerBlockPostNorm(nn.Module):
    """
    Post-Norm 版本的 Transformer Block。

    结构:
        x = LayerNorm(x + Attention(x))
        x = LayerNorm(x + FFN(x))

    TODO: 实现 __init__ 和 forward
    """
    def __init__(self, d_model, n_heads, n_kv_heads=None, d_ff=None, max_seq_len=2048):
        # TODO: 初始化 Attention、FFN、LayerNorm
        pass

    def forward(self, x):
        # TODO: 实现 Post-Norm 前向传播
        pass


# TODO: 训练函数


# TODO: 主对比实验
def compare_pre_vs_post_norm():
    """
    TODO:
    1. 创建相同配置的 Pre-Norm 和 Post-Norm 模型
    2. 在相同数据上用相同超参训练
    3. 记录每个 step 的 loss
    4. 绘制对比 loss 曲线
    5. 观察 Post-Norm 是否需要更小的学习率或更长的 warmup
    """
    pass


if __name__ == "__main__":
    compare_pre_vs_post_norm()
