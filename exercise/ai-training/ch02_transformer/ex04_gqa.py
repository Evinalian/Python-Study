"""
练习 4（进阶）：实现 Grouped Query Attention (GQA)
==================================================

在 MultiHeadCausalAttention 中添加 GQA 支持。

GQA 的思想: 多个 Q 头共享同一个 K/V 头。例如:
- n_heads = 8, n_kv_heads = 2
- Q 有 8 个头，K/V 各只有 2 个头
- 每 4 个 Q 头共享 1 组 KV

你需要完成:
1. 修改 MultiHeadCausalAttention，添加 n_kv_heads 参数
2. 当 n_kv_heads < n_heads 时，K/V 投影后 repeat_interleave 以匹配 Q 的维度
3. 分析 GQA 对 KV Cache 显存的影响:
   - 计算标准 MHA 和 GQA 在推理时的 KV Cache 大小
   - 对 LLaMA 2 70B (n_layers=80, d_model=8192, n_heads=64, n_kv_heads=8):
     生成 4096 个 token 时，KV Cache 分别是多少？
4. 实验: 用相同参数量训练 GQA 和 MHA 版本，比较性能和推理速度

思考题:
- GQA 为什么能提升推理速度？（提示: KV Cache 更小 = 更少的数据移动）
- GQA 是否损害模型质量？为什么？
- GQA 和 MQA（Multi-Query Attention, n_kv_heads=1）的关系是什么？
"""

import torch
import torch.nn as nn
import math


# TODO: 修改教程中的 MultiHeadCausalAttention，添加 GQA 支持
class MultiHeadAttentionWithGQA(nn.Module):
    """
    支持 GQA 的多头注意力。

    TODO: 实现 __init__ 和 forward。

    当 n_kv_heads == n_heads 时 = 标准 MHA
    当 n_kv_heads == 1 时 = MQA (Multi-Query Attention)
    当 1 < n_kv_heads < n_heads 时 = GQA
    """
    def __init__(self, d_model, n_heads, n_kv_heads=None, max_seq_len=2048):
        # TODO: W_Q 投影到 n_heads * d_k 维
        # TODO: W_K, W_V 投影到 n_kv_heads * d_k 维（更少的头）
        # TODO: 注意 repeat_factor = n_heads // n_kv_heads 必须是整数
        pass

    def forward(self, x):
        # TODO:
        # 1. Q 投影到所有 n_heads
        # 2. K, V 投影到 n_kv_heads
        # 3. K 和 V 通过 repeat_interleave 扩展到 n_heads
        # 4. 后续计算与标准 Attention 相同
        pass


# TODO: 计算并比较 KV Cache 大小
def compare_kv_cache_size(n_layers, d_model, n_heads, n_kv_heads_list, seq_len):
    """
    TODO: 计算不同 KV 头数下的 KV Cache 大小和节省比例。

    参数:
        n_layers: Transformer 层数
        d_model: 隐藏维度
        n_heads: Q 头总数
        n_kv_heads_list: 不同的 KV 头数列表，如 [1, 2, 4, 8, 16]
        seq_len: 序列长度

    对每个 n_kv_heads，计算 KV Cache = 2 * n_layers * seq_len * (n_kv_heads * d_k) * 2 bytes
    与标准 MHA (n_kv_heads == n_heads) 比较，计算节省的百分比。
    """
    pass


if __name__ == "__main__":
    # TODO: LLaMA 2 70B 的例子
    pass
