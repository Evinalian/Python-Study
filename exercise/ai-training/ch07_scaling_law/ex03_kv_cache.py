"""
练习 3：实现 KV Cache 推理加速
==============================

为 MiniGPT 的 generate 方法添加 KV Cache 支持。

你需要完成:
1. 修改 CausalSelfAttention，使其支持 KV Cache:
   - forward 方法接受可选的 past_k, past_v 参数
   - 如果提供了 past_k/past_v，拼接后计算 Attention
   - 返回更新后的 KV Cache
2. 修改 TransformerBlock，透传 KV Cache
3. 修改 MiniGPT.generate，使用 KV Cache 加速生成:
   - 第一次前向: 无 cache，计算完整序列，保存 KV
   - 后续前向: 只输入最后一个 token，复用 KV Cache
4. 性能对比:
   - 比较有无 KV Cache 的生成速度
   - 生成 50、100、200、500 个 token，测量耗时
   - 绘制生成时间 vs 生成长度的关系

KV Cache 的显存计算:
    cache_size = 2 × n_layers × seq_len × n_heads × d_k × dtype_bytes
    对于 LLaMA 7B: 2 × 32 × 4096 × 32 × 128 × 2 = 约 2GB

思考题:
- 为什么 KV Cache 能加速推理？（提示: 避免 O(N²) 重计算）
- KV Cache 的显存占用和生成长度的关系是什么？
- GQA 如何减少 KV Cache 的大小？
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math


# TODO: 修改 CausalSelfAttention 以支持 KV Cache
class CausalSelfAttentionWithCache(nn.Module):
    """
    TODO: 支持 KV Cache 的 Self-Attention。

    forward(self, x, past_kv=None):
        参数:
            x: (batch, seq_len, d_model)
            past_kv: Optional[Tuple[Tensor, Tensor]]，过去的 K 和 V

        返回:
            output: (batch, seq_len, d_model)
            new_kv: (K, V)，更新后的 KV Cache
    """
    pass


# TODO: 修改 MiniGPT.generate 以使用 KV Cache
def generate_with_kv_cache(model, input_ids, max_new_tokens=100, temperature=1.0):
    """
    TODO: 使用 KV Cache 的生成函数。

    步骤:
    1. 首次前向: 对整个 prompt 做前向传播，保存每层的 KV Cache
    2. 后续: 只输入最后一个 token，拼接 KV Cache

    返回:
        generated_ids: (1, prompt_len + max_new_tokens)
    """
    pass


# TODO: 性能对比
def benchmark_generation(model, input_ids, max_tokens_list, use_cache):
    """
    TODO: 测试不同长度和有无 Cache 的生成速度。

    对每个 max_tokens:
    1. 记录生成开始时间
    2. 生成 max_tokens 个 token
    3. 记录耗时
    4. 计算 tokens/sec

    绘制 时间 vs 生成长度 的对比图。
    """
    pass


if __name__ == "__main__":
    # TODO: 加载模型，运行 benchmark
    pass
