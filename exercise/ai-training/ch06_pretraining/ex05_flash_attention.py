"""
练习 5（进阶）：集成 Flash Attention
====================================

将 MiniGPT 中的标准 Attention 替换为 Flash Attention。

你需要完成:
1. 使用 PyTorch 2.0 的 scaled_dot_product_attention 替代手动实现的 Attention
2. 或安装并使用 flash-attn 库
3. 比较替换前后的:
   - 训练速度（tokens/sec），在 seq_len=512, 1024, 2048, 4096 下
   - 显存占用
   - 是否影响模型精度（loss 是否有差异）
4. 分析 Flash Attention 的加速原理

使用 PyTorch 2.0 SDPA:
    output = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    这会自动选择最快的实现: Flash Attention -> Memory Efficient Attention -> 朴素实现

使用 flash-attn 库:
    pip install flash-attn --no-build-isolation
    from flash_attn import flash_attn_func
    output = flash_attn_func(Q, K, V, causal=True)

思考题:
- Flash Attention 为什么在长序列上加速更明显？
- 为什么 Flash Attention 节省显存？（提示: S 矩阵）
- PyTorch 的 SDPA 和原版 flash-attn 库有什么不同？
"""

import torch
import torch.nn.functional as F
import time


# TODO: 标准 Attention（用于对比）
class StandardAttention(torch.nn.Module):
    """TODO: 复制教程中的标准 Attention 实现"""
    pass


# TODO: Flash Attention 包装
class FlashAttention(torch.nn.Module):
    """
    TODO: 使用 Pytorch 2.0 SDPA 实现的 Attention。

    与标准 Attention 相同的前后处理（投影、分头、合并），
    但核心计算使用 F.scaled_dot_product_attention。
    """
    pass


# TODO: 性能对比
def benchmark_attention(attn_module, batch_size, seq_len, d_model, n_heads, num_runs=100):
    """
    TODO: 测试 Attention 模块的性能。

    测试在不同 seq_len 下的:
    - 前向传播时间
    - 反向传播时间（含梯度 = 实际训练场景）
    - 峰值显存占用

    返回:
        timing: dict，{seq_len: {'forward': ms, 'backward': ms, 'memory': MB}}
    """
    pass


# TODO: 精度对比
def compare_precision():
    """
    TODO: 验证 Flash Attention 和标准 Attention 的输出是否一致。

    使用相同的输入，计算两种实现的输出差异。
    期望差异在 1e-3 量级以内（由于浮点计算顺序不同）。
    """
    pass


if __name__ == "__main__":
    # TODO: 运行 benchmark
    seq_lengths = [512, 1024, 2048, 4096]
    for seq_len in seq_lengths:
        pass
