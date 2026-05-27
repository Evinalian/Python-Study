"""
练习 2：Flash Attention 集成与性能分析
=======================================

将 MiniGPT 中的 Attention 替换为 PyTorch SDPA，并进行性能对比。

你需要完成:
1. 修改 MiniGPT 的 CausalSelfAttention，使用 F.scaled_dot_product_attention
2. 测试不同 seq_len (512, 1024, 2048, 4096) 下的:
   - 前向传播时间
   - 反向传播时间
   - GPU 显存占用峰值
   - 吞吐量 (tokens/sec)
3. 测量 SDPA 选择的实际后端（Flash Attention / Memory Efficient / 朴素实现）
4. 绘制速度对比图（标准 vs SDPA）

注意:
- F.scaled_dot_product_attention 需要 PyTorch >= 2.0
- 在支持的 GPU 上会自动选择 Flash Attention
- 可以用 torch.backends.cuda.sdp_kernel 控制后端选择
- 测量显存可以用 torch.cuda.max_memory_allocated()

思考题:
- 在 seq_len=512 时 Flash Attention 有加速吗？为什么？
- 在 seq_len=4096 时加速比是多少？
- SDPA 的 is_causal=True 参数有什么作用？
"""

import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt


# TODO: 标准 Attention（用于对照）


# TODO: SDPA Attention
class SDPAAttention(torch.nn.Module):
    """
    TODO: 使用 F.scaled_dot_product_attention 的 Attention。

    forward 的核心调用:
        output = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
        )
    """
    pass


# TODO: 性能测试
def benchmark(model_fn, batch_size, seq_len, num_runs=50):
    """
    TODO: 测试模型的前向和反向传播性能。

    返回:
        {
            'forward_time_ms': float,
            'backward_time_ms': float,
            'total_time_ms': float,
            'max_memory_mb': float,
            'tokens_per_sec': float,
        }
    """
    pass


# TODO: 确认实际使用的后端
def check_sdpa_backend():
    """
    TODO: 检查 SDPA 实际选择了哪个后端。

    方法:
    1. 设置 torch.backends.cuda.sdp_kernel 来强制特定后端
    2. 或使用 torch._C._autograd._get_sdpa_backend_info()
    """
    pass


# TODO: 主对比实验
def run_benchmarks():
    """
    TODO:
    1. 对于每个 seq_len in [512, 1024, 2048, 4096]:
       - 测试标准 Attention 的性能
       - 测试 SDPA Attention 的性能
    2. 打印对比表格
    3. 绘制加速比 vs seq_len 的图
    """
    pass


if __name__ == "__main__":
    run_benchmarks()
