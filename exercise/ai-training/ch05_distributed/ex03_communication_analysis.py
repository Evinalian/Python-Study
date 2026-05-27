"""
练习 3：通信量分析
==================

分析模型在 DDP 下的通信量（无需实际运行分布式训练）。

你需要完成:
1. 编写函数计算模型的总参数量和梯度通信量
2. 对于不同规模的 Transformer 配置，计算每次 AllReduce 需要传输的数据量
3. 根据 Ring AllReduce 的通信量公式: 2×(N-1)/N × 数据量，计算:
   - 不同 world_size (2, 4, 8, 16, 32, 64) 下的总通信量
   - 假设 NVLink 带宽 (600 GB/s) 和 InfiniBand 带宽 (200 GB/s)，
     估算通信耗时
4. 比较通信耗时和计算耗时，判断是否存在通信瓶颈

参考数据:
- A100 FP16 算力: 312 TFLOPS
- A100 NVLink: 600 GB/s
- InfiniBand HDR: 200 GB/s

思考题:
- 在什么条件下 DDP 的通信会成为瓶颈？
- 为什么大模型的 DDP 通信开销相对较小？
  （提示: 通信量 O(P)，计算量 O(P × seq_len)）
- 如何估算计算耗时 vs 通信耗时的比值？
"""

import math


# TODO: 模拟不同规模的 Transformer 配置
TRANSFORMER_CONFIGS = {
    'gpt2_small':   {'n_layers': 12, 'd_model': 768,   'n_heads': 12},
    'gpt2_medium':  {'n_layers': 24, 'd_model': 1024,  'n_heads': 16},
    'gpt2_large':   {'n_layers': 36, 'd_model': 1280,  'n_heads': 20},
    'gpt2_xl':      {'n_layers': 48, 'd_model': 1600,  'n_heads': 25},
    'llama_7b':     {'n_layers': 32, 'd_model': 4096,  'n_heads': 32},
    'llama_13b':    {'n_layers': 40, 'd_model': 5120,  'n_heads': 40},
    'llama_70b':    {'n_layers': 80, 'd_model': 8192,  'n_heads': 64},
}


# TODO: 计算参数量
def estimate_params(n_layers, d_model, vocab_size=32000):
    """
    TODO: 估算 Transformer 的参数量。

    使用公式: P ≈ n_layers × (4×d_model² + 3×d_model×d_ff)
    其中 d_ff ≈ 8/3 × d_model（SwiGLU）
    """
    pass


# TODO: 计算通信量
def compute_communication_volume(params_count, world_size, dtype_bytes=2):
    """
    TODO: 计算一次 AllReduce 的通信量。

    参数:
        params_count: 参数量
        world_size: GPU 数量
        dtype_bytes: 每个参数的字节数（FP16=2, FP32=4）

    返回:
        total_bytes: 总通信字节数（Ring AllReduce 公式）
        2 × (N-1)/N × params_count × dtype_bytes
    """
    pass


# TODO: 估算通信耗时
def estimate_communication_time(params_count, world_size, bandwidth_gbps):
    """
    TODO: 估算 AllReduce 通信耗时（秒）。

    bandwidth_gbps: NVLink (600) 或 InfiniBand (200)
    """
    pass


# TODO: 对比计算耗时
def estimate_compute_time(params_count, seq_len, batch_size, tflops=312):
    """
    TODO: 估算一次前向+反向传播的计算耗时。

    近似公式: FLOPs_per_step ≈ 6 × params × seq_len × batch_size
    """
    pass


if __name__ == "__main__":
    for name, config in TRANSFORMER_CONFIGS.items():
        params = estimate_params(**config)
        print(f"{name}: {params/1e6:.0f}M 参数")

        for ws in [1, 2, 4, 8]:
            comm_bytes = compute_communication_volume(params, ws)
            comm_time = estimate_communication_time(params, ws, bandwidth_gbps=600)
            print(f"  world_size={ws}: comm={comm_bytes/1e9:.2f} GB, time={comm_time*1000:.2f} ms")
