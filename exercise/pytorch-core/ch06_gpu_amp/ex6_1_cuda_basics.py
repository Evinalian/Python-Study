"""
练习 6-1：CUDA 环境探测与 Tensor 转移

目标：
  编写脚本探测 CUDA 环境信息，并在 CPU/GPU 间转移 tensor，测量转移耗时。

要求：
  1. 检查 CUDA 是否可用，如果不可用则打印提示并以 CPU 模式运行
  2. 打印以下信息：
     - CUDA 版本
     - GPU 数量
     - 每张 GPU 的名称、总显存、计算能力
     - 当前 GPU 的已分配显存和缓存显存
  3. 创建一个大张量（如 shape=(10000, 10000)），测量：
     - CPU 上创建耗时
     - CPU → GPU 传输耗时（使用 .to("cuda")）
     - GPU → CPU 传输耗时
     - 使用 pin_memory 对比传输速度差异
  4. 测量 GPU 上的矩阵乘法 vs CPU 上的矩阵乘法速度差异

建议步骤：
  1. 导入 torch, time
  2. 实现 print_cuda_info() 函数
  3. 实现 benchmark_transfer() 函数（测量 CPU↔GPU 传输耗时）
  4. 实现 benchmark_matmul() 函数（测量矩阵乘法速度）
  5. 在主函数中依次调用，并输出格式化报告

提示：
  - 使用 torch.cuda.is_available() 检查 CUDA
  - 使用 torch.cuda.get_device_properties(i) 获取 GPU 属性
  - 使用 time.perf_counter() 测量精确耗时
  - 传输测试建议运行多次取平均（如 10 次）
  - 如果 CUDA 不可用，在 print_cuda_info 中打印 "CUDA 不可用" 即可
"""

# TODO: 导入 torch, time


# TODO: 实现 print_cuda_info() 函数


# TODO: 实现 benchmark_transfer(size=(10000, 10000), n_runs=10) 函数


# TODO: 实现 benchmark_matmul(size=(5000, 5000), n_runs=10) 函数


# TODO: 主函数：依次调用上述函数并打印结果
