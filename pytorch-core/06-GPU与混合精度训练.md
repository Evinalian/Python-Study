# 第 6 章：GPU 与混合精度训练

## 学习目标

- 理解 CPU 与 GPU 的架构差异，以及为什么 GPU 适合深度学习
- 掌握 PyTorch 中 tensor 和模型在 CPU/GPU 之间的转移方法
- 理解 CUDA 显存管理，能用 `torch.cuda` API 监控显存使用
- 掌握 DataParallel (DP) 和 DistributedDataParallel (DDP) 的使用方式及差异
- 理解 FP32 / FP16 / BF16 精度格式及其对训练的影响
- 掌握混合精度训练 (AMP) 的完整流程：`autocast` + `GradScaler`
- 能够用 `torchrun` 启动多卡 DDP 训练

## 前置知识

- PyTorch 训练循环（前向、反向、优化器更新）
- Python 多进程概念（`multiprocessing`、进程间通信）
- 基本的计算机体系结构概念（CPU、内存、总线）

---

## 6.1 CUDA 基础概念

### 6.1.1 CPU vs GPU 的架构差异

理解 CPU 和 GPU 的设计哲学差异，是掌握 GPU 编程的第一步。

**CPU (Central Processing Unit)：**

```
┌─────────────────────────────────────────────────────┐
│                    CPU 芯片                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Core 0  │ │ Core 1  │ │ Core 2  │ │ Core 3  │   │
│  │  ┌────┐ │ │  ┌────┐ │ │  ┌────┐ │ │  ┌────┐ │   │
│  │  │L1  │ │ │  │L1  │ │ │  │L1  │ │ │  │L1  │ │   │
│  │  └────┘ │ │  └────┘ │ │  └────┘ │ │  └────┘ │   │
│  │  ┌────┐ │ │  ┌────┐ │ │  ┌────┐ │ │  ┌────┐ │   │
│  │  │L2  │ │ │  │L2  │ │ │  │L2  │ │ │  │L2  │ │   │
│  │  └────┘ │ │  └────┘ │ │  └────┘ │ │  └────┘ │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│  ┌───────────────────────────────────────────────┐  │
│  │                  L3 Cache (共享)               │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

- **核心数少**（4-64 核），但每个核心**非常强大**
- **主频高**（3-5 GHz），适合复杂的**串行**逻辑
- **分支预测、乱序执行**等复杂优化
- **大缓存**（L1/L2/L3），减少内存访问延迟
- 设计目标：**最小化单个任务的延迟（latency）**

**GPU (Graphics Processing Unit)：**

```
┌─────────────────────────────────────────────────────┐
│                    GPU 芯片                          │
│  ┌───────────────────────────────────────────────┐  │
│  │  SM 0       SM 1       SM 2       ...  SM N   │  │
│  │  ┌───────┐ ┌───────┐ ┌───────┐     ┌───────┐  │  │
│  │  │CUDA核 │ │CUDA核 │ │CUDA核 │     │CUDA核 │  │  │
│  │  │ x 128 │ │ x 128 │ │ x 128 │     │ x 128 │  │  │
│  │  └───────┘ └───────┘ └───────┘     └───────┘  │  │
│  │  ┌───────┐ ┌───────┐ ┌───────┐     ┌───────┐  │  │
│  │  │共享内存│ │共享内存│ │共享内存│     │共享内存│  │  │
│  │  └───────┘ └───────┘ └───────┘     └───────┘  │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │                L2 Cache (共享)                  │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │              VRAM / 显存 (HBM/GDDR)             │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

- **核心数极多**（数千到上万），但每个核心**相对简单**
- **主频较低**（1-2 GHz），但总吞吐量巨大
- **SIMT 架构**（Single Instruction, Multiple Threads）：一组线程（32 个，称为 warp）同时执行相同指令
- **显存带宽极高**（HBM2e 可达 1.5+ TB/s，相比之下 DDR5 约 50 GB/s）
- 设计目标：**最大化吞吐量（throughput）**

**数值对比：**

| 指标 | Intel i9-13900K (CPU) | NVIDIA RTX 4090 (GPU) |
|------|----------------------|----------------------|
| 核心数 | 24 (8P+16E) | 16384 CUDA Cores |
| 主频 | 3.0-5.8 GHz | 2.23-2.52 GHz |
| 内存带宽 | ~90 GB/s (DDR5) | ~1008 GB/s (GDDR6X) |
| FP32 理论算力 | ~2 TFLOPS | ~82.6 TFLOPS |
| 显存/内存 | 系统内存 32-128 GB | 24 GB GDDR6X |
| 功耗 | 125-253W | 450W |

**结论：** GPU 用**海量的简单核心 + 超高内存带宽**换来了对**高度并行化计算**（矩阵乘法、卷积）的碾压级优势。深度学习恰恰就是这种高度并行化的计算。

### 6.1.2 什么操作适合 GPU

**适合 GPU 的操作（计算密集 + 高度并行）：**

- 矩阵乘法（`torch.matmul` / `@`）
- 卷积（`torch.nn.functional.conv2d`）
- 大批量 element-wise 操作（激活函数、加法等）
- 大张量的 reduction（sum、mean、max）
- Softmax 等需要跨维度通信的操作

**不适合 GPU 的操作（串行 + 小数据量）：**

- 小张量运算（单个标量加法，kernel 启动开销比计算还大）
- 复杂控制流（if/else 导致 warp divergence）
- 顺序依赖强的操作（下一个结果依赖于上一个）
- Python 原生的列表/字典操作（在 CPU 上执行）

### 6.1.3 PCIe 数据传输瓶颈

CPU 和 GPU 之间通过 PCIe 总线连接。这是深度学习训练中的**关键瓶颈**。

```
┌──────┐    PCIe 4.0 x16     ┌──────┐
│ CPU  │◄──────────────────►│ GPU  │
│ RAM  │   ~32 GB/s 单向     │ VRAM │
└──────┘                     └──────┘
```

**关键事实：**

- PCIe 4.0 x16 单向带宽约 32 GB/s，双向约 64 GB/s
- GPU 内部显存带宽约 1008 GB/s（RTX 4090）
- **差距约 30 倍**

这意味着：**每次 CPU-GPU 数据传输都是昂贵的。** 训练中应该：
- 数据一次性加载到 GPU，避免频繁的 `tensor.cpu()` 操作
- 使用 `pin_memory=True` + `non_blocking=True` 来异步传输数据

---

## 6.2 PyTorch CUDA 编程

### 6.2.1 检查 CUDA 环境

```python
import torch

# ---- 1. 检查 CUDA 是否可用 ----
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

# ---- 2. GPU 数量 ----
print(f"GPU 数量: {torch.cuda.device_count()}")

# ---- 3. 当前设备 ----
if torch.cuda.is_available():
    print(f"当前设备索引: {torch.cuda.current_device()}")
    print(f"当前设备名称: {torch.cuda.get_device_name(0)}")

    # ---- 4. 显存信息 ----
    print(f"已分配显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"已缓存显存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    # 注意：memory_allocated 是你的 tensor 实际占用的
    # memory_reserved 是 PyTorch 的缓存池大小（包括已释放但未归还给 OS 的）

    # ---- 5. GPU 属性 ----
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU 详细属性:")
    print(f"  名称: {props.name}")
    print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
    print(f"  多处理器数(SM): {props.multi_processor_count}")
    print(f"  计算能力: {props.major}.{props.minor}")
    print(f"  Warp 大小: 32")
    print(f"  每个 SM 最大线程数: {props.max_threads_per_multi_processor}")

# ---- 6. 列出所有 GPU ----
for i in range(torch.cuda.device_count()):
    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
    props = torch.cuda.get_device_properties(i)
    print(f"  显存: {props.total_memory / 1024**3:.2f} GB")
```

### 6.2.2 device 的概念

在 PyTorch 中，`device` 是一个表示计算设备（CPU 或某个 GPU）的对象。

```python
# 创建 device 对象的几种方式
device_cpu = torch.device("cpu")
device_gpu0 = torch.device("cuda")       # 等价于 cuda:0，默认 GPU
device_gpu1 = torch.device("cuda:1")     # 第 2 块 GPU
device_gpu2 = torch.device("cuda", 2)    # 也等价于 cuda:2

# 最常用的写法：自动选择可用设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
```

### 6.2.3 Tensor 在 CPU 和 GPU 间转移

```python
import torch

# ---- 在创建时指定 device ----
x_cpu = torch.randn(3, 4)                     # 默认在 CPU
x_gpu = torch.randn(3, 4, device="cuda")      # 直接在 GPU 创建
x_dev = torch.randn(3, 4, device=device)      # 使用变量指定的设备

print(f"x_cpu device: {x_cpu.device}")        # cpu
print(f"x_gpu device: {x_gpu.device}")        # cuda:0

# ---- 转移已存在的 Tensor (方式一：.to 方法，推荐) ----
x = torch.randn(3, 4)                         # CPU
x_gpu = x.to("cuda")                          # 转移到 GPU（深拷贝）
x_gpu2 = x.to(device)                         # 使用 device 变量
x_back = x_gpu.to("cpu")                      # 转回 CPU

# .to 方法的完整签名：
# tensor.to(device, dtype=None, non_blocking=False, copy=False)
# dtype: 同时转换数据类型（如 .to(torch.float16)）
# non_blocking: 异步传输（仅对 pinned memory 有效）
# copy: 如果已在目标设备，是否强制拷贝

# ---- 转移方式二：.cuda() / .cpu() ----
x_gpu = x.cuda()                              # 转移到默认 GPU
x_gpu = x.cuda(0)                             # 转移到 GPU 0
x_gpu = x.cuda(1)                             # 转移到 GPU 1
x_cpu = x_gpu.cpu()                           # 转回 CPU

# ---- 检查张量所在的设备 ----
def print_device(tensor, name):
    print(f"{name}: shape={tensor.shape}, device={tensor.device}, "
          f"dtype={tensor.dtype}")

print_device(x, "x")
print_device(x_gpu, "x_gpu")
```

### 6.2.4 模型转移到 GPU

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)

# ---- 方式 1：.to(device) ----
model = MyModel()
model = model.to(device)  # 模型的所有参数和 buffer 都转移到指定 device

# ---- 方式 2：.cuda() ----
model = MyModel().cuda()

# ---- 验证 ----
print(f"fc.weight device: {model.fc.weight.device}")  # 应为 cuda:0

# ---- 重要原则：模型和数据必须在同一设备 ----
x = torch.randn(5, 100)  # CPU 上的数据
# output = model(x)        # RuntimeError! 模型在 GPU，数据在 CPU

x = x.to(device)          # 数据也转移到 GPU
output = model(x)          # 正确！模型和数据都在 GPU
print(f"output device: {output.device}")  # cuda:0
```

### 6.2.5 常见错误：CPU 和 GPU Tensor 混用

这是 GPU 编程新手的头号问题。

```python
# ---- 错误场景 1：模型和数据不在同一设备 ----
model = MyModel().cuda()
x = torch.randn(5, 100)  # 忘记 to(device)
# y = model(x)  # RuntimeError: Expected all tensors to be on the same device

# ---- 错误场景 2：两个 Tensor 在不同设备 ----
a = torch.randn(3, 4).cuda(0)
b = torch.randn(3, 4).cuda(1)
# c = a + b  # RuntimeError: Expected all tensors to be on the same device

# ---- 错误场景 3：loss 函数在 CPU ----
model = MyModel().cuda()
x = torch.randn(5, 100).cuda()
target = torch.randn(5, 10).cuda()
criterion = nn.MSELoss()  # 默认在 CPU，但 loss 函数通常不包含可学习参数
# 实际上 nn.MSELoss() 在 CPU 定义是可以的，只要输入都在 GPU 上
# 但为了安全，建议：
criterion = nn.MSELoss().to(device)

# ---- 错误场景 4：自定义指标计算产生 CPU Tensor ----
model = MyModel().cuda()
x = torch.randn(5, 100).cuda()
output = model(x)
# 如果自定义准确率计算中创建了新 tensor，默认在 CPU
accuracy_tensor = torch.tensor([0.5])  # CPU!
# 这个 tensor 如果参与后续 GPU 上的计算，会报错

# ---- 调试工具：快速定位设备不一致 ----
def check_device(model, *tensors):
    """检查模型和所有输入 tensor 是否在同一 device"""
    model_device = next(model.parameters()).device
    for i, t in enumerate(tensors):
        if t.device != model_device:
            print(f"[WARNING] tensor[{i}] is on {t.device}, "
                  f"but model is on {model_device}")
    print(f"All devices consistent: {all(t.device == model_device for t in tensors)}")
```

### 6.2.6 显存管理

```python
import torch

# ---- 查看显存使用 ----
if torch.cuda.is_available():
    # 已分配显存（tensor 实际占用）
    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"已分配: {allocated:.4f} GB")

    # 已缓存显存（PyTorch 缓存池，包括已释放但未归还 OS 的）
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"已缓存: {reserved:.4f} GB")

    # 最大分配量（峰值记录）
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    print(f"峰值分配: {max_allocated:.4f} GB")

    # 重置峰值记录
    torch.cuda.reset_peak_memory_stats()

# ---- 清空缓存 ----
# 注意：这不能释放被 tensor 引用的显存，只释放 PyTorch 缓存池中未使用的
torch.cuda.empty_cache()

# ---- 手动释放显存 ----
x = torch.randn(1000, 1000, 1000, device="cuda")
print(f"分配大张量后: {torch.cuda.memory_allocated() / 1024**3:.4f} GB")
del x  # 引用计数减 1
# 如果这是最后一个引用，PyTorch 会释放显存回缓存池
# 但不会立即归还给 OS（需要 empty_cache）
torch.cuda.empty_cache()
print(f"del + empty_cache 后: {torch.cuda.memory_allocated() / 1024**3:.4f} GB")

# ---- 内存快照（调试 OOM 的强大工具） ----
# torch.cuda.memory._record_memory_history()
# ... 你的代码 ...
# torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
# 然后在浏览器中用 https://pytorch.org/memory_viz 可视化
```

### 6.2.7 Pinned Memory 与异步传输

```python
# ---- 普通 DataLoader ----
# 每次从 CPU 内存分页传输到 GPU，串行等待
from torch.utils.data import DataLoader, TensorDataset

data = torch.randn(1000, 100)
labels = torch.randint(0, 10, (1000,))
dataset = TensorDataset(data, labels)

# 标准 DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---- 优化：pin_memory + non_blocking ----
loader_pinned = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True,    # 将数据锁定在物理内存页，加速 CPU->GPU 传输
    num_workers=2,       # 多进程预加载数据
)

# 训练循环中使用 non_blocking
for inputs, labels in loader_pinned:
    # non_blocking=True：异步传输，GPU 拷贝的同时 CPU 继续执行
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    # ... 后续训练代码（GPU 会在需要 inputs/labels 时自动等待传输完成）

# ---- pin_memory 的工作原理 ----
# 普通内存页可以被 OS 换出到磁盘（page fault），DMA 传输时需要先锁定
# pin_memory 提前锁定了内存页，DMA 可以直接传输，无需 CPU 参与
# 耗时从 ~10ms 降低到 ~1ms（取决于数据量）
```

---

## 6.3 多 GPU 训练

### 6.3.1 为什么需要多 GPU

- **显存不足**：单个 GPU 放不下模型 + batch + 中间激活
- **加速训练**：多卡并行，吞吐量线性增长（理想情况）
- **更大的有效 batch size**：每一卡算一部分，梯度汇总

### 6.3.2 DataParallel (DP) —— 简单但不推荐

DP 是 PyTorch 最早的多卡方案，只需一行代码，但性能有严重问题。

**工作原理：**

```
Step 1: GPU 0 加载完整模型（主卡）
Step 2: 将 batch 分成 N 份，scatter 到各卡
Step 3: 每张卡各自前向传播
Step 4: 所有卡的输出 gather 回 GPU 0
Step 5: GPU 0 计算 loss
Step 6: loss scatter 回各卡做反向传播
Step 7: 所有梯度 gather 到 GPU 0 做参数更新
Step 8: GPU 0 广播更新后的参数到各卡

问题：GPU 0 负载过重 + 大量 gather/scatter + Python GIL
```

**代码示例：**

```python
import torch.nn as nn

model = MyModel()
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 张 GPU (DataParallel)")
    model = nn.DataParallel(model)
model = model.to(device)  # device 应为 cuda:0（主卡）

# 训练代码完全不变（DP 对用户透明）
# 缺点：
# 1. GPU 0 负载远大于其他卡
# 2. Python GIL 导致线程间切换开销
# 3. 每次 gather/scatter 有额外通信
# 4. 不支持多机
```

**DP 问题详解：**

| 问题 | 原因 | 影响 |
|------|------|------|
| GPU 0 过载 | 所有 scatter/gather 和参数更新都在 GPU 0 | GPU 0 利用率 > 90%，其他卡可能只有 60% |
| GIL 竞争 | DP 用 Python 多线程，受 GIL 限制 | 多线程退化为串行 |
| 负载不均 | 最后一卡分到的数据可能更少 | 各卡等待时间不一致 |
| 无法跨机 | DP 假设所有 GPU 在同一机器 | 不支持分布式 |

### 6.3.3 DistributedDataParallel (DDP) —— 推荐方案

DDP 是当前 PyTorch 多卡训练的标准方案。核心思想：**每个进程独立运行，各持一张卡**。

**工作原理：**

```
┌──────────────────────────────────────────────────┐
│                   训练流程                         │
│                                                    │
│  GPU 0 (进程0)   GPU 1 (进程1)   GPU 2 (进程2)    │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐     │
│  │ 完整模型  │    │ 完整模型  │    │ 完整模型  │     │
│  │ batch_0  │    │ batch_1  │    │ batch_2  │     │
│  │ forward  │    │ forward  │    │ forward  │     │
│  │ backward │    │ backward │    │ backward │     │
│  │          │    │          │    │          │     │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘     │
│       │               │               │           │
│       └───────────────┼───────────────┘           │
│                       │                           │
│              AllReduce 梯度（NCCL）               │
│        每个进程得到相同的平均梯度                    │
│                       │                           │
│       ┌───────────────┼───────────────┐           │
│       │               │               │           │
│  ┌────▼─────┐    ┌────▼─────┐    ┌────▼─────┐     │
│  │optim.step│    │optim.step│    │optim.step│     │
│  └──────────┘    └──────────┘    └──────────┘     │
│                                                    │
│  关键：每个进程的模型初始状态相同，梯度也通过       │
│  AllReduce 求平均，所以参数更新后仍然一致           │
└──────────────────────────────────────────────────┘
```

**完整 DDP 代码：**

```python
"""
DDP 训练脚本: ddp_train.py
启动方式: torchrun --nproc_per_node=2 ddp_train.py
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

# ============================================================
# 1. 初始化分布式环境
# ============================================================
def setup_ddp():
    """初始化 DDP 进程组"""
    # torchrun 会自动设置以下环境变量：
    #   LOCAL_RANK: 当前进程在本机上的 GPU 索引
    #   WORLD_SIZE: 总进程数
    #   RANK: 当前进程的全局编号
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # 初始化进程组
    # backend: "nccl" (NVIDIA GPU), "gloo" (CPU/通用), "mpi"
    dist.init_process_group(
        backend="nccl",
        init_method="env://",  # 从环境变量读取配置（torchrun 设置）
    )

    # 设置当前进程使用的 GPU
    torch.cuda.set_device(local_rank)

    return local_rank, world_size, rank

# ============================================================
# 2. 清理
# ============================================================
def cleanup_ddp():
    dist.destroy_process_group()

# ============================================================
# 3. 定义模型
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ============================================================
# 4. 分布式训练主循环
# ============================================================
def main():
    local_rank, world_size, rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}/{world_size}] 初始化完成，使用 device={device}")

    # ---- 准备数据（使用 DistributedSampler） ----
    # 每个进程只加载自己的那部分数据，不重复
    X = torch.randn(1000, 1, 28, 28)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)

    # DistributedSampler: 确保每个 epoch 每个样本只被一个进程处理
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,   # 总进程数
        rank=rank,                  # 当前进程编号
        shuffle=True,               # 是否打乱
        drop_last=True,             # 丢弃不完整的最后一个 batch
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,           # 使用 sampler 代替 shuffle
        num_workers=2,
        pin_memory=True,
    )

    # ---- 创建模型 ----
    model = SimpleCNN().to(device)

    # ---- 用 DDP 包装模型 ----
    # DDP 会在反向传播时自动同步梯度
    model = DDP(
        model,
        device_ids=[local_rank],    # 当前进程使用的 GPU
        output_device=local_rank,   # 输出所在的 GPU
    )
    # 注意：DDP 包装后，原来的 model 变成了 model.module
    # 访问原始模型用 model.module（保存 state_dict 时需要用）

    # ---- 损失函数和优化器 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---- 训练循环 ----
    NUM_EPOCHS = 10
    for epoch in range(NUM_EPOCHS):
        # 关键：每个 epoch 设置 sampler 的 epoch
        # 这确保了每个 epoch 数据的随机顺序不同
        sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # DDP 在 backward() 时自动做 AllReduce 梯度同步
            optimizer.step()

            total_loss += loss.item()

            # 只在 rank 0 打印，避免重复输出
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
                      f"Batch {batch_idx}/{len(dataloader)} "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)

        # 所有进程的 loss 求平均（因为各进程处理的样本数可能不同）
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_tensor /= world_size

        if rank == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
                  f"Avg Loss: {avg_loss_tensor.item():.4f}")

    # ---- 保存模型（只在 rank 0 保存） ----
    if rank == 0:
        # 注意：保存的是 model.module.state_dict()，不是 model.state_dict()
        # 因为 model 被 DDP 包装了，model.module 才是原始模型
        torch.save(model.module.state_dict(), "ddp_model.pth")
        print("[Rank 0] 模型已保存")

    cleanup_ddp()

if __name__ == "__main__":
    main()
```

**启动方式：**

```bash
# 单机 2 卡
torchrun --nproc_per_node=2 ddp_train.py

# 单机 4 卡
torchrun --nproc_per_node=4 ddp_train.py

# 指定 master 地址和端口（多机场景）
torchrun --nproc_per_node=2 \
         --nnodes=2 \
         --node_rank=0 \
         --master_addr="192.168.1.100" \
         --master_port=29500 \
         ddp_train.py
```

### 6.3.4 DP vs DDP 完整对比

| 维度 | DataParallel (DP) | DistributedDataParallel (DDP) |
|------|-------------------|-------------------------------|
| **实现方式** | 单进程多线程 | 多进程（每卡独立进程） |
| **Python GIL** | 受 GIL 影响（多线程） | 不受影响（多进程） |
| **通信方式** | gather/scatter/broadcast（GPU 0 是瓶颈） | AllReduce（Ring AllReduce，各卡平等） |
| **通信后端** | 隐式（PyTorch 内部） | NCCL（NVIDIA 专用，极快） |
| **负载均衡** | GPU 0 过载 | 各卡负载均匀 |
| **梯度同步时机** | 所有梯度 gather 到 GPU 0 后再广播 | backward 时自动 AllReduce（重叠计算与通信）|
| **代码复杂度** | 一行代码 (`nn.DataParallel`) | 需要进程初始化 + DistributedSampler |
| **多机支持** | 不支持 | 支持 |
| **性能（以 ResNet-50 为例）** | 1.5x (2卡) | 1.9x (2卡)，几乎线性 |
| **推荐程度** | 不推荐（仅用于快速实验） | **强烈推荐** |

---

## 6.4 混合精度训练 (AMP)

### 6.4.1 精度格式概览

深度学习中的浮点数格式：

| 格式 | 总位数 | 符号位 | 指数位 | 尾数位 | 数值范围 | 最小精度 |
|------|--------|--------|--------|--------|----------|----------|
| FP32 | 32 | 1 | 8 | 23 | ~1.18e-38 到 ~3.4e38 | ~1.19e-7 |
| FP16 | 16 | 1 | 5 | 10 | ~6.10e-5 到 65504 | ~9.77e-4 |
| BF16 | 16 | 1 | 8 | 7 | ~1.18e-38 到 ~3.39e38 | ~7.81e-3 |
| TF32 | 19 | 1 | 8 | 10 | ~1.18e-38 到 ~3.39e38 | ~9.77e-4 |

**关键区别：**

```
FP32: [S][EEEEEEEE][MMMMMMMMMMMMMMMMMMMMMMM]   ← 基准
FP16: [S][EEEEE][MMMMMMMMMM]                    ← 范围小，精度低
BF16: [S][EEEEEEEE][MMMMMMM]                    ← 范围同 FP32，精度更低
                                    ↑
                         指数位同 FP32，所以动态范围一致
```

- **FP16 的问题**：指数位只有 5 位，最大值仅 65504。训练中梯度值很容易超过这个范围，变成 `inf`/`NaN`。
- **BF16 的优势**：指数位同 FP32（8 位），不会溢出。但尾数位少（7 位），精度损失比 FP16 大。
- **TF32**：NVIDIA Ampere 及以后架构支持的格式。内部计算用 TF32（19 位），输入输出仍为 FP32。精度与 FP32 几乎相同，速度接近 FP16。

### 6.4.2 为什么需要混合精度而不是纯 FP16

纯 FP16 训练的问题：

```python
# 示例 1：梯度下溢
x = torch.tensor([0.0001], dtype=torch.float16)
print(x)  # tensor([9.9945e-05], dtype=torch.float16) -- 有舍入误差
# 更小的值可能直接变为 0！

# 示例 2：梯度上溢
y = torch.tensor([100000.0], dtype=torch.float16)
print(y)  # tensor([inf], dtype=torch.float16) -- 超过 FP16 最大值！
```

**混合精度 = FP16 计算（快） + FP32 权重副本（稳）**

- **前向/反向计算**：使用 FP16，速度快，显存减半
- **权重存储**：保留 FP32 副本，参数更新在 FP32 精度下进行
- **Loss Scaling**：将 loss 放大（乘以一个 scaling factor），使小梯度值不会在 FP16 下变为 0；反向传播后再缩小

### 6.4.3 AMP 核心组件

PyTorch 的 `torch.cuda.amp` 模块提供了两个核心工具：

**1. `autocast` 上下文管理器**

```python
# autocast 自动选择每个操作的最优精度
# 大部分操作用 FP16，精度敏感的操作自动用 FP32
with torch.cuda.amp.autocast():
    output = model(input)     # 自动用 FP16 进行卷积/矩阵乘法
    loss = criterion(output, target)  # loss 也自动选精度
```

哪些操作在 autocast 下保持 FP32？
- 某些 reduction 操作（sum、mean 等）
- Softmax、LayerNorm 等数值敏感操作
- 涉及大数值范围的操作

**2. `GradScaler` 梯度缩放器**

```python
scaler = torch.cuda.amp.GradScaler()

# 训练循环中：
scaler.scale(loss).backward()   # 将 loss 放大后反向传播
scaler.step(optimizer)           # 将梯度缩小后更新参数
scaler.update()                  # 动态调整缩放因子
```

### 6.4.4 完整的 AMP 训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---- 准备数据 ----
X = torch.randn(10000, 100).cuda()
y = torch.randint(0, 3, (10000,)).cuda()
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ---- 模型 ----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )
    def forward(self, x):
        return self.net(x)

model = MLP().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---- AMP：创建 GradScaler ----
scaler = torch.cuda.amp.GradScaler()
# GradScaler 的默认初始缩放因子是 2^16 = 65536
# 如果连续 N 次没有 inf/NaN，缩放因子会翻倍（最多到 2^24）
# 如果出现 inf/NaN，缩放因子会减半

# ---- 训练循环 ----
NUM_EPOCHS = 5
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for inputs, labels in loader:
        optimizer.zero_grad()

        # 步骤 1：autocast 前向传播
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # 步骤 2：缩放 loss 并反向传播
        # scaler.scale(loss) 返回 loss * scale_factor
        # .backward() 产生的梯度是缩放后的
        scaler.scale(loss).backward()

        # 步骤 3：将梯度缩放回来，然后更新参数
        # scaler.step(optimizer) 做了两件事：
        #   a. unscale_ gradients（梯度除以 scale_factor）
        #   b. optimizer.step()（正常更新参数）
        # 如果发现梯度中有 inf/NaN，跳过 optimizer.step()
        scaler.step(optimizer)

        # 步骤 4：更新缩放因子
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    current_scale = scaler.get_scale()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | "
          f"Scale: {current_scale:.0f}")

print("训练完成！")
```

### 6.4.5 AMP + DDP 组合代码

```python
"""
AMP + DDP 完整训练脚本
启动: torchrun --nproc_per_node=2 amp_ddp_train.py
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

def setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()

    # ---- 数据 ----
    X = torch.randn(5000, 100)
    y = torch.randint(0, 3, (5000,))
    dataset = TensorDataset(X, y)
    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(dataset, batch_size=64, sampler=sampler,
                        num_workers=2, pin_memory=True)

    # ---- 模型 ----
    model = MLP().to(device)
    model = DDP(model, device_ids=[local_rank])

    # ---- 损失函数、优化器 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---- AMP: GradScaler ----
    # DDP 下 GradScaler 的使用与单卡完全相同
    scaler = torch.cuda.amp.GradScaler()

    # ---- 训练 ----
    for epoch in range(5):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0

        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # autocast + DDP: DDP 的梯度同步在 backward 中自动完成
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if rank == 0:
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} "
                  f"| Scale: {scaler.get_scale():.0f}")

    if rank == 0:
        torch.save(model.module.state_dict(), "amp_ddp_model.pth")

    cleanup()

if __name__ == "__main__":
    main()
```

### 6.4.6 AMP 显存与速度收益

**典型数据（以 ResNet-50 在 RTX 3090 上的训练为例）：**

| 配置 | Batch Size | 显存占用 | 训练速度 (img/s) | 精度 (Top-1) |
|------|-----------|---------|-----------------|-------------|
| FP32 | 128 | 18.2 GB | 450 | 76.1% |
| AMP (FP16) | 128 | 10.5 GB (-42%) | 720 (+60%) | 76.0% |
| AMP (FP16) | 256 | 20.8 GB | 820 (+82%) | 76.1% |

**原因分析：**

- 显存减半：权重、激活值、梯度在 FP16 下各占一半空间
- 速度提升：NVIDIA GPU 的 Tensor Core 专门优化了 FP16 矩阵乘法（FP16 吞吐量是 FP32 的 8 倍以上，Ampere 架构）
- 精度无损：Loss scaling 防止了梯度下溢，FP32 权重副本保证了参数更新精度

**当 AMP 训练出现 NaN 时：**

```python
# 问题排查步骤：
# 1. 检查 GradScaler 的 scale 是否持续下降
print(f"Current scale: {scaler.get_scale()}")  # 如果很小（< 100），说明频繁出现溢出

# 2. 降低学习率（AMP 对学习率更敏感）
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 从 0.001 降到 0.0001

# 3. 使用 BF16 代替 FP16（Ampere+ GPU 支持）
with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # BF16 范围更大，不易溢出
    ...

# 4. 检查数据中是否有 NaN/Inf
if torch.isnan(inputs).any() or torch.isinf(inputs).any():
    print("输入数据包含 NaN/Inf！")

# 5. 梯度裁剪（限制梯度范数）
scaler.unscale_(optimizer)  # 手动 unscale
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

---

## 6.5 基础练习

### 练习 6-1：CUDA 环境探测与 Tensor 转移

在 `exercise/pytorch-core/ch06_gpu_amp/ex6_1_cuda_basics.py` 中实现。

**题目描述：** 编写脚本探测 CUDA 环境信息（GPU 数量、名称、显存），并在 CPU/GPU 间转移 tensor，测量转移耗时。

### 练习 6-2：AMP 混合精度训练对比

在 `exercise/pytorch-core/ch06_gpu_amp/ex6_2_amp_comparison.py` 中实现。

**题目描述：** 训练同一个模型，分别使用 FP32 和 AMP (FP16) 两种模式，对比显存占用、训练速度和最终 loss。

### 练习 6-3：CPU/GPU Tensor 混用错误复现与修复

在 `exercise/pytorch-core/ch06_gpu_amp/ex6_3_device_mismatch.py` 中实现。

**题目描述：** 故意写出 4-5 种常见的 CPU/GPU Tensor 混用错误，然后用 `try/except` 捕获并打印错误信息，最后给出正确写法。

---

## 6.6 进阶练习

### 练习 6-4：DDP 多卡训练

在 `exercise/pytorch-core/ch06_gpu_amp/ex6_4_ddp_training.py` 中实现。

**题目描述：** 实现一个完整的 DDP 训练脚本，包含 `DistributedSampler`、梯度同步、只在 rank 0 保存模型和打印日志。

### 练习 6-5：AMP + DDP 组合

在 `exercise/pytorch-core/ch06_gpu_amp/ex6_5_amp_ddp.py` 中实现。

**题目描述：** 将 AMP 和 DDP 组合到同一个训练脚本中，实现完整的分布式混合精度训练。

---

## 6.7 常见错误

### 错误 1：CPU 和 GPU Tensor 直接运算

```python
# ---- 错误 ----
model = MyModel().cuda()
x = torch.randn(5, 100)  # CPU
y = model(x)  # RuntimeError: Expected all tensors to be on the same device

# ---- 正确 ----
x = torch.randn(5, 100).to(device)  # 或 .cuda()
y = model(x)
```

### 错误 2：loss 计算中引入 CPU Tensor

```python
# ---- 错误 ----
outputs = model(inputs)  # GPU
# 自定义 loss 计算时创建了 CPU tensor
weights = torch.tensor([1.0, 2.0, 3.0])  # CPU
loss = (outputs * weights).sum()  # RuntimeError!

# ---- 正确 ----
weights = torch.tensor([1.0, 2.0, 3.0], device=device)
loss = (outputs * weights).sum()
```

### 错误 3：DDP 下未使用 DistributedSampler

```python
# ---- 错误 ----
# 每个进程都会加载全部数据，导致每个 epoch 中数据被处理 world_size 次
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ---- 正确 ----
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
loader = DataLoader(dataset, batch_size=32, sampler=sampler)
```

### 错误 4：DDP 下忘记 set_epoch

```python
# ---- 错误 ----
for epoch in range(NUM_EPOCHS):
    # 忘记 sampler.set_epoch(epoch)
    # 每个 epoch 的数据顺序完全相同！相当于只在 epoch 0 shuffle 了一次
    for inputs, labels in loader:
        ...

# ---- 正确 ----
for epoch in range(NUM_EPOCHS):
    sampler.set_epoch(epoch)  # 必须！保证每个 epoch 数据顺序不同
    for inputs, labels in loader:
        ...
```

### 错误 5：在 torch.inference_mode 或 torch.no_grad 下用 AMP

```python
# ---- 错误 ----
# 推理时不需要 autocast 或 scaler，也不需要 no_grad + autocast 混用
model.eval()
with torch.no_grad():
    with torch.cuda.amp.autocast():  # 不必要，推理不需要 AMP
        outputs = model(inputs)

# ---- 正确（推理） ----
model.eval()
with torch.inference_mode():
    outputs = model(inputs)
# 如果在推理中想用 FP16 加速，可以：
# model = model.half()  # 将模型转为 FP16
# with torch.inference_mode():
#     outputs = model(inputs.half())
```

### 错误 6：AMP 训练时手动梯度裁剪忘记 unscale

```python
# ---- 错误 ----
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度还是 scaled 的！
scaler.step(optimizer)
scaler.update()

# ---- 正确 ----
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # 先 unscale 梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

---

## 6.8 本章小结

- **GPU 适合高度并行计算**（矩阵乘法、卷积），不适合串行逻辑和小张量操作。
- **PCIe 是 CPU-GPU 数据传输瓶颈**：使用 `pin_memory=True` + `non_blocking=True` 优化传输。
- **`tensor.to(device)` 是转移 tensor 的推荐方式**，模型和数据必须在同一设备。
- **DP 简单但低效**（GPU 0 过载、GIL 问题），不推荐生产使用。
- **DDP 是标准多卡方案**：多进程、NCCL 后端、AllReduce 梯度同步、`DistributedSampler` 分配数据，用 `torchrun` 启动。
- **混合精度训练 (AMP)** 用 FP16 计算加速 + FP32 权重保精度。核心组件：`autocast`（自动精度选择）+ `GradScaler`（防止梯度下溢）。
- **AMP 可节省约 40% 显存**，加速 50-80%，精度几乎无损失。
- **BF16**（Ampere+ GPU）具有与 FP32 相同的动态范围，是 FP16 的替代方案，不易溢出。
- **AMP 出现 NaN 时**：降学习率、换 BF16、检查数据质量、使用梯度裁剪。
