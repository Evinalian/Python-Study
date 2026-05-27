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

## 6.1 为什么是 GPU：一个关于"军队"和"将军"的比喻

### 6.1.1 两种计算哲学

要理解 GPU 为什么适合深度学习，最有效的心理模型是一个军事比喻。

**CPU 是一支精英特种部队**。每个成员（核心）都经过极其复杂的训练，能独立完成高难度的战术任务——分析情报、制定计划、处理意外的战场变化。但特种部队的人数有限（4-64 人）。他们追求的是**单个任务的延迟最小化**——从接到命令到完成任务，越快越好。

**GPU 是一支庞大的步兵军团**。每个士兵（CUDA 核心）的能力远不如特种兵——只能执行简单的指令（加、乘、比较）。但军团的数量是惊人的（数千到上万个士兵）。他们追求的不是单人速度，而是**单位时间内完成的总工作量最大化**——也就是吞吐量。

现在想象你要完成一个任务：把 100 万对数字相加。特种部队再精英，也只有几十个人，他们需要排队一个接一个地做，耗时漫长。而步兵军团有上万人，每个人分 100 对数字，瞬间完成。深度学习恰恰就是这种"把 100 万对数字相加"的任务——矩阵乘法、卷积、批量归一化，本质上都是大规模并行的简单算术。

用这个心理模型武装自己，你在写 GPU 代码时会始终自问："这段代码能让成千上万个 CUDA 核心同时忙起来吗？还是让它们排队等一个人？"

### 6.1.2 从硅片层面理解差异

将军队的类比落实到芯片层面：

**CPU 的设计**围绕"如何把单个任务做得更快"展开。每个核心有庞大的缓存层级（L1/L2/L3，加起来几十 MB），复杂的分支预测器（猜测 `if` 语句会走哪条路）、乱序执行引擎（不按代码顺序执行，哪个指令准备好了就先执行哪个）、超标量流水线（同时发射多条指令）。这些复杂机制占据了芯片的大部分面积——实际上 CPU 芯片中真正做算术的单元（ALU）只占很小一部分。

**GPU 的设计**围绕"如何在单位面积内塞进更多的算术单元"展开。一个 GPU 芯片上，绝大部分面积是成千上万个简单的 CUDA 核心（每个核心本质上是一个小型浮点乘加器）。控制逻辑被最小化——而且 32 个 CUDA 核心共享同一套控制单元（这 32 个核心组成一个 warp，执行完全相同的指令，被称为 SIMT——单指令多线程）。缓存很小（几十 KB 的共享内存 + L2），因为 GPU 的战术不是减少内存延迟，而是用海量线程隐藏延迟——当一个 warp 在等内存时，硬件切换到另一个 warp 执行。

**数值对比让你感受量级差异：**

| 指标 | Intel i9-13900K (CPU) | NVIDIA RTX 4090 (GPU) |
|------|----------------------|----------------------|
| 核心数 | 24 (8P+16E) | 16384 CUDA Cores |
| 主频 | 3.0-5.8 GHz | 2.23-2.52 GHz |
| 内存带宽 | ~90 GB/s (DDR5) | ~1008 GB/s (GDDR6X) |
| FP32 理论算力 | ~2 TFLOPS | ~82.6 TFLOPS |
| 显存/内存 | 系统内存 32-128 GB | 24 GB GDDR6X |
| 功耗 | 125-253W | 450W |

注意 FP32 算力的对比：82.6 vs 2——**41 倍的差距**。这不是因为 GPU 的核心更快（实际上 CPU 的主频是 GPU 的两倍以上），而是因为 GPU 有 16000 多个核心同时工作。在一个时钟周期内，GPU 能完成的数量级的浮点操作是 CPU 无法企及的。

### 6.1.3 什么操作适合 GPU，什么不适合

这个判断是 GPU 编程的核心能力。判断标准只有一个：**任务是否可以分解为大量独立的、同质的子任务？**

**高度适合 GPU 的操作**（计算密集 + 数据并行）：
- 矩阵乘法：`C = A @ B`，C 中每个元素的计算相互独立，可以分配给不同的 CUDA 核心
- 卷积：每个输出位置的卷积运算相互独立
- 大批量 element-wise 操作：激活函数 ReLU/Sigmoid 对每个元素独立操作
- 大张量的 reduction：sum、mean——可以分层并行汇总

**不适合 GPU 的操作**（串行依赖 + 小数据量）：
- 小张量运算：一个 2x2 的矩阵乘法，GPU kernel 启动开销（几微秒）比计算本身还大
- 依赖输入数据的复杂控制流：`if x > 0: 做 A else: 做 B`——如果 32 个线程（一个 warp）中有 15 个走 A、17 个走 B，它们必须串行执行两个分支（warp divergence）
- 顺序依赖强的操作：RNN 的逐时间步循环——`h_t = f(h_{t-1}, x_t)`，必须先算出 `h_{t-1}` 才能算 `h_t`
- Python 原生的列表/字典操作：在 CPU 上运行，即使数据在 GPU 上

### 6.1.4 PCIe：CPU 和 GPU 之间的"独木桥"

CPU 和 GPU 不是坐在同一张桌子上的——它们通过 PCIe 总线连接。这条总线的带宽是深度学习训练中的关键瓶颈。以 RTX 4090 为例：

```
┌──────┐    PCIe 4.0 x16     ┌──────┐
│ CPU  │◄──────────────────►│ GPU  │
│ RAM  │   ~32 GB/s 单向     │ VRAM │
└──────┘                     └──────┘
```

GPU 内部显存带宽约 1008 GB/s。PCIe 4.0 x16 单向带宽约 32 GB/s。**差距约 30 倍。**

这个数字的含义是：如果你的训练代码频繁在 CPU 和 GPU 之间搬运数据，PCIe 会成为一个巨大的瓶颈——GPU 大部分时间在等数据，它的算力完全浪费了。这解释了为什么训练的最佳实践是：**尽可能一次性把数据搬到 GPU 上，在 GPU 上完成尽可能多的操作，只在必要时把结果搬回 CPU**。

`pin_memory=True` 和 `non_blocking=True` 是缓解 PCIe 瓶颈的两个关键优化。`pin_memory` 将 CPU 内存锁定在物理页上（避免被 OS 换出），使 DMA 传输可以直接进行而无需 CPU 介入。`non_blocking` 让数据传输异步进行——GPU 拷贝数据的同时 CPU 可以继续执行（比如准备下一批数据）。

---

## 6.2 PyTorch CUDA 编程基础

### 6.2.1 理解 device 的概念

在 PyTorch 中，`device` 对象就像一个地址标签——它告诉框架"这个张量/模型住在哪里"。CPU 上的张量和 GPU 上的张量是完全不同的物理实体，它们之间的运算必须先搬迁到同一地址。

```python
import torch

# device 对象的几种创建方式
device_cpu = torch.device("cpu")
device_gpu0 = torch.device("cuda")       # 等价于 cuda:0，默认第一块 GPU
device_gpu1 = torch.device("cuda:1")     # 第二块 GPU
device_gpu2 = torch.device("cuda", 2)    # 等价于 cuda:2

# 最常用的自动选择写法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
```

### 6.2.2 检查 CUDA 环境和 GPU 属性

在开始写 GPU 代码之前，了解你手头的硬件是必要的：

```python
import torch

# ---- 基础检查 ----
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"GPU 数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"当前设备名称: {torch.cuda.get_device_name(0)}")

    # ---- 显存使用 ----
    print(f"已分配显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"已缓存显存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    # memory_allocated: 张量实际占用的显存
    # memory_reserved: PyTorch 缓存池大小（包括已释放但未归还 OS 的）

    # ---- GPU 详细属性 ----
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU 详细属性:")
    print(f"  名称: {props.name}")
    print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
    print(f"  多处理器数(SM): {props.multi_processor_count}")
    print(f"  计算能力: {props.major}.{props.minor}")
```

`memory_allocated` 和 `memory_reserved` 的区别值得强调：PyTorch 使用自己的显存缓存池来提高分配效率。当你 `del x` 删除一个大张量后，它的显存回到缓存池而不是立即归还给操作系统。下次创建新张量时可以直接从缓存池里拿，省去了向 OS 申请显存的开销。这就是为什么你看到 `memory_reserved > memory_allocated` 是正常的。

### 6.2.3 Tensor 在 CPU 和 GPU 之间转移

张量的设备迁移有几种等价写法，推荐统一使用 `.to(device)` 风格——它明确表达了意图，且与 dtype 转换使用相同的 API。

```python
import torch

# 在创建时指定 device
x_cpu = torch.randn(3, 4)                     # 默认 CPU
x_gpu = torch.randn(3, 4, device="cuda")      # 直接在 GPU 创建
x_dev = torch.randn(3, 4, device=device)      # 使用变量

print(f"x_cpu device: {x_cpu.device}")        # cpu
print(f"x_gpu device: {x_gpu.device}")        # cuda:0

# .to 方法（推荐）：统一设备迁移和类型转换
x = torch.randn(3, 4)                         # CPU
x_gpu = x.to("cuda")                          # CPU -> GPU
x_gpu2 = x.to(device)                         # 使用 device 变量
x_back = x_gpu.to("cpu")                      # GPU -> CPU

# .to 方法的完整签名:
# tensor.to(device, dtype=None, non_blocking=False, copy=False)
# dtype: 同时转换数据类型（如 .to(torch.float16)）
# non_blocking: 异步传输（仅对 pinned memory 有效）

# 旧式写法 .cuda() / .cpu() —— 仍可用但不推荐
x_gpu = x.cuda()                              # 转移到默认 GPU
x_cpu = x_gpu.cpu()                           # 转回 CPU
```

### 6.2.4 模型转移到 GPU

模型转移理解起来很简单——`model.to(device)` 会递归地将所有 `nn.Parameter` 和 buffer 移到目标设备。但有一个非常重要的规则：**模型和数据必须在同一设备上**。违反这个规则会得到一个清晰的 RuntimeError，不会产生静默错误。

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)

# 模型转移到 GPU
model = MyModel().to(device)
print(f"fc.weight device: {model.fc.weight.device}")  # cuda:0

# 数据和模型必须同设备
x = torch.randn(5, 100)  # CPU 上的数据
# output = model(x)        # RuntimeError! 模型在 GPU，数据在 CPU

x = x.to(device)          # 数据也转移到 GPU
output = model(x)          # 正确！模型和数据都在 GPU
```

### 6.2.5 CPU/GPU Tensor 混用的常见陷阱

这类错误是 GPU 编程新手最常遇到的，且报错信息非常明确（"Expected all tensors to be on the same device"），但排查起来有时候很费劲——尤其是当问题来自于自定义 loss 函数或指标计算中悄悄创建的 CPU tensor。

最常见的三种情况：

1. **模型和数据不同设备**——模型初始化后 `to(device)` 了，但 DataLoader 出来的数据忘记 `to(device)`
2. **两个操作数在不同 GPU**——`a.cuda(0) + b.cuda(1)`，多卡编程中的典型错误
3. **自定义计算中创建了新 CPU tensor**——比如在 loss 或 acc 计算中写了 `torch.tensor([1.0])`（默认 CPU），然后和 GPU tensor 运算

```python
# 调试工具：快速定位设备不一致
def check_device(model, *tensors):
    """检查模型和所有输入 tensor 是否在同一 device"""
    model_device = next(model.parameters()).device
    for i, t in enumerate(tensors):
        if t.device != model_device:
            print(f"[WARNING] tensor[{i}] is on {t.device}, "
                  f"but model is on {model_device}")
```

### 6.2.6 显存管理

理解 PyTorch 的显存使用模式有助于你避免 OOM（Out of Memory）。几个关键概念：

- **allocated**：张量实际占用的显存。这是你真正关心的数字——当它接近 GPU 总显存时就会 OOM
- **reserved**：PyTorch 缓存池的大小。通常 1.2-1.5x allocated。缓存池是为了加速内存分配——频繁向 OS 申请/释放显存很慢
- **empty_cache()**：将缓存池中未使用的显存归还给 OS。这不会释放被张量引用的显存（你需要 `del` 那些张量），只释放已经 `del` 了但还留在缓存池里的部分

```python
if torch.cuda.is_available():
    # 查看显存使用
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    print(f"已分配: {allocated:.4f} GB | 已缓存: {reserved:.4f} GB | 峰值: {max_allocated:.4f} GB")

    # 重置峰值记录
    torch.cuda.reset_peak_memory_stats()

    # 清空缓存
    torch.cuda.empty_cache()
```

**显存快照**是调试 OOM 的利器：在代码开始处开启历史记录，在 OOM 发生后 dump 快照文件，然后上传到 PyTorch 官方的 Memory Visualizer 网站查看——它会以时间线的方式展示每个张量的创建和释放，让你精确定位是谁吃掉了显存。

### 6.2.7 Pinned Memory 与异步传输

`pin_memory=True` 和 `non_blocking=True` 是 DataLoader 配合 GPU 训练的两个关键开关。它们的共同目标是**将数据传输和 GPU 计算重叠起来**。

普通内存页可以被 OS 换出到磁盘（page fault）。DMA（直接内存访问）在传输数据前需要锁定这些页面。`pin_memory=True` 提前锁定了 DataLoader 输出的内存页，使 DMA 可以零等待传输。`non_blocking=True` 让 `to(device)` 变成异步调用——GPU 拷贝数据的同时 CPU 继续执行后面的代码。GPU 会追踪数据传输的状态，当 `model(inputs)` 真正需要数据时会自动等待传输完成。

```python
from torch.utils.data import DataLoader, TensorDataset

# 优化配置
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True,    # 锁定内存页，加速 DMA 传输
    num_workers=2,       # 多进程预加载数据
)

# 训练循环中使用 non_blocking
for inputs, labels in loader:
    inputs = inputs.to(device, non_blocking=True)   # 异步传输
    labels = labels.to(device, non_blocking=True)
    # 后续 GPU 操作会自动等待传输完成
```

---

## 6.3 多 GPU 训练：DP 到 DDP 的进化

### 6.3.1 单卡不够时的选项

三种情况迫使你考虑多 GPU 训练：(1) 模型太大，单卡显存放不下；(2) 你想用更大的 batch_size 以获得更稳定的梯度估计；(3) 你想缩短训练时间。多 GPU 训练即是将上述需求分摊到多张卡上并行执行。

### 6.3.2 DataParallel (DP)：一条船上的所有人

DP 是 PyTorch 最古老的多卡方案——只需要在模型上包一层 `nn.DataParallel`，看起来无比简单。但它的简单是表象，背后是严重的架构缺陷。

DP 的工作原理可以概括为"主从模式"：GPU 0 是主卡，其他卡是苦力。每步训练中，主卡将 batch 拆成 N 份分发给各卡（scatter）-> 各卡各自前向传播 -> 主卡收集所有输出（gather）-> 主卡计算 loss -> 主卡分发 loss 到各卡（scatter）-> 各卡反向传播 -> 主卡收集所有梯度（gather）-> **只有**主卡更新参数 -> 主卡广播更新后的参数到各卡。

这导致三个问题：(1) GPU 0 负载远超其他卡——它的利用率可能是 90%，而其他卡只有 60%；(2) Python 多线程受 GIL 限制，导致 scatter/gather 操作串行化；(3) 无法跨机器——所有 GPU 必须通过主卡的线程访问。

```python
model = MyModel()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)  # device 应为 cuda:0（主卡）
# 训练代码不变，但 GPU 0 是瓶颈
```

**DP 是快速实验的工具，不是生产部署的方案。** 如果你只是想在自己的双卡台式机上跑个小实验试试，DP 够用。但任何时候只要涉及多卡训练的效率优化，请用 DDP。

### 6.3.3 DistributedDataParallel (DDP)：平等联邦

DDP 的设计哲学与 DP 截然相反：**每个 GPU 是一个独立进程，所有进程平权**。每个进程有自己完整的模型副本、自己的数据分片、自己的优化器。前向传播各自独立，反向传播各自独立——唯一需要通信的是梯度。DDP 在 `backward()` 的过程中自动完成 AllReduce 梯度同步：每个进程计算出自己的梯度后，所有进程交换并计算平均值，使得所有进程上的参数更新完全一致。

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
│  └────┬─────┘    └────┬─────┘    └────┬─────┘     │
│       │               │               │           │
│       └───────────────┼───────────────┘           │
│                       │                           │
│              AllReduce 梯度（NCCL）               │
│        每个进程得到相同的平均梯度                    │
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
    """初始化 DDP 进程组。torchrun 自动设置环境变量。"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    dist.init_process_group(
        backend="nccl",          # NVIDIA GPU 专用通信后端
        init_method="env://",    # 从环境变量读取（torchrun 设置）
    )
    torch.cuda.set_device(local_rank)

    return local_rank, world_size, rank

def cleanup_ddp():
    dist.destroy_process_group()

# ============================================================
# 2. 定义模型
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ============================================================
# 3. 分布式训练主循环
# ============================================================
def main():
    local_rank, world_size, rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    print(f"[Rank {rank}/{world_size}] 初始化完成，使用 device={device}")

    # ---- 准备数据（DistributedSampler 确保不重复） ----
    X = torch.randn(1000, 1, 28, 28)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    dataloader = DataLoader(
        dataset, batch_size=32, sampler=sampler,
        num_workers=2, pin_memory=True,
    )

    # ---- 创建模型并 DDP 包装 ----
    model = SimpleCNN().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # DDP 包装后，原始模型在 model.module 中

    # ---- 损失函数和优化器 ----
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---- 训练循环 ----
    NUM_EPOCHS = 10
    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)  # 必须！每个 epoch 重新洗牌

        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # DDP 在 backward 时自动 AllReduce 梯度
            optimizer.step()

            total_loss += loss.item()

            if rank == 0 and batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} "
                      f"Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)

        # 跨进程平均 loss（各进程处理的样本数可能不同）
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss_tensor /= world_size

        if rank == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} Avg Loss: {avg_loss_tensor.item():.4f}")

    # ---- 只在 rank 0 保存 ----
    if rank == 0:
        torch.save(model.module.state_dict(), "ddp_model.pth")

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

# 多机（2 台机器，每台 2 卡）
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 \
         --master_addr="192.168.1.100" --master_port=29500 \
         ddp_train.py
```

**DDP 的关键细节：**

- `sampler.set_epoch(epoch)` 不能忘记——否则每个 epoch 的数据顺序完全相同，失去了 shuffle 的意义
- 日志打印最好只在 `rank == 0` 时执行，否则所有进程同时刷屏
- 模型保存用 `model.module.state_dict()` 而不是 `model.state_dict()`——因为 `model` 被 DDP 包装了，`model.module` 才是原始模型
- DDP 的梯度同步在 `backward()` 中自动完成，不需要显式调用通信

### 6.3.4 DP vs DDP 完整对比

| 维度 | DataParallel (DP) | DistributedDataParallel (DDP) |
|------|-------------------|-------------------------------|
| **实现方式** | 单进程多线程 | 多进程（每卡独立进程） |
| **Python GIL** | 受 GIL 影响（多线程） | 不受影响（多进程） |
| **通信方式** | gather/scatter/broadcast（GPU 0 是瓶颈） | AllReduce（Ring AllReduce，各卡平等） |
| **通信后端** | 隐式（PyTorch 内部） | NCCL（NVIDIA 专用，极快） |
| **负载均衡** | GPU 0 过载 | 各卡负载均匀 |
| **梯度同步时机** | 所有梯度 gather 到 GPU 0 后再广播 | backward 时自动 AllReduce（重叠计算与通信）|
| **多机支持** | 不支持 | 支持 |
| **推荐程度** | 不推荐（仅用于快速实验） | **强烈推荐** |

---

## 6.4 混合精度训练 (AMP)：两全其美

### 6.4.1 精度格式：为什么 FP16 和 FP32 不能互相替代

在进入混合精度之前，必须理解 FP32、FP16 和 BF16 三种格式的本质差异。浮点数由三部分组成：符号位（正负）、指数位（决定数值范围）、尾数位（决定精度或"分辨率"）。

| 格式 | 总位数 | 符号位 | 指数位 | 尾数位 | 数值范围 | 最小正数 |
|------|--------|--------|--------|--------|----------|----------|
| FP32 | 32 | 1 | 8 | 23 | ~1.18e-38 到 ~3.4e38 | ~1.19e-7 |
| FP16 | 16 | 1 | 5 | 10 | ~6.10e-5 到 65504 | ~9.77e-4 |
| BF16 | 16 | 1 | 8 | 7 | ~1.18e-38 到 ~3.39e38 | ~7.81e-3 |

**FP16 的致命弱点是数值范围**。因为指数位只有 5 位，它能表示的最大值约 65504，最小值约 6e-5。训练中的梯度经常小于 6e-5（下溢变零）——对应的参数就永远不更新了。也很容易超过 65504（上溢变无穷大）——loss 直接 NaN。这就是为什么纯 FP16 训练几乎行不通。

**BF16 的聪明之处**：它把指数位扩展到 8 位（和 FP32 一样），换来和 FP32 完全相同的数值范围。代价是尾数位只有 7 位（精度更低）。对于深度学习来说，数值范围远比精度重要——梯度下溢是灾难性的，而精度损失通常可以通过训练过程自动补偿。

**数据在 FP16 下的行为演示：**

```python
# 梯度下溢示例
x = torch.tensor([0.0001], dtype=torch.float16)
print(x)  # tensor([9.9945e-05]) -- 有舍入误差。更小的值可能直接变 0

# 梯度上溢示例
y = torch.tensor([100000.0], dtype=torch.float16)
print(y)  # tensor([inf]) -- 超过 FP16 最大值
```

### 6.4.2 混合精度的核心思想：分而治之

既然 FP32 稳但慢、FP16 快但不稳，那就各用所长。混合精度训练的架构是三层分工：

- **前向/反向计算**：FP16——速度快、显存省（权重和激活值各减半）
- **权重主副本**：FP32——参数更新在 FP32 精度下进行，维持长期稳定性
- **Loss Scaling**：在反向传播前将 loss 乘以一个大因子（初始 65536），放大后的梯度脱离了 FP16 的下溢区。在 `step()` 更新参数前，把梯度再缩小回原始尺度。缩放和还原完全在 FP32 精度下进行

### 6.4.3 AMP 核心组件

PyTorch 的 `torch.cuda.amp` 模块提供了两个核心工具，它们配合使用：

**`autocast` 上下文管理器**：自动为每个操作选择最合适的精度。大部分矩阵乘法和卷积自动用 FP16，而数值敏感的操作（Softmax、LayerNorm、某些 reduction）自动保持 FP32。你不需要手动指定——PyTorch 维护了一个"FP32 白名单"，这些层在 autocast 下自动被保护。

**`GradScaler` 梯度缩放器**：动态管理 loss scaling factor。初始值 65536。如果连续多次迭代都没有出现 inf/NaN 梯度，scale 翻倍（更少的下溢风险）；一旦出现 inf/NaN，scale 减半并跳过此次参数更新。这个动态调整机制让 scales 始终维持在"既不会下溢也不会上溢"的甜蜜点。

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
            nn.Linear(100, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 3),
        )
    def forward(self, x):
        return self.net(x)

model = MLP().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---- AMP：创建 GradScaler ----
scaler = torch.cuda.amp.GradScaler()

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
        scaler.scale(loss).backward()

        # 步骤 3：unscale 梯度 -> 更新参数
        # scaler.step 内部做了 unscale_ + optimizer.step
        # 如果发现 inf/NaN，自动跳过此次更新
        scaler.step(optimizer)

        # 步骤 4：更新缩放因子
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    current_scale = scaler.get_scale()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Scale: {current_scale:.0f}")

print("训练完成！")
```

**关键步骤解读**：

1. `with autocast():` 包裹前向传播——自动为各层选择精度
2. `scaler.scale(loss).backward()` 替代 `loss.backward()`——loss 乘以 scale_factor 后再反向传播，梯度也被等比放大
3. `scaler.step(optimizer)` 替代 `optimizer.step()`——内部先将梯度除以 scale_factor 恢复原始尺度，再正常更新参数
4. `scaler.update()` 在每个 batch 后调用——动态调整 scale_factor

### 6.4.5 AMP + DDP 组合

AMP 和 DDP 可以无缝组合。关键的注意事项：如果需要在 AMP 训练中使用梯度裁剪，**必须在 `scaler.step()` 之前手动 `unscale_`**，否则裁剪的是缩放后的梯度（而不是真实的梯度），效果完全错误。

### 6.4.6 AMP 的显存与速度收益

**典型数据（ResNet-50 在 RTX 3090 上的训练）：**

| 配置 | Batch Size | 显存占用 | 训练速度 (img/s) | 精度 (Top-1) |
|------|-----------|---------|-----------------|-------------|
| FP32 | 128 | 18.2 GB | 450 | 76.1% |
| AMP (FP16) | 128 | 10.5 GB (-42%) | 720 (+60%) | 76.0% |
| AMP (FP16) | 256 | 20.8 GB | 820 (+82%) | 76.1% |

三个关键发现：(1) 43% 的显存节省让你可以用更大的 batch_size（256 vs 128），而更大的 batch_size 通常意味着更稳定的梯度估计；(2) 60%+ 的速度提升来自 Tensor Core 对 FP16 的原生加速——Ampere 架构的 FP16 吞吐量是 FP32 的 8 倍；(3) 精度几乎无损——76.0% vs 76.1% 的差异在统计波动范围内。

**当 AMP 训练出现 NaN 时的排查流程：**

NaN 意味着某次操作产生了超出 FP16 表示范围的值。排查的顺序是关键：

```python
# 1. 检查 GradScaler 的 scale 是否持续下降
print(f"Current scale: {scaler.get_scale()}")  # 如果 < 100，说明频繁出现溢出

# 2. 降低学习率（AMP 对学习率更敏感）
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 从 0.001 降到 0.0001

# 3. 换用 BF16（Ampere+ GPU 支持，数值范围与 FP32 相同）
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    ...

# 4. 检查数据中是否有 NaN/Inf
if torch.isnan(inputs).any() or torch.isinf(inputs).any():
    print("输入数据包含 NaN/Inf！")

# 5. 梯度裁剪（在 unscale 之后）
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

---

## 6.5 基础练习

### 练习 6-1：CUDA 环境探测与 Tensor 转移

编写脚本探测 CUDA 环境信息（GPU 数量、名称、显存），并在 CPU/GPU 间转移 tensor，测量转移耗时。

### 练习 6-2：AMP 混合精度训练对比

训练同一个模型，分别使用 FP32 和 AMP (FP16) 两种模式，对比显存占用、训练速度和最终 loss。

### 练习 6-3：CPU/GPU Tensor 混用错误复现与修复

故意写出 4-5 种常见的 CPU/GPU Tensor 混用错误，然后用 `try/except` 捕获并打印错误信息，最后给出正确写法。

---

## 6.6 进阶练习

### 练习 6-4：DDP 多卡训练

实现一个完整的 DDP 训练脚本，包含 `DistributedSampler`、梯度同步、只在 rank 0 保存模型和打印日志。

### 练习 6-5：AMP + DDP 组合

将 AMP 和 DDP 组合到同一个训练脚本中，实现完整的分布式混合精度训练。

---

## 6.7 常见错误

### 错误 1：CPU 和 GPU Tensor 直接运算

```python
# ---- 错误 ----
model = MyModel().cuda()
x = torch.randn(5, 100)  # CPU
y = model(x)  # RuntimeError!

# ---- 正确 ----
x = torch.randn(5, 100).to(device)  # 或 .cuda()
y = model(x)
```

### 错误 2：loss 计算中引入 CPU Tensor

自定义 loss 或指标计算中容易无意间创建 CPU tensor。规则：**在 GPU 训练中，显式创建的 tensor 默认在 CPU 上。** 始终加 `device=device` 或 `.to(device)`。

### 错误 3：DDP 下未使用 DistributedSampler

不使用 DistributedSampler 的后果：每个进程都会独立加载全部数据，导致数据被处理 `world_size` 次——每个 epoch 的有效数据量被放大了 N 倍。

### 错误 4：DDP 下忘记 set_epoch

`sampler.set_epoch(epoch)` 的作用是让每个 epoch 获得不同的随机种子。忘记这行代码的结果是：每个 epoch 的数据顺序完全相同。

### 错误 5：AMP 训练时手动梯度裁剪忘记 unscale

```python
# ---- 错误 ----
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度还是 scaled 的！
scaler.step(optimizer)

# ---- 正确 ----
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # 先 unscale 梯度
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

gradients 在被 scaler 缩放后值域是放大的，如果直接裁剪就失去了缩放的意义。必须先 `unscale_` 恢复真实梯度尺度，再裁剪。

---

## 6.8 本章小结

**GPU 是吞吐量机器，不是速度机器**。它用数千个简单的核心同时处理大量同质任务，来换取对 CPU 的数量级算力优势。深度学习的矩阵乘法和卷积天然适合这种"海量并行、简单指令"的计算模式。但 GPU 的力量伴随着约束：CPU-GPU 之间的 PCIe 传输是瓶颈、显存是稀缺资源、控制流会导致性能骤降。

**device 管理是基本功**。模型和数据必须在同一设备上——这个规则的违反者最容易产生报错但也最容易修复。`pin_memory=True` 和 `non_blocking=True` 将数据传输与计算重叠，是免费的加速手段。

**多 GPU 训练从 DP 进化到 DDP**。DP 用一个主进程管理所有卡——简单但低效（GPU 0 过载、GIL 瓶颈）。DDP 是当前的标准方案：每个 GPU 独立进程、NCCL 通信后端、AllReduce 梯度同步、DistributedSampler 分片数据。用 `torchrun` 启动，每个进程一个独立的 Python 解释器。

**混合精度训练通过 FP16 和 FP32 的分工实现了"鱼和熊掌兼得"**：FP16 负责计算（快、省显存），FP32 负责权重存储和参数更新（稳），GradScaler 防止梯度下溢。收益是实质性的：约 40% 显存节省，50-80% 速度提升，精度几乎无损。BF16（Ampere+ GPU）以尾数精度的代价换来了与 FP32 相同的数值范围，是 FP16 的强替代。

**排查 AMP 中的 NaN**：降学习率（最常见的原因）、换 BF16、检查数据质量、使用梯度裁剪。GradScaler 的 scale 持续下降是 Grad 中频繁出现 inf/NaN 的信号。
