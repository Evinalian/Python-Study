# 第07章 Scaling Law 与效率优化

## 学习目标

1. 理解 Kaplan Scaling Law 的幂律关系：参数 N、数据 D、计算 C 与 Loss L 的关系
2. 掌握 Chinchilla Law 的核心结论：模型大小和数据量应等比放大
3. 学会用 Scaling Law 进行小规模实验外推大模型性能
4. 深入理解 Flash Attention 的分块计算原理
5. 理解混合精度训练的策略：FP16/BF16/FP32 的角色分工
6. 掌握 Gradient Checkpointing 的"计算换显存"权衡
7. 了解 torch.compile 的图编译优化
8. 了解高效推理技术：KV Cache、Continuous Batching、Speculative Decoding

## 前置知识

- 第01章：优化器、损失函数、训练流程的完整理解
- 第02章：Attention 机制的详细计算过程
- 第05章：分布式训练、GPU 显存组成（参数、梯度、优化器状态、激活值）
- 第06章：预训练流程、参数量计算
- 对 GPU 架构的基本了解：知道 GPU 有 HBM（高带宽显存）和 SRAM（共享内存/寄存器），知道 I/O 操作比计算操作慢几个数量级

---

## 1. Kaplan Scaling Law

### 1.1 什么是 Scaling Law

2020 年，OpenAI 发表了论文《Scaling Laws for Neural Language Models》（第一作者 Jared Kaplan），系统性地研究了语言模型性能如何随规模变化。这篇论文的核心发现被称为 **Kaplan Scaling Law**。

**Scaling Law 的基本形式**：

对于使用 Cross-Entropy Loss 训练的自回归语言模型，测试 Loss `L` 和三个核心变量之间存在幂律（power-law）关系：

```
L(N) ∝ N^(-α_N)            ← 模型参数量 N 的影响
L(D) ∝ D^(-α_D)            ← 训练数据量 D 的影响
L(C) ∝ C^(-α_C)            ← 总计算量 C 的影响
```

其中 `α` 是幂律指数（power-law exponent），约在 0.05-0.10 之间。这意味着：Loss 随 N、D、C 的增加而降低，但降低的速度在递减（因为指数很小）。

**具体来说**，Kaplan 等人的研究发现：
- 模型参数量 N 每增加 10 倍，Loss 约降低一定量
- 但单独增大模型而不同时增大数据会导致收益递减
- Loss 和计算量 C 的关系最为稳健

### 1.2 幂律关系的直观理解

幂律（power law）是自然界中普遍存在的一种关系。它描述的是：一个变量的相对变化导致另一个变量的相对变化。换句话说，在双对数坐标（log-log plot）上，幂律关系表现为一条直线。

```
log(Loss)
| \
|  \
|   \
|    \
|     \
+----------→ log(Compute)
```

**为什么幂律指数的数值很重要？** 假设 `α_C = 0.05`，这意味计算量增加 10 倍，Loss 降低约 `10^(-0.05) ≈ 0.89` 倍，即 Loss 降低约 11%。要让 Loss 降低一半，计算量需要增加约 `2^(1/0.05) ≈ 10^6` 倍——一百万倍。这说明通过增加计算量来降低 Loss 变得越来越"贵"。

### 1.3 Kaplan 的具体结论

1. **模型大小的影响**：固定数据量时，增加模型大小会降低 Loss，但收益递减。超过某个点后，增大模型反而有害（因为数据不足，模型过拟合）。

2. **数据量的影响**：固定模型大小时，增加数据量会降低 Loss，收益同样递减。

3. **计算量的影响**：这是最稳健的关系。在最优配置下（同时调整 N 和 D），Loss 与计算量 C 之间的幂律关系最为可靠。

4. **Batch Size**：最优 batch size 随 Loss 的降低而增大（粗略正比于 `1/Loss`）。

5. **大模型的样本效率更高**：令人反直觉的是，大模型可以用更少的数据达到同样的性能。一个 10B 参数的模型可能比一个 100M 参数的模型"更高效"地利用每个训练样本。

### 1.4 Kaplan Law 的历史局限性

Kaplan 论文的一个重要假设是：**随着模型规模增大，不需要等比例地增加训练数据**。具体说，他们建议模型参数 N 增大 8 倍时，数据 D 只需增大 5 倍。

这个假设后来被证明是不最优的。这就是下一节要讨论的 Chinchilla Law。

---

## 2. Chinchilla Law：最优计算分配

### 2.1 Chinchilla 的核心发现

2022 年，DeepMind 发表了论文《Training Compute-Optimal Large Language Models》，提出了 Chinchilla 模型和与之同名的 **Chinchilla Law**。

**Chinchilla 最重要的结论**：在给定计算预算下，模型大小和训练数据量应该**等比例**扩大。

如果计算预算增大 10 倍，那么：
- 模型参数量 N 应增大约 3.16 倍（√10）
- 训练 token 数 D 也应增大约 3.16 倍
- 总计算量 = 6 × N × D（近似公式），增大 10 倍

这个结论和 Kaplan 的建议形成了鲜明对比。按照 Chinchilla 的建议，对于 70B 参数的模型，应该用约 1.4T tokens 训练——大概是常见实践中的 2-3 倍数据量。

### 2.2 Chinchilla 的数值结果

Chinchilla 论文通过在 400 多种不同配置下训练模型（从 70M 到 16B 参数），拟合出了最优分配的公式：

```
N_opt(C) ∝ C^(0.50)
D_opt(C) ∝ C^(0.50)
```

即 N 和 D 的指数相等（都是 0.50），意味着它们应该等比放大。

在具体的数值上，对于计算量 C（FLOPs），最优的 N 和 D 大致满足：

```
N_opt ≈ C / 6D_opt  （近似关系）
```

或等价地：
```
D_opt ≈ 20 × N_opt   （每个参数约 20 个训练 token）
```

**"Chinchilla Optimal"的实践含义**：

如果一个模型有 70B 个参数，"Chinchilla 最优"的配置约需要 70B × 20 = 1.4T 个训练 token。但实际中：
- LLaMA 7B 用了约 1T tokens（远超 7B×20=140B，属于"过度训练"或"数据充足型"）
- Chinchilla 70B 用了约 1.4T tokens（恰好 Chinchilla 最优）
- GPT-3 175B 用了约 300B tokens（远低于 175B×20=3.5T，属于"训练不足"或"模型过大型"）

### 2.3 Kaplan vs Chinchilla 的对比实验

为什么两者的结论不同？主要原因：

1. **Kaplan 的实验范围有限**：他们主要改变了模型大小，而数据量变化较小。在有限的数据量下，更大的模型确实会饱和。但他们在分析时将"固定数据量下的饱和"误推为"所有情况下的规律"。

2. **学习率调度**：Kaplan 使用了固定步数的 cosine decay（无论数据量多少），这意味着当数据量增加时模型没有充分训练。Chinchilla 使用了更合理的 cosine decay 策略（每个 token 训练一次）。

3. **参数拟合方法**：Kaplan 使用了三变量幂律拟合，而 Chinchilla 使用了更精确的方法。

**现在的社区共识**：Chinchilla Law 更准确地描述了最优分配。在实践中，许多团队会选择"稍微过度训练"——使用比 Chinchilla 最优更多的数据。因为数据相对便宜（可以一直喂），而大模型的推理成本高（更大的模型 = 更慢的推理）。

### 2.4 Chinchilla Law 的实际应用

假设你有 1000 GPU-天的计算预算，你想训练一个性能最好的语言模型。你应该选择多大的模型？用多少数据？

使用 Chinchilla 的公式：
1. C = 1000 GPU-天 × A100 的 FP16 TFLOPS × 利用率
2. 计算最优的 N 和 D
3. 确保 N × D × 6 ≈ C

实际中还可以参考公开的 scaling 曲线（如 LLaMA 系列、Pythia 系列），它们提供了不同规模下的 perplexity 参考值。

---

## 3. Scaling Law 的实际应用

### 3.1 用小模型预测大模型性能

Scaling Law 最实用的价值之一：**你可以在小规模上做实验，然后外推大模型的性能。**

方法是：
1. 训练多个不同规模的小模型（如 1M、10M、100M 参数），每个用不同数据量
2. 拟合 Scaling Law 的幂律参数（α_N, α_D 等）
3. 将拟合的曲线外推到目标规模，预测 Loss
4. 用预测结果决定是否值得投入资源训练大模型

例如，如果一个 10M 模型的验证 loss 是 3.5，而 Scaling Law 预测 100M 模型的 loss 是 3.0，100B 模型约 1.8，你可以在不实际训练 100B 模型的情况下，提前评估大规模训练是否值得。

### 3.2 外推的技术细节

**IsoFLOP 曲线**（等计算量曲线）：

给定一个固定的计算预算 C，针对不同的模型大小 N，绘制最终的验证 Loss。这产生一条 U 形曲线——模型太小欠拟合，模型太大过拟合，中间有一个最优的 N。

```
Loss
↑
|  \         /
|   \       /
|    \_____/  ← 最优点
|     \   /
|      \ /
+----------→ N (固定计算预算)
```

在不同计算预算下重复这个实验，就能拟合出 N_opt(C) 和 D_opt(C) 的曲线。

**注意事项**：
- 外推的可靠性依赖于幂律假设。如果模型架构在尺度变化时行为发生变化（如涌现能力），外推可能不准确。
- 不同数据分布可能有不同的 Scaling Law。用维基百科训练的 Scaling Law 不一定适用于 CommonCrawl 训练。
- 学习率、batch size 等超参在大规模时可能需要调整。

### 3.3 Scaling Law 的例外

不是所有任务都服从平滑的 Scaling Law：
- **涌现能力**：某些能力（如数学推理、代码理解）只在模型达到一定规模后才突然出现，在此之前性能接近随机。这些能力不服从平滑的幂律。
- **某些 benchmark**：特定任务可能存在"相变"，很难用小型模型的性能外推。这就是为什么 scaling law 主要用于预测 perplexity 而非下游任务准确率。

---

## 4. Flash Attention

### 4.1 标准 Attention 的瓶颈

回顾 Standard Self-Attention 的计算过程：

```
S = Q @ K^T       # (N, N) 矩阵，N = 序列长度
P = softmax(S)    # 归一化
O = P @ V         # 输出
```

对于序列长度 N，这是 O(N²) 的计算和显存复杂度。当 N = 2048 时，注意力矩阵 S 有 4M 个元素；当 N = 32K 时，S 有 10 亿个元素。

但计算量不是唯一的瓶颈。**显存带宽**是更严重的瓶颈。

标准的 Attention 计算的流程是：
1. 将 Q、K 从 HBM（高带宽显存，如 GPU 的全局内存）加载到 SRAM（片上共享内存）
2. 在 SRAM 中计算 S = Q @ K^T，结果写回 HBM
3. 将 S 从 HBM 加载回 SRAM，计算 softmax，结果 P 写回 HBM
4. 将 P、V 从 HBM 加载，计算 O = P @ V，结果写回 HBM

**关键浪费**：中间结果 S 和 P 被写入 HBM 又被读回，而实际上它们不需要被长期保存。HBM 的带宽远低于 SRAM 的带宽（约 10-20 倍），这些不必要的 HBM 读写成为了瓶颈。

### 4.2 Flash Attention 的核心思想

Flash Attention（Dao et al., 2022）的核心洞察：**我们不需要一次性计算整个 S 矩阵。我们可以分块计算，让每个块的计算在 SRAM 中完整执行，中间结果不需要写回 HBM。**

具体来说，Flash Attention 做了两件事：

**第一，分块（Tiling）**：
将 Q、K、V 分成若干个块（block），每个块可以在 SRAM 中完全处理。一次加载一块 Q 和一块 K，计算局部的注意力分数，做局部的 softmax，累加到输出，然后加载下一块。

**第二，重计算（Recomputation）**：
在反向传播时，不保存完整的 S 矩阵（那需要 O(N²) 的显存），而是重新计算注意力分数。这是"计算换显存"的策略——用额外的计算避免存储巨大的中间结果。

### 4.3 为什么 softmax 可以分块计算

直观上，softmax 似乎是不可分块的——我们需要知道所有元素的值才能计算分母（归一化因子）。但数学上，softmax 可以通过一种"在线"算法增量计算。

标准 softmax：
```
softmax_i = exp(x_i) / Σ_j exp(x_j)
```

在线 softmax（用于分块计算）：
1. 计算第一个块的 max 和 sum(exp)
2. 对于第二个块，更新全局 max 和全局 sum(exp)，并修正之前的计算结果
3. 依次处理所有块

**具体公式**：假设已经处理了前 k 个分块，当前全局最大值是 `m_old`，全局 `exp` 和是 `l_old`，第 (k+1) 个分块的局部最大值是 `m_new`，局部 `exp` 和是 `l_block`。更新方式为：

```
m_combined = max(m_old, m_new)
l_combined = l_old * exp(m_old - m_combined) + l_block * exp(m_new - m_combined)
```

这是一个标准的"并行扫描"模式，可以增量、分块地计算 softmax。

### 4.4 Flash Attention 的效果

| 指标 | Standard Attention | Flash Attention |
|------|-------------------|-----------------|
| 显存占用 | O(N²) | O(N)（无需存储 S 矩阵） |
| HBM 读写量 | O(N²) | O(N²/B_r)（B_r 是块的 SRAM 大小） |
| 计算复杂度 | O(N²) | O(N²)（相同） |
| 实际加速（N=2K） | 1× | 2-4× |
| 实际加速（N=8K） | 1× | 5-8× |
| 关键因素 | 受限于 HBM 带宽 | I/O 优化：SRAM 内计算 |

Flash Attention-2（2023）进一步优化：减少了非矩阵乘法操作、优化了分块调度、更好地利用了 GPU 的 warp 级并行。

Flash Attention-3（2024）针对 H100 的新硬件特性（FP8、TMA 异步拷贝）做了进一步优化。

### 4.5 如何在 PyTorch 中使用 Flash Attention

PyTorch 2.0+ 内置了 `torch.nn.functional.scaled_dot_product_attention`，会自动选择最优的 Attention 实现（Flash Attention、Memory-Efficient Attention 或朴素实现）：

```python
import torch.nn.functional as F

# 自动选择最优实现
output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=mask,
    dropout_p=0.0,
    is_causal=True,
)
```

也可以手动安装 `flash-attn` 库（Tri Dao 的官方实现）：

```bash
pip install flash-attn --no-build-isolation
```

```python
from flash_attn import flash_attn_func

# 直接使用 Flash Attention
output = flash_attn_func(query, key, value, causal=True)
```

---

## 5. 混合精度训练回顾

### 5.1 为什么需要混合精度

深度学习训练中涉及三种浮点精度：

| 精度 | 位数 | 指数位 | 尾数位 | 表示范围 | 精度 |
|------|------|--------|--------|---------|------|
| FP32 | 32 | 8 | 23 | ~10^38 | ~7 位十进制 |
| FP16 | 16 | 5 | 10 | ~65504 | ~3 位十进制 |
| BF16 | 16 | 8 | 7 | ~10^38 | ~2 位十进制 |

- **FP32** 是传统的训练精度，精确但慢，显存占用大。
- **FP16** 速度快（Tensor Core 专用），但表示范围小（最大 65504），容易上溢。
- **BF16**（Brain Floating Point）是 Google 提出的格式：和 FP32 相同的指数位数（8 位），因此有相同的表示范围——不会上溢。但精度较低（只有 7 位尾数）。在 TPU 和 Ampere 以上的 GPU（A100/H100）原生支持。

### 5.2 混合精度的策略

**标准混合精度（FP16 + FP32）**：
- 前向和反向传播使用 FP16（速度快，省显存）
- 维护一份 FP32 的"主参数副本"（Master Weights），用于参数更新
- 使用 Loss Scaling 防止梯度下溢：将 loss 乘以一个大因子（如 1024），使得小梯度在 FP16 中也可以表示；反向传播后再除以该因子

**BF16 混合精度（无需 Loss Scaling）**：
- 前向和反向传播使用 BF16
- 参数更新使用 FP32
- BF16 的指数范围与 FP32 相同，不会上溢，因此不需要 Loss Scaling
- 在 A100/H100 上这是推荐的训练方案

### 5.3 PyTorch 中的混合精度

```python
import torch

# BF16 混合精度（推荐，A100/H100）
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)
    loss = loss_fn(output, target)

# FP16 混合精度（需要 Loss Scaling）
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
    loss = loss_fn(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 5.4 精度选择的实际建议

- **A100/H100 + 训练**：使用 BF16（简单位、稳定、高效）
- **V100 + 训练**：使用 FP16 + Loss Scaling（V100 不支持 BF16 的 Tensor Core 加速）
- **推理**：使用 FP16 或 INT8/INT4 量化（追求速度）
- **注意**：LayerNorm、Softmax 等数值敏感操作通常保持在 FP32 精度（autocast 自动处理）

---

## 6. Gradient Checkpointing

### 6.1 显存的"大头"

在训练中，显存的主要占用者是：
1. 模型参数（FP16/BF16）
2. 优化器状态（FP32，约为参数的 6 倍）
3. **激活值**（Activation Memory）——前向传播中每层的中间输出

**激活值为什么是显存大户？**

在标准反向传播中，我们需要保存每层前向传播的中间输出（激活值），因为反向传播计算梯度时需要它们。例如，对于一个 N 层的 Transformer：
- 每层需要保存 Attention 的 Q、K、V 和 Attention 输出
- 每层需要保存 FFN 的中间激活值
- 总激活值 ≈ N × batch_size × seq_len × d_model × (constant)

当 batch_size 和 seq_len 很大时，激活值可能占据总显存的 50% 以上。

### 6.2 Gradient Checkpointing 的原理

**核心思想**：只保存部分层的激活值（通常是每 1-2 个 Transformer Block 保存一个"checkpoint"），其余层的激活值在反向传播时重新计算。

```
标准反向传播:
前向: 保存所有层的激活值
反向: 使用保存的激活值计算梯度

Gradient Checkpointing:
前向: 只保存 checkpoint 层的激活值
反向: 对于 checkpoint 之间的层 —— 从最近 checkpoint 重新做前向传播, 恢复激活值, 再算梯度
```

这是**计算换显存**的经典实例。我们用额外的约 20-30% 的计算时间，换来 50-70% 的激活值显存节省。

### 6.3 PyTorch 中的使用

```python
from torch.utils.checkpoint import checkpoint

# 对每个 Transformer Block 应用 checkpointing
class TransformerBlock(nn.Module):
    def forward(self, x):
        # 使用 checkpoint 包装整个 block 的前向传播
        # 前向时不保存中间激活值，反向时重新计算
        return checkpoint(self._forward, x, use_reentrant=False)

    def _forward(self, x):
        # 实际的 block 计算
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

也可以对整个 block 的列表应用 checkpointing：

```python
from torch.utils.checkpoint import checkpoint_sequential

# 将每 2 个 block 作为一组应用 checkpointing
segments = 4  # 分成 4 组
x = checkpoint_sequential(self.layers, segments, x)
```

### 6.4 Gradient Checkpointing 的权衡

- 对于小模型（<100M），激活值占总显存比例不高，Gradient Checkpointing 收益有限
- 对于大模型（>1B），激活值是显存瓶颈，Gradient Checkpointing 几乎必须
- LLAMA 等模型的典型配置：每个 Transformer Block 都应用 checkpointing
- 额外的计算开销约 20-30%，在 FSDP/ZeRO-3 场景中可以通过重叠通信和计算来部分隐藏

---

## 7. torch.compile

### 7.1 图编译的威力

PyTorch 默认使用"eager mode"（及时执行模式）——每一行代码都被立即执行。这提供了调试的灵活性，但牺牲了优化空间。编译器可以分析整个计算图（或子图），进行各种优化：

- **算子融合**（Operator Fusion）：将多个连续的算子合并成一个。例如 `x.gelu() + x.dropout()` 可以合并为一个 kernel，减少显存读写。
- **内存规划**（Memory Planning）：分配一大块内存，复用不同中间结果的空间，避免频繁的显存分配和释放。
- **自动混合精度**（Automatic Mixed Precision）：编译器可以自动判断哪些操作可以用低精度。
- **图级优化**：消除死代码、常量折叠、代数化简。

### 7.2 torch.compile 的使用

PyTorch 2.0 引入了 `torch.compile`，它的接口极其简单：

```python
# 一行代码，模型被编译优化
model = torch.compile(model)

# 或者指定后端
model = torch.compile(model, backend="inductor")  # 默认
model = torch.compile(model, backend="cudagraphs")  # CUDA Graph
model = torch.compile(model, mode="reduce-overhead")  # 更激进的优化
```

`torch.compile` 的底层使用了 TorchDynamo（将 eager 代码转为 FX graph）+ TorchInductor（将 FX graph 编译为高效的 CUDA kernel）。

### 7.3 实际效果

对于 LLM 训练和推理：
- 训练加速：通常 1.2-1.5×（取决于模型和硬件）
- 推理加速：通常 1.3-2.0×
- 对 GPU 架构（A100/H100）的加速效果更明显

**注意事项**：
- 第一次运行时会编译（warmup），后续运行使用编译后的 kernel
- 动态形状（如变长序列）可能导致多次重新编译（graph break）
- 不是所有操作都支持编译。遇到不支持的操作，编译器中会回退到 eager mode（graph break），影响加速效果
- 使用 `TORCH_LOGS="+dynamo"` 和 `TORCHDYNAMO_VERBOSE=1` 可以查看 graph break 信息

---

## 8. 高效推理技巧

### 8.1 KV Cache

**问题**：在自回归生成中，每生成一个新的 token，都要对整个序列（包括已生成的 token）重新计算 Attention。这意味着：
- 第 1 步：计算 1 个 token 的 Attention
- 第 2 步：计算 2 个 token 的 Attention
- ...
- 第 N 步：计算 N 个 token 的 Attention

总计算量为 `1 + 2 + ... + N = O(N²)`。

**KV Cache 的思想**：既然已生成 token 的 Key 和 Value 不会改变（因为它们的输入是固定的），为什么每次都要重新计算？将它们缓存起来。

```
第 1 步: 计算 Q₁, K₁, V₁  → 缓存 K₁, V₁
第 2 步: 计算 Q₂, 用缓存的 K₁ 和新算的 K₂ 做 Attention → 缓存 K₂, V₂
第 3 步: 计算 Q₃, 用缓存的 K₁,K₂ 和新算的 K₃  → 缓存 K₃, V₃
...
```

这样每步只需计算当前 token 的 Q、K、V（O(1) 复杂度），而不是重新计算整个序列。总计算量降为 O(N)。

**KV Cache 的显存占用**：
```
KV Cache 大小 = 2 × n_layers × seq_len × d_model × dtype_bytes
```
对于 LLaMA 7B（n_layers=32, d_model=4096），生成长度为 2048 的序列：
```
KV Cache ≈ 2 × 32 × 2048 × 4096 × 2 字节 ≈ 1GB
```
对于 LLaMA 2 70B，这可能是 10+ GB——推理中的显存主要瓶颈。

**GQA（Grouped Query Attention）对 KV Cache 的影响**：LLaMA 2 使用了 GQA（多个 Q 头共享 KV 头）。如果 `n_kv_heads = 8` 而 `n_heads = 64`，KV Cache 的大小减少了 8 倍——这正是 GQA 的主要动机之一。

### 8.2 Continuous Batching

传统批处理推理中，一个 batch 中的所有请求必须"一起开始、一起结束"。这意味着：
- 一个短请求（5 个生成 token）完成后，必须等待同 batch 中的长请求（100 个 token）也完成
- GPU 处理短请求的位置时其实在空闲等待

**Continuous Batching（也叫 In-flight Batching）**：当一个请求完成时，立即将一个新请求注入到 batch 中取代它。这样 GPU 始终保持"满负荷"。

实现 continuous batching 的挑战是：不同请求处于不同的生成阶段，序列长度不同。需要使用特殊的 attention kernel（如 Flash Attention 的 varlen 模式）处理变长序列。

vLLM、TensorRT-LLM、Text Generation Inference (TGI) 等推理框架都实现了 continuous batching。

### 8.3 Speculative Decoding（投机解码）

**核心思想**：用一个小而快的"草稿模型"（draft model，如 100M 参数）快速生成 K 个候选 token，然后用大模型（target model）一次验证这 K 个 token。

```
小模型（快）: 生成 K 个 token 的草稿     [~1ms]
大模型（慢）: 一次前向验证 K 个 token     [~10ms，但只用 1 次推理]
             → 接受前 M 个正确的 token     [M ≤ K]
             → 拒绝后 (K-M) 个，大模型重新生成

期望每步接受 token 数 ≈ K × (小模型和大模型的一致性)
```

**为什么有效？**
- 大模型的一次前向传播可以同时验证 K 个 token（不像自回归需要 K 步）
- 如果小模型和大模型高度一致（通常如此），大部分草稿 token 被接受
- 总加速 = 期望接受 token 数 / (1 + K × 小模型推理时间 / 大模型推理时间)

**实际效果**：在代码生成、翻译等预测性强的任务上，可以实现 2-3× 的推理加速，且不改变模型输出的分布（因为只接受大模型认为概率高的 token）。

---

## 9. 基础练习

### 练习 1：实验 Scaling Law
使用第06章的 MiniGPT 框架，训练 3-4 个不同规模的模型（如 5M、10M、20M、50M 参数）。记录每个模型的最终验证 Loss，在双对数坐标上绘制 Loss vs Params 的关系，尝试拟合幂律指数。

### 练习 2：Flash Attention 集成
将第06章 MiniGPT 中的标准 Attention 替换为 `torch.nn.functional.scaled_dot_product_attention`，测量在序列长度 512、1024、2048 下的训练速度和显存占用。对比替换前后的差异。

### 练习 3：实现 KV Cache 推理加速
为 MiniGPT 的 generate 方法添加 KV Cache 支持。测量在生成长度为 50、100、200、500 的文本时，有无 KV Cache 的推理时间差异。绘制时间 vs 长度的关系图。

---

## 10. 进阶练习

### 练习 4：Chinchilla 最优分析
基于公开的模型训练数据（如 Pythia 系列、OPT 系列），收集不同规模模型的训练 token 量和最终 perplexity，分析它们离 Chinchilla 最优有多远，并讨论"过度训练"（如 LLaMA）在实际中的利弊。

### 练习 5：实现 Speculative Decoding
用一个训练好的小模型（如 5M 参数的 MiniGPT）作为 draft model，一个更大的模型（如 50M）作为 target model，实现 Speculative Decoding。测量相对于纯自回归生成的加速比，并分析不同草稿长度 K 对加速比的影响。

---

## 11. 常见错误

### 错误 1：Scaling Law 外推过于乐观
**症状**：小模型实验预测大模型 loss 很低，但实际训练出来 loss 高很多。
**原因**：幂律假设在某些规模下可能不成立，特别是存在"涌现"现象的任务。另一个常见问题是数据分布在大规模时可能不够多样化。
**修复**：在外推时使用保守估计（多拟合几个不同范围的参数，看外推的稳定性）；在多个不同规模上都进行验证实验，而不是只在最小规模做实验。

### 错误 2：Flash Attention 和 Attention Dropout 的兼容性
**症状**：开启 Flash Attention 后模型训练不稳定或结果不对。
**原因**：Flash Attention 通过重计算来避免存储 S 矩阵，但这也意味着 dropout 的随机掩码在反向传播时会被重新生成（不同于前向传播时的掩码）。这破坏了 dropout 的一致性。
**修复**：如果需要 attention dropout，确保 Flash Attention 的实现正确处理了 dropout 的随机种子。大多数现代 LLM（包括 LLaMA）实际上不在 Attention 中使用 dropout。

### 错误 3：混合精度训练中 LayerNorm 仍用 FP16
**症状**：训练中出现 NaN 或 Loss 异常波动。
**原因**：LayerNorm 涉及平方、开方和除法操作，对精度敏感。在 FP16 下，LayerNorm 的中间计算可能溢出或精度不足，导致归一化后的值异常。而 autocast 通常在 LayerNorm 和 Softmax 等操作上自动使用 FP32——但如果手动绕过 autocast，可能会忘记这一点。
**修复**：LayerNorm、Softmax、CrossEntropy 等数值敏感操作保持 FP32。PyTorch 的 autocast 会自动处理，但自定义 CUDA kernel 需要注意。

### 错误 4：Gradient Checkpointing 和需要梯度的非参数变量冲突
**症状**：使用 `torch.utils.checkpoint.checkpoint` 时报错或梯度断裂。
**原因**：Gradient Checkpointing 会丢弃中间激活值，反向时重新计算。但如果模型中有一些非常规的梯度流（如自定义的 `ctx.save_for_backward`），重新计算可能不正确地恢复状态。
**修复**：对简单的 Transformer Block（标准 Attention + FFN + Residual），Gradient Checkpointing 是安全的。如果遇到问题，确保使用 `use_reentrant=False`（PyTorch 1.11+）。

### 错误 5：KV Cache 中忘记处理 batch 中的不同序列长度
**症状**：批量推理时部分序列的输出错乱。
**原因**：在批量推理中，不同请求的生成阶段不同，序列长度不同。如果一个请求已经完成（生成了 EOS），而其他请求还在继续，需要正确地处理"完成"的请求（保持其 KV Cache 但不再更新或将其移除）。
**修复**：维护一个"活动请求"的掩码。对已完成的请求，将其 attention mask 遮蔽为零（或从 batch 中移除并压缩 batch）。

### 错误 6：torch.compile 导致的动态形状问题
**症状**：使用 `torch.compile` 后，首次运行每条序列长度都需要重新编译，导致速度反而更慢。
**原因**：如果输入序列长度频繁变化（如自然语言数据集），每次形状变化都会触发 graph break 和重新编译。
**修复**：使用 `dynamic=True` 参数启用动态形状支持，或填充/裁剪输入到固定形状。在推理中使用 `mode="reduce-overhead"` 或 CUDA Graphs。

### 错误 7：在 Speculative Decoding 中忽略采样一致性
**症状**：Speculative Decoding 的输出分布和原始大模型不同。
**原因**：投机解码的验证阶段需要特殊的采样方法来确保输出分布和原始自回归生成完全一致。标准的实现使用修正采样——只接受那些在原模型分布下概率足够高的草稿 token。如果不做修正，输出分布会偏向草稿模型。
**修复**：使用修正采样。对于每个草稿 token `x_i`，接受概率为 `min(1, p_target(x_i) / p_draft(x_i))`。如果被拒绝，从调整后的分布中重新采样。

---

## 12. 本章小结

本章作为 ai-training 模块的最后一章，覆盖了从理论到实践的两个核心主题：

**Scaling Law**（规模定律）：
1. **Kaplan Law** 揭示了 L(N)、L(D)、L(C) 的幂律关系，但初始的"模型应比数据增长更快"的建议是不最优的。
2. **Chinchilla Law** 纠正了这一点：计算预算最优时，模型大小和数据量应等比放大（约每个参数 20 个 token）。
3. **Scaling Law 的实际价值**在于用小规模实验外推大规模性能，帮助做出"是否值得投入"的决策。

**效率优化**（从训练到推理）：
4. **Flash Attention** 通过分块计算和重计算，将 Attention 的显存复杂度从 O(N²) 降到 O(N)，实现了 2-8× 的实际加速。
5. **混合精度**（BF16/FP16+FP32）在保持训练稳定性的同时，将计算速度提升约 2×，显存占用减少约 50%。
6. **Gradient Checkpointing** 用 20-30% 的额外计算换取 50-70% 的激活值显存节省。
7. **torch.compile** 通过图级优化（算子融合、内存规划）进一步榨取硬件性能。
8. **推理优化**（KV Cache、Continuous Batching、Speculative Decoding）是生产环境部署 LLM 的核心技术。

至此，你已经系统学习了从深度学习训练基础到 Transformer 架构、分词器、数据工程、分布式训练、预训练实战、以及 Scaling Law 和效率优化的完整知识体系。祝你学有所成。

---

文件版本：v1.0 | 最后更新：2026年 | 阅读时间估计：60-90 分钟
