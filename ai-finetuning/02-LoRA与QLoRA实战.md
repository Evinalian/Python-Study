# 第02章 LoRA与QLoRA实战

## 学习目标

完成本章后，你将能够：

1. 用直觉和数学两个层面理解 LoRA（Low-Rank Adaptation）的核心思想
2. 独立配置 LoRA 的参数（rank、alpha、target_modules、dropout）并根据任务调参
3. 使用 HuggingFace PEFT 库加载和配置 LoRA，完成模型微调
4. 理解 QLoRA 的量化机制及其在 4-bit NormalFloat 下的工作原理
5. 对比分析全量微调、LoRA、QLoRA 三者的显存消耗
6. 理解内在维度假设并解释为什么 LoRA 用极少参数就能达到接近全量微调的效果

## 前置知识

- PyTorch 基础：nn.Module、nn.Linear、矩阵乘法、梯度计算（pytorch-core 第01-03章水平）
- 线性代数基础：矩阵的秩、低秩分解、SVD（不需要精通，但理解基本概念会有帮助）
- HuggingFace Transformers 基础使用：加载模型和 tokenizer（pytorch-core 第05章水平）
- 了解全量微调的基本概念（第01章已覆盖）

---

## 1. LoRA 的核心思想：冻结与旁路

### 1.1 问题的起点：为什么不能直接改权重？

想象一下这个场景：你有一个 70 亿参数的预训练模型（LLaMA-7B）。这个模型存储在硬盘上是 14GB（FP16 格式），加载到 GPU 显存中也是 14GB。如果你想全量微调这个模型，你需要额外的显存来存储：

- 梯度：每个参数都需要对应一个梯度值（另外 14GB）
- 优化器状态：Adam 优化器需要为每个参数存储一阶动量（m）和二阶动量（v）（28GB）
- 中间激活值：前向传播时每层的输出需要保留以供反向传播使用（10-20GB）

加在一起，大约需要 60-80GB 的显存。这远超大多数开发者拥有的硬件（一块 RTX 4090 只有 24GB）。就算你有多张 A100，全量微调的时间和电费成本也相当可观。

**问题的本质**：我们真的需要更新全部 70 亿个参数吗？预训练模型已经学会了语言的基本规律——语法、常识、推理链条。大多数下游任务只是要求模型在已有的知识基础上做一些"微调"，而不是推翻重来。

这就引出了 LoRA 的核心洞察。

### 1.2 LoRA 的核心洞察："增量是低秩的"

LoRA 的论文《LoRA: Low-Rank Adaptation of Large Language Models》提出了一个看似大胆但被实验证明正确的假设：

**预训练模型在下游任务上的权重更新（增量）具有低"内在秩"（intrinsic rank）。**

这句话是什么意思？让我们逐步拆解。

在数学中，一个矩阵的"秩"（rank）指的是它的行（或列）的线性无关的维度数。一个满秩的 1000x1000 矩阵需要 1000 个独立的方向才能完整描述。但一个低秩矩阵——比如秩为 8 的 1000x1000 矩阵——虽然表面上有 100 万个元素，但实际上只需要约 8x1000 + 8x1000 = 16000 个"关键参数"就能完全确定。

LoRA 的核心假设是：当你把一个预训练模型适配到下游任务时，权重的变化量 ΔW 不是任意方向的随机变化，而是集中在一个低维子空间内的结构化的变化。换句话说，**模型学习新任务时并不需要调整所有的参数方向，只需要在少数几个关键方向上做调整就够了**。

这个假设已经被大量实验验证。实践中，用 rank=8 或 rank=16 的 LoRA（大约 0.1%-1% 的参数量）往往能达到接近全量微调的效果。

### 1.3 LoRA 的"旁路"架构

LoRA 的设计非常优雅。它不是直接修改原始的权重矩阵 W，而是在旁边"并联"一个低秩的增量模块。

具体来说，对于一个原始的线性层：

```
y = W * x        # W: (d_out, d_in) 的权重矩阵，x: (d_in, 1) 的输入
```

LoRA 添加一个旁路：

```
y = W * x + (B * A) * x        # 绿色部分为 LoRA 增量
```

其中：
- W 是原始的预训练权重，**完全冻结**（不计算梯度、不更新）
- A 是一个 (r, d_in) 的矩阵，将输入从 d_in 维压缩到 r 维
- B 是一个 (d_out, r) 的矩阵，将中间表示从 r 维恢复到 d_out 维
- r 就是 rank（秩），通常远小于 d_in 和 d_out（例如 r=8, d_in=d_out=4096）

**关键观察**：A 和 B 合在一起（B*A）形成一个 (d_out, d_in) 的矩阵——和 W 一模一样的大小。但是！如果直接存储这个 (d_out, d_in) 的增量矩阵，需要 d_in * d_out 个参数。而 LoRA 通过低秩分解，只需要 r * (d_in + d_out) 个参数来间接表达这个增量。

举例说明量的差异：
- 原始 W：4096 * 4096 = 16,777,216 个参数
- 全量微调增量 ΔW：16,777,216 个参数（等于 W）
- LoRA 增量 (r=8)：8 * (4096 + 4096) = 65,536 个参数
- 参数减少比例：16,777,216 / 65,536 = 256 倍

**为什么这个设计如此精妙？**

第一，参数量极大减少。原来需要更新 1600 万个参数，现在只需要更新 6.5 万个——减少了 256 倍。显存消耗也随之大幅下降。

第二，原始权重完好无损。因为你没有修改 W，你随时可以"拔掉" LoRA adapter 恢复到原始模型。这就像给模型装了一个"插件"——装上就是微调后的模型，拔掉就是原始模型。

第三，多任务复用。你可以为每个下游任务训练一个独立的 LoRA adapter（每个只有几 MB），它们共享同一个基座模型。切换任务时只需要换 adapter，不需要重新加载 14GB 的基座模型。

第四，训练效率高。由于只需要更新极少参数，反向传播的梯度计算量也大幅减少，训练速度显著快于全量微调。

### 1.4 一个形象的比喻

如果你觉得上面的数学描述过于抽象，下面这个类比可能有帮助。

把预训练模型想象成一位资深的钢琴家。这位钢琴家已经学会了所有的基础技能——音阶、和弦、演奏技巧、乐理知识。现在，你需要他学习一首新曲子。

- **全量微调**：等同于让钢琴家重新学习每一个手指的每一个动作，尽管大部分动作和之前一样。消耗的时间和精力巨大。
- **LoRA**：等同于给钢琴家一份"调整清单"——"第 3 小节把 C 大调改成 G 大调"、"第 8 小节 tempo 放慢 10%"。调整清单只有一页纸（低秩），但足以让他的演奏焕然一新。

---

## 2. LoRA 的数学原理：深入低秩分解

### 2.1 低秩分解的形式化定义

LoRA 的形式化定义很简单。对于预训练权重矩阵 W0 （d_out, d_in），LoRA 将其更新约束为低秩分解形式：

```
W = W0 + ΔW = W0 + B * A
```

其中：
- W0 保持不变（frozen）
- B  (d_out, r)
- A  (r, d_in)
- r << min(d_out, d_in)

前向传播变为：

```
h = W0 * x + B * A * x * (alpha / r)
```

注意这里多了一个缩放因子 `alpha / r`。这是 LoRA 设计中的一个细节：
- `alpha` 是一个缩放超参数，控制 LoRA 增量对原始输出的影响强度
- `r` 是 rank
- 使用 `alpha / r` 作为缩放因子意味着：当你增加 rank 时，缩放因子自动减小，使得整体更新的"幅度"保持相对稳定

### 2.2 为什么低秩分解"够用"——内在维度假设

你可能会问：凭什么认为 ΔW 一定是低秩的？如果 ΔW 实际上需要很高的秩怎么办？

这是一个非常好的问题。LoRA 论文中的"内在维度假设"（Intrinsic Dimension Hypothesis）给出了答案。

内在维度假设指出：深度神经网络虽然生活在高维参数空间中，但优化过程实际上只发生在一个低维流形上。也就是说，虽然理论上有 1600 万个参数可以调整，但真正需要调整来适配新任务的"方向"只有少数几个。

这个假设并非凭空想象。之前的研究表明：
- 对于图像分类任务，即使随机冻结 90% 的参数，模型仍然可以达到不错的准确率
- 很多预训练任务只需要几百个"内在维度"就能得到合理的解
- 使用 SVD 分析预训练权重的更新，发现前几个奇异值包含了大部分的更新信息

LoRA 的设计正是基于这个假设的实验验证。在实践中，即使 r 只有 4 或 8，LoRA 也能在大多数下游任务上达到接近全量微调的效果。这说明绝大多 数情况下，"内在维度"确实相当低。

**但注意**：内在维度假设并非在所有任务上都成立。对于需要注入大量新领域知识的任务（比如完全学习一种新的语言、学习全新的编程范式），可能需要更高的 rank（64-256），甚至可能需要全量微调。

### 2.3 A 和 B 的初始化策略

LoRA 中 A 和 B 的初始化方式直接影响到训练的稳定性和收敛速度：

- **A 的初始化**：使用随机高斯初始化（均值为 0，标准差为 1/sqrt(r) 或使用 Kaiming uniform）。这样做的目的是让 A 一开始包含丰富的方向信息。
- **B 的初始化**：使用零初始化。这样做的目的是让 ΔW = B * A 在训练开始时等于零矩阵，确保模型从原始的预训练行为出发，不会因为随机的 LoRA 初始化而崩溃。

这种初始化组合（A 随机、B 全零）非常巧妙：
- 训练开始时：LoRA 不影响原始模型（因为 B=0，所以 ΔW=0）
- 训练过程中：梯度回传让 A 和 B 逐步学习到合适的增量
- 训练结束后：B * A 形成了一个有意义的低秩更新

---

## 3. LoRA 的配置参数详解

在实际使用 LoRA 时，你需要配置几个关键参数。每个参数的选择都直接影响微调效果和资源消耗。

### 3.1 rank (r)

**含义**：低秩分解的秩。决定了 A 和 B 矩阵的中间维度。

**对效果的影响**：
- rank 越大，LoRA 的可训练参数越多（线性增长），表达能力越强
- rank 太小（如 r=1 或 r=2），可能无法捕捉足够的任务特征，微调效果有限
- rank 太大（如 r=256），收益递减——效果不再明显提升，但参数量和训练时间却线性增加

**实践经验**：
- r=4 或 r=8：适用于简单的风格/格式适配，效果往往已经不错
- r=16 或 r=32：适用于大多数通用的指令微调任务
- r=64 或 r=128：适用于较复杂的领域知识注入，或对效果要求极高的场景
- r=256+：除非你有明确的实验证据表明需要这么高的秩，否则通常不需要

**选型建议**：从 r=8 开始实验。如果效果不达标，逐步增加到 r=16、r=32。注意观察每次增加的边际收益——如果从 r=8 到 r=16 效果有明显提升，但从 r=16 到 r=32 几乎没有变化，说明 r=16 已经足够。

### 3.2 alpha

**含义**：LoRA 增量的缩放系数。前向传播时实际使用的缩放因子是 alpha/r。

**对效果的影响**：
- alpha 越大，LoRA 增量的影响越大。如果 alpha 太大，模型可能过度偏离原始行为
- alpha 太小，LoRA 增量的影响太小，微调效果不明显
- 常见的设置是 alpha = 2 * r（即实际缩放因子为 2）或 alpha = r（实际缩放因子为 1）

**实践经验**：
- alpha = 16（配合 r=8）：常用组合，提供适中的 LoRA 影响强度
- alpha = 32（配合 r=16）：影响更强，适合需要较大行为改变的场景
- 调整 alpha 等同于调整学习率的大小——alpha 越大，每次参数更新对模型输出的影响越大

**重要**：不要被"alpha/r 是缩放因子"这个公式迷惑。alpha 和 r 不是独立设置的——改变 r 时通常也要相应调整 alpha。一个稳定的做法是保持 alpha/r 的比例固定（比如恒为 2），然后通过调整学习率来控制训练强度。

### 3.3 target_modules

**含义**：指定对哪些模块（层）应用 LoRA。

**对效果的影响**：
- 只对 attention 层的 Q 和 V 矩阵应用 LoRA：这是 LoRA 论文中最基础的设置，可训练参数最少，效果也能接受
- 对 attention 层的 Q、K、V、O 全部应用：显著增加可训练参数，通常能提升效果
- 额外对 FFN 层（MLP）应用 LoRA：进一步增加参数量和表达能力

**实践经验**：
- 对于 LLaMA/Qwen 等模型，常见的 target_modules 包括：
  - `["q_proj", "v_proj"]`（最基础，参数最少）
  - `["q_proj", "k_proj", "v_proj", "o_proj"]`（推荐，效果和参数量的较好平衡）
  - `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`（覆盖 attention + FFN，参数最多）
- 不确定时，可以使用 PEFT 库的辅助函数自动查找模型中的线性层名称

### 3.4 lora_dropout

**含义**：在 LoRA 层的 A 矩阵前应用的 dropout 概率。

**对效果的影响**：
- dropout 是一种正则化手段，随机"丢弃"一部分神经元的激活值，防止过拟合
- 对于数据量较小（<1000 条）的微调场景，dropout 尤其重要
- 对于数据量充足（>10000 条）的场景，可以设很小的 dropout 或直接设为 0

**实践经验**：
- lora_dropout = 0.1：常用值，提供适中的正则化
- lora_dropout = 0.05：轻量正则化，适合数据量较多的场景
- lora_dropout = 0：不使用 dropout，仅在数据量非常充足时考虑

---

## 4. QLoRA：在 LoRA 基础上进一步压缩

### 4.1 为什么还需要 QLoRA

LoRA 已经大幅减少了可训练参数量。但这里有一个容易被忽视的问题：虽然 LoRA 不需要更新原始权重 W，但 W 本身仍然需要加载到显存中（用于前向传播计算）。对于一个 7B 模型，FP16 格式的原始权重占据约 14GB 显存。这对于消费级 GPU（如 RTX 3060 12GB、RTX 4070 12GB）来说仍然太重了。

QLoRA 解决的就是这个问题：**在 LoRA 的基础上，把冻结的原始权重也量化到 4-bit，进一步压缩显存占用**。

### 4.2 QLoRA 的核心技术

QLoRA 综合运用了以下几个关键技术：

**4-bit NormalFloat (NF4)**：

这是 QLoRA 论文中提出的一种专门为神经网络权重量化设计的数据格式。常规的 4-bit 整数量化（INT4）均匀地划分值域空间，但神经网络的权重分布通常是近似正态分布的（集中在零附近）。NF4 根据正态分布的累积分布函数来非均匀地划分值域空间，使得量化误差最小。

简单理解：NF4 在"权重密集"的区域（靠近零）使用更细的量化格子，在"权重稀疏"的区域（远离零）使用更粗的量化格子，从而在保持 4-bit 表示的前提下最大化信息保真度。

**双重量化（Double Quantization）**：

量化一个权重矩阵需要额外的"量化常数"（scaling constants）。这些常数本身也是 FP32 格式的，虽然相对权重来说很小（通常每个 block 一个 scaling factor），但对于超大模型来说累积起来也不小。双重量化就是对量化常数再做一次量化——将 FP32 的 scaling factors 进一步压缩到 FP8。对于 7B 模型，这大约节省 0.4GB 显存。

**分页优化器（Paged Optimizer）**：

在训练过程中，优化器状态（梯度的一阶和二阶动量）可能占用大量显存。当显存不足时，系统会抛出 OOM（Out of Memory）错误。分页优化器借鉴了操作系统中"分页"的思想：当 GPU 显存紧张时，自动将优化器状态的一部分卸载到 CPU 内存中，等需要时再加载回来。这允许你在显存稍微不够的情况下仍能完成训练（代价是速度变慢）。

### 4.3 量化与反量化的推理过程

QLoRA 的前向传播过程值得仔细理解，因为它涉及一个"量化-反量化"的循环：

1. **存储**：原始权重 W 以 NF4 格式存储（占用极少显存）
2. **前向传播时**：将 NF4 权重反量化回 FP16（或 BF16）
3. **计算**：用反量化后的 FP16 权重进行矩阵乘法
4. **LoRA 增量**：LoRA 的 A 和 B 始终在 FP16/BF16 精度下计算
5. **合并**：y = W_dequantized * x + B * A * x
6. **反向传播**：梯度只流经 LoRA 的 A 和 B（保持 FP16/BF16），不流经量化的 W

关键在于：**反量化只在每次前向传播时临时进行**。W 在显存中始终以 4-bit 存储，只是计算时临时转换为 FP16。这意味着你不必永久性地将 14GB 的 FP16 权重存在显存中——只需要约 3.5GB 的 NF4 权重。

### 4.4 NF4 的非均匀量化原理

为了理解 NF4 的优越性，让我们对比两种量化方式：

**均匀量化（INT4）**：
```
值域 [-1.0, 1.0] → 均匀划分为 16 个区间
区间边界: -1.0, -0.875, -0.75, ..., 0.75, 0.875, 1.0
量化级别: 0-15
```
问题：神经网络权重 80% 以上落在 [-0.25, 0.25] 之间，但均匀量化在这个密集区只有 4 个区间。大部分"量化精度"浪费在了几乎没有权重落在的区域。

**NF4 量化**：
```
根据正态分布的分位数来划分 16 个区间
在 [-0.25, 0.25] 区间分配更密集的格点
在 [-1.0, -0.75] 和 [0.75, 1.0] 区间分配更稀疏的格点
```
这样，在权重最密集的区域，量化精度最高，整体量化误差最小。

---

## 5. PEFT 库实战：完整的 LoRA/QLoRA 微调

### 5.1 环境准备

首先安装必要的依赖：

```bash
pip install transformers accelerate peft bitsandbytes datasets torch
```

其中：
- `transformers`：HuggingFace 模型库，提供预训练模型的加载和管理
- `accelerate`：分布式训练和混合精度训练的抽象层
- `peft`：Parameter-Efficient Fine-Tuning，提供 LoRA/QLoRA 等高效微调方法
- `bitsandbytes`：提供 4-bit/8-bit 量化功能（QLoRA 的依赖）
- `datasets`：HuggingFace 数据集库，用于高效加载和处理训练数据
- `torch`：PyTorch 深度学习框架

### 5.2 完整的 LoRA 微调代码

下面的代码演示了使用 HuggingFace PEFT 库加载模型、配置 LoRA、准备数据并进行微调的完整流程。每一行都会用注释解释其作用。

```python
"""
使用 HuggingFace PEFT 库进行 LoRA 微调的完整示例。

演示流程:
1. 加载基座模型和 tokenizer
2. 配置 LoRA 参数
3. 准备微调数据
4. 使用 Trainer 进行训练
5. 保存和加载 LoRA adapter
"""
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
from datasets import Dataset


# ============================================================
# 1. 加载基座模型和 tokenizer
# ============================================================

# 模型名称（可以从 HuggingFace Hub 加载，也可以从本地路径加载）
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # 使用小模型便于演示
# MODEL_NAME = "Qwen/Qwen2.5-7B"  # 较大的模型用于实际微调

print(f"加载模型: {MODEL_NAME}")

# 加载 tokenizer
# use_fast=True 使用 Rust 实现的快速 tokenizer
# trust_remote_code=True 允许加载自定义模型代码（Qwen 等模型需要）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    use_fast=True,
)

# 设置 pad_token（如果模型没有定义，使用 eos_token 作为 pad_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载模型
# torch_dtype=torch.bfloat16 使用 BF16 精度以节省显存（BF16 和 FP16 对比：
#   BF16 范围更大但精度稍低，FP16 精度稍高但范围较小；大模型训练中 BF16 更稳定）
# device_map="auto" 自动将模型分配到可用的 GPU 上
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"模型加载完成。参数量: {model.num_parameters():,}")


# ============================================================
# 2. 配置 LoRA
# ============================================================

# LoRA 配置
lora_config = LoraConfig(
    # task_type: 任务类型
    #   TaskType.CAUSAL_LM: 因果语言模型（GPT 系列、LLaMA、Qwen 等）
    #   TaskType.SEQ_2_SEQ_LM: 序列到序列模型（T5、BART 等）
    #   TaskType.SEQ_CLS: 序列分类任务
    task_type=TaskType.CAUSAL_LM,

    # r (rank): 低秩分解的秩
    #   决定了 LoRA 的可训练参数量。r 越大表达能力越强，但参数也越多。
    #   常用值: 4, 8, 16, 32, 64
    r=8,

    # lora_alpha: LoRA 增量的缩放系数
    #   实际缩放因子 = lora_alpha / r
    #   这里设为 16，配合 r=8，实际缩放因子为 2
    lora_alpha=16,

    # target_modules: 对哪些模块应用 LoRA
    #   需要根据模型架构来选择。可以使用如下方法查看模块名:
    #   for name, _ in model.named_modules(): print(name)
    #   对于 Qwen2.5 系列，常见的线性层名称如下:
    target_modules=[
        "q_proj",    # Query 投影
        "k_proj",    # Key 投影
        "v_proj",    # Value 投影
        "o_proj",    # Output 投影
        "gate_proj", # FFN 门控投影（SwiGLU 架构）
        "up_proj",   # FFN 上投影
        "down_proj", # FFN 下投影
    ],

    # lora_dropout: LoRA 层的 dropout 概率
    #   用于防止过拟合。数据量少时应该设大一些（0.1-0.2）
    #   数据量充足时可设为 0.05 或 0
    lora_dropout=0.1,

    # bias: 是否训练 bias 项
    #   "none": 不训练任何 bias（推荐，参数量最少）
    #   "all": 训练所有 bias
    #   "lora_only": 只训练 LoRA 模块中的 bias
    bias="none",

    # inference_mode: 是否以推理模式加载（False 表示我们要训练）
    inference_mode=False,
)

# 将 LoRA 配置应用到模型
# get_peft_model 会：
# 1. 冻结所有原始参数（requires_grad=False）
# 2. 在指定的 target_modules 上添加 LoRA adapter
# 3. 只让 LoRA adapter 的参数可训练
model = get_peft_model(model, lora_config)

# 打印可训练参数信息
model.print_trainable_parameters()
# 输出示例: trainable params: 6,815,744 || all params: 494,032,768 || trainable%: 1.38%


# ============================================================
# 3. 准备微调数据
# ============================================================

def format_instruction(example: dict) -> dict:
    """
    将数据格式化为 Chat Template 并 tokenize。

    每条数据包含:
    - instruction: 用户指令
    - output: 期望回复
    - system (可选): 系统指令
    """
    system_content = example.get("system", "你是一个有帮助的AI助手。")
    user_content = example.get("instruction", example.get("user", ""))
    assistant_content = example.get("output", example.get("response", example.get("assistant", ""))

    # 构建 messages 列表
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    # 使用 tokenizer 的 apply_chat_template 方法
    # 这会自动应用模型正确的 Chat Template（如 Qwen 的 <|im_start|>...<|im_end|>）
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,         # 先返回字符串（不转 token）
        add_generation_prompt=False,  # 不添加生成提示（因为我们已经提供了完整的回复）
    )

    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,          # 超过 max_length 截断
        max_length=2048,          # 最大序列长度
        padding=False,            # 不在这里 padding（由 DataCollator 统一处理）
        return_tensors=None,      # 返回 Python 列表而非 tensor
    )

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


# 准备示例训练数据
# 实际使用时从 JSONL 文件加载（参见第01章的数据准备 Pipeline）
train_data = [
    {
        "instruction": "Python 中如何读取文件内容？",
        "output": "在 Python 中，你可以使用 open() 函数和 read() 方法来读取文件内容。基本用法如下：\n\n```python\nwith open('file.txt', 'r', encoding='utf-8') as f:\n    content = f.read()\n    print(content)\n```\n\nwith 语句确保文件在使用后被正确关闭，编码指定为 utf-8 避免中文乱码。"
    },
    {
        "instruction": "解释一下什么是梯度下降算法",
        "output": "梯度下降（Gradient Descent）是机器学习中最基础的优化算法。其核心思想是：沿着损失函数梯度的反方向更新参数，从而逐步减小损失值。\n\n具体来说，对于参数 θ，每次更新：θ = θ - η * ∇L(θ)\n\n其中：\n- η（学习率）：控制每次更新的步长\n- ∇L(θ)：损失函数 L 对参数 θ 的梯度，指向函数增长最快的方向\n\n梯度下降的变体包括：\n1. 批量梯度下降（BGD）：使用全部数据计算梯度\n2. 随机梯度下降（SGD）：每次只用一个样本\n3. 小批量梯度下降（Mini-batch SGD）：折中方案，最常用"
    },
    {
        "instruction": "请翻译为英文：人工智能正在改变世界。",
        "output": "Artificial intelligence is changing the world."
    },
    # ... 实际使用时会有成百上千条数据
]

# 转换为 HuggingFace Dataset
train_dataset = Dataset.from_list(train_data)

# 应用格式化函数
train_dataset = train_dataset.map(format_instruction, remove_columns=train_dataset.column_names)

print(f"训练数据准备完成。样本数: {len(train_dataset)}")
print(f"第一个样本的 input_ids 长度: {len(train_dataset[0]['input_ids'])}")


# ============================================================
# 4. 配置 Trainer 并开始训练
# ============================================================

# 训练参数配置
training_args = TrainingArguments(
    # 输出目录（保存 checkpoint 和最终模型）
    output_dir="./lora_output",

    # 训练轮数
    #   指令微调通常 3-5 个 epoch 即可，多了容易过拟合
    num_train_epochs=3,

    # 每个设备（GPU）上的 batch size
    #   值越大训练越稳定但显存需求越高
    #   对于 7B 模型 + LoRA，per_device_train_batch_size=4 通常可以在 24GB GPU 上运行
    per_device_train_batch_size=2,

    # 梯度累积步数
    #   "有效 batch size" = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
    #   如果单卡 batch size 太小，可以用梯度累积来模拟更大的 batch
    gradient_accumulation_steps=4,

    # 学习率
    #   微调的学习率应该远小于预训练（通常是 1e-5 到 5e-5）
    #   LoRA 的学习率可以比全量微调稍大一些（因为它只更新少量参数）
    learning_rate=2e-4,  # LoRA 常用 1e-4 到 5e-4

    # 预热步数（warmup）
    #   训练开始时学习率从 0 线性增加到目标值
    #   预热可以让训练更稳定，避免初始梯度爆炸
    warmup_steps=100,

    # 日志记录间隔
    logging_steps=10,

    # 模型保存策略
    save_strategy="epoch",    # 每个 epoch 结束时保存
    save_total_limit=2,       # 只保留最近 2 个 checkpoint（节省磁盘空间）

    # 精度设置
    bf16=True,                # 使用 BF16 混合精度训练（需要 GPU 支持）
    # fp16=True,              # 如果 GPU 不支持 BF16，使用 FP16

    # AdamW 优化器参数
    optim="adamw_torch",      # 使用 PyTorch 原生的 AdamW
    weight_decay=0.01,        # 权重衰减（L2 正则化），防止过拟合

    # 学习率调度器
    lr_scheduler_type="cosine",  # 余弦退火：学习率按余弦曲线逐渐减小

    # 数据加载
    dataloader_num_workers=2,    # 数据加载的子进程数
    remove_unused_columns=False, # 不删除未使用的列（保留 input_ids 等）

    # 报告
    report_to="none",            # 不上报到 wandb/tensorboard（设为 "tensorboard" 可启用）
)

# Data Collator
# DataCollatorForLanguageModeling 负责：
# 1. 将不等长的序列 padding 到同一长度
# 2. 创建 labels（对于因果语言模型，labels 就是 input_ids 右移一位）
# 3. 将 padding token 的 label 设为 -100（忽略 loss 计算）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # 不使用 Masked Language Modeling（我们是因果 LM）
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 开始训练
print("开始 LoRA 微调训练...")
trainer.train()

# 保存 LoRA adapter
# 注意：trainer.save_model() 只保存 LoRA adapter 权重（几 MB 到几十 MB），
# 不保存基座模型（几个 GB）。推理时需要先加载基座模型，再加载 adapter。
trainer.save_model("./lora_adapter_final")
tokenizer.save_pretrained("./lora_adapter_final")
print("LoRA adapter 已保存到 ./lora_adapter_final")


# ============================================================
# 5. 加载 LoRA adapter 进行推理
# ============================================================

def load_lora_model(base_model_name: str, adapter_path: str) -> PeftModel:
    """
    加载基座模型并挂载 LoRA adapter。

    注意：推理时也只需要加载 adapter（几 MB），
    基座模型本身保持原始权重。
    """
    # 加载基座模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 挂载 LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # 切换到推理模式（合并 LoRA 权重以避免推理时的额外计算）
    # model = model.merge_and_unload()  # 将 LoRA 权重合并回基座模型

    return model


def generate_response(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    instruction: str,
    system_prompt: str = "你是一个有帮助的AI助手。",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """
    使用微调后的模型生成回复。

    参数:
        model: 挂载了 LoRA adapter 的模型
        tokenizer: 分词器
        instruction: 用户指令
        system_prompt: 系统指令
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度（越高越随机）
    """
    # 构建 messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    # 应用 Chat Template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 添加生成提示，让模型知道接下来要生成回复
    )

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    # 将 input 移到模型所在的设备上
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,        # 启用采样（而非贪婪解码）
            top_p=0.9,             # nucleus sampling
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码（只取生成的新 token，不包括输入的 token）
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response


# 测试微调效果
print("\n===== 微调效果测试 =====")
loaded_model = load_lora_model(MODEL_NAME, "./lora_adapter_final")

test_questions = [
    "Python 中如何处理异常？",
    "什么是深度学习中的反向传播？",
]

for q in test_questions:
    print(f"\n问题: {q}")
    answer = generate_response(loaded_model, tokenizer, q)
    print(f"回答: {answer}")
    print("-" * 50)
```

### 5.3 关键代码细节解读

上面这段代码虽然看起来很长，但每个部分都有其设计理由。让我们重点解读几个容易出错的细节：

**关于 `device_map="auto"`**：
这个参数让 `accelerate` 库自动决定把模型的不同部分放在哪里。如果你的 GPU 显存足够放下整个模型，它会全部放在 GPU 上。如果显存不够，它会自动把一些层卸载到 CPU 内存中。这对于在显存有限的设备上加载大模型非常有用。但要注意，CPU 上的计算非常慢，如果你的模型大部分被卸载到了 CPU，训练速度会急剧下降。

**关于 `apply_chat_template`**：
这是 tokenizer 提供的一个关键方法。不同的模型使用不同的特殊 token 和格式（LLaMA 用 `[INST]`，Qwen 用 `<|im_start|>`），手动拼接很容易出错。`apply_chat_template` 从 tokenizer 的配置文件中自动读取正确的模板并应用。**永远不要手动拼接 Chat Template 字符串**——用这个方法就对了。

**关于训练时的 `labels`**：
对于因果语言模型（Causal LM），训练目标是"预测下一个 token"。所以 `labels` 就是 `input_ids` 右移一位。`DataCollatorForLanguageModeling` 会自动处理这个。它会将 padding token 位置的 label 设为 -100（PyTorch 的 CrossEntropyLoss 默认忽略 label=-100 的位置）。

**一个重要的限制**：上面的代码中对所有 token 都计算了 loss，包括 system prompt 和 user 部分的 token。这意味着模型也在学习"生成 system prompt 和 user 输入"——这不合理。实际上我们只希望模型学习"生成 assistant 回复"部分。这就是第03章要讲的 **Loss Masking** 技术。

---

## 6. QLoRA 实战：4-bit 量化微调

### 6.1 QLoRA 的配置

要将上面的 LoRA 代码改为 QLoRA，只需要在加载模型时加上量化配置：

```python
"""
QLoRA 微调的核心变化：在加载基座模型时使用 4-bit 量化。

与 LoRA 的区别：
- 基座模型权重以 NF4 格式存储（显存约为 FP16 的 1/4）
- 前向传播时临时反量化到 BF16 进行计算
- LoRA adapter 仍在 BF16 精度下训练
- 需要 bitsandbytes 库（pip install bitsandbytes）
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,  # QLoRA 的关键：量化配置
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,  # QLoRA 需要：为 k-bit 训练做准备
)
from datasets import Dataset


# ============================================================
# QLoRA 的关键：4-bit 量化配置
# ============================================================

# BitsAndBytesConfig 定义了量化策略
bnb_config = BitsAndBytesConfig(
    # load_in_4bit=True: 使用 4-bit 量化加载模型权重
    #   权重以 NF4 格式存储，显存消耗约为 FP16 的 1/4
    load_in_4bit=True,

    # bnb_4bit_compute_dtype: 计算时使用的数据类型
    #   虽然权重存储为 4-bit，但前向传播计算时先反量化到这个精度
    #   torch.bfloat16 是推荐选择（范围大，训练稳定）
    bnb_4bit_compute_dtype=torch.bfloat16,

    # bnb_4bit_quant_type: 量化数据类型
    #   "nf4": NormalFloat 4-bit（推荐，QLoRA 论文中提出的格式）
    #   "fp4": 标准的 4-bit 浮点数
    bnb_4bit_quant_type="nf4",

    # bnb_4bit_use_double_quant: 是否使用双重量化
    #   True: 对量化常数再做一次量化，节省约 0.4GB 显存（推荐开启）
    #   False: 不做双重量化
    bnb_4bit_use_double_quant=True,
)

# 使用量化配置加载模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",           # 7B 模型，FP16 约 14GB
    quantization_config=bnb_config,  # 使用 NF4 量化，约 3.5-4GB
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# ============================================================
# QLoRA 的额外步骤：prepare_model_for_kbit_training
# ============================================================

# prepare_model_for_kbit_training 做了什么？
# 1. 冻结量化后的权重（它们已经通过 bitsandbytes 的量化层管理）
# 2. 启用梯度检查点（gradient checkpointing）：用计算换显存，
#    不在前向传播时保存全部中间激活值，而是在反向传播时重新计算
# 3. 将某些需要高精度的操作（如 LayerNorm）转为 FP32
#    因为 4-bit 精度对于归一化操作来说太粗糙了
model = prepare_model_for_kbit_training(model)

# ============================================================
# LoRA 配置（与普通 LoRA 完全相同）
# ============================================================

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,              # QLoRA 中可以设稍大的 rank（因为基座模型更小）
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 后续的训练流程与普通 LoRA 完全相同
# （TrainingArguments、Trainer 等保持不变）
```

### 6.2 QLoRA 的注意事项

**训练速度**：QLoRA 比 LoRA 慢。因为每次前向传播都需要将 NF4 权重反量化到 BF16，这增加了计算开销。但速度的下降通常是可以接受的（慢 20-30%），而显存的节省是巨大的（节省约 70-75%）。

**精度损失**：4-bit 量化不可避免地会有精度损失。但在实践中，这种损失对最终微调效果的影响很小（通常 <1% 的性能差异）。对于大多数微调场景来说，用一点精度损失换取能在消费级 GPU 上运行的可能性，是非常值得的。

**不适用场景**：如果你的 GPU 显存本就足够以 FP16/BF16 加载模型（比如你有 A6000 48GB 跑 7B 模型），那么不需要 QLoRA——直接 LoRA 即可，精度更好且速度更快。

---

## 7. 显存消耗对比

### 7.1 三种方案的理论对比

下面以外 LLaMA-7B 为例，分析三种微调方案的显存消耗（估算值，实际消耗受 batch size、序列长度、梯度检查点等因素影响）：

| 组件 | 全量微调 | LoRA (FP16) | QLoRA (NF4) |
|------|---------|------------|------------|
| 模型权重 | 14 GB (FP16) | 14 GB (FP16) | 3.5 GB (NF4) |
| LoRA 可训练参数 | N/A | 0.05-0.2 GB | 0.05-0.2 GB |
| 梯度 | 14 GB | 0.05-0.2 GB | 0.05-0.2 GB |
| 优化器状态 | 28 GB (Adam) | 0.1-0.4 GB | 0.1-0.4 GB |
| 激活值 (batch=2, seq=2048) | ~12 GB | ~12 GB | ~12 GB |
| 额外开销 | ~4 GB | ~2 GB | ~2 GB |
| **总计** | **~72 GB** | **~28 GB** | **~18 GB** |
| **推荐 GPU** | 4x A100 80GB | 1x A100 40GB | 1x RTX 3090/4090 |

可以看到：
- 全量微调需要 72GB 显存，至少需要多张顶级 GPU
- LoRA 只需要 28GB，一张 A100 40GB 就能跑
- QLoRA 只需要 18GB，一张 RTX 4090 24GB 就能跑 7B 模型

### 7.2 影响显存的其他因素

**序列长度**：序列越长，注意力矩阵的大小呈平方增长。从 2048 token 增加到 4096 token，激活显存可能增加 2-4 倍。如果你的训练数据中包含长文本，务必截断到合理长度。

**Batch Size**：每个 batch 中的序列越多，需要的激活值显存越大。如果显存不够，减小 batch size 并增加梯度累积步数是标准做法。

**梯度检查点（Gradient Checkpointing）**：用计算换显存。开启后，不在前向传播时保存中间激活值，反向传播时重新计算。这可以减少约 60-70% 的激活值显存，但训练速度会下降约 20-30%。

**Flash Attention**：一种高效的注意力计算实现，可以将注意力计算的显存复杂度从 O(N^2) 降到 O(N)。现代 transformers 库已经默认集成了 Flash Attention。

---

## 基础练习

1. **LoRA 配置与加载**：使用 PEFT 库加载 Qwen2.5-0.5B 模型，配置 LoRA（r=8, alpha=16, 覆盖 attention 层），打印可训练参数比例，理解每种 target_module 对参数量的贡献。

2. **计算 LoRA 参数量**：编写一个函数，接受模型隐藏维度 d、rank r、target_modules 列表，计算理论上 LoRA 的可训练参数量，并与 PEFT 库的输出做对比，验证你的理解。

3. **QLoRA 量化配置**：对比加载同一个模型在 FP16、INT8、NF4 三种精度下的显存占用，编写脚本自动测量并输出对比结果。

## 进阶练习

1. **显存消耗对比实验**：编写一个自动化脚本，测试不同配置（全量微调/LoRA/QLoRA，不同 rank，不同 batch size）下的实际显存消耗和训练吞吐量，生成对比表格和图表。

2. **不同 rank 的效果对比**：在同一数据集上，分别用 r=4, r=8, r=16, r=32, r=64 进行 LoRA 微调，用 eval loss 和生成质量来评估效果差异，找到该任务的最佳 rank。

---

## 常见错误

1. **忘记设置 pad_token**：很多基座模型（尤其是 LLaMA）没有定义 pad_token。如果不设置（通常设为 eos_token），tokenizer 在 padding 时会报错或使用错误的 pad token id。设置方式：`tokenizer.pad_token = tokenizer.eos_token`。

2. **target_modules 写错**：不同模型的模块命名不同。LLaMA 用的是 `q_proj`，ChatGLM 用的是 `query_key_value`。写错名字会导致 LoRA 没有应用到任何模块，训练没有任何效果。建议先用 `model.named_modules()` 查看实际的模块名。

3. **学习率设得太大**：很多人习惯预训练的 learning rate（1e-4 到 1e-3），但微调时这个范围太大了。LoRA 微调推荐的学习率范围是 1e-4 到 5e-4，全量微调更低（1e-5 到 5e-5）。太大的学习率会导致模型迅速偏离原始行为，甚至 loss 爆炸。

4. **QloRA 中忘记调用 `prepare_model_for_kbit_training()`**：这个方法不是可选的。如果不调用它，量化模型的梯度计算会出问题，训练可能根本不收敛或抛出错误。

5. **BF16 和 FP16 混用**：模型用 BF16 加载，但 TrainingArguments 中设置了 `fp16=True`（或反过来）。这会导致数据类型不匹配的错误。确保模型加载的 dtype 和训练参数的精度设置一致。

6. **忽略 LoRA adapter 的保存方式**：`model.save_pretrained()` 只保存 LoRA adapter 权重（几 MB），不保存基座模型。推理时正确的加载方式是：先加载基座模型，再用 `PeftModel.from_pretrained(base_model, adapter_path)` 挂载 adapter。如果直接把 adapter 当作完整模型加载会报错。

7. **对不合适的层应用 LoRA**：LoRA 通常只应用于线性层（nn.Linear）。如果对 Embedding 层或 LayerNorm 层应用 LoRA，不仅浪费参数，还可能破坏这些层的正常功能。

8. **数据没有正确应用 Chat Template**：手动拼接字符串而不是用 `tokenizer.apply_chat_template()`。结果训练时的格式和推理时不一致（train-inference mismatch），模型行为异常。永远使用 `apply_chat_template`。

9. **在 QLoRA 训练时启用了 FSDP 或 DeepSpeed ZeRO-3**：bitsandbytes 的量化层与某些分布式策略不兼容。如果在多卡训练中使用 QLoRA，需要确认分布式策略与量化兼容。

10. **忽略 `AdaLoRA` 和 `IA3` 等其他 PEFT 方法**：虽然 LoRA 是最主流的，但 PEFT 库还提供了其他高效微调方法。对于特定场景，AdaLoRA（自适应分配 rank）或 IA3（更少参数）可能更合适。不要过早锁定在 LoRA 上。

---

## 本章小结

本章我们从 LoRA 的核心洞察——"增量是低秩的"——出发，用直觉（钢琴家的调整清单）和数学（低秩分解 B*A）两个层面详细解释了 LoRA 的工作原理。我们深入探讨了内在维度假设：为什么用如此少的参数就能达到接近全量微调的效果。

在实践部分，我们逐一拆解了 LoRA 的四个配置参数（rank、alpha、target_modules、dropout），给出了基于经验的选值建议。然后通过完整的代码示例展示了使用 HuggingFace PEFT 库进行 LoRA 微调的全流程：加载模型、配置 LoRA、准备数据、训练、保存和推理。

QLoRA 部分我们深入探讨了 NF4 量化格式的设计原理（非均匀量化）、双重量化和分页优化器的机制，理解了为什么 QLoRA 能让 7B 模型在 24GB 消费级 GPU 上微调成为可能。

最后，我们通过显存消耗的详细对比表，直观地展示了全量微调、LoRA、QLoRA 三种方案在不同资源条件下的适用性。

下一章，我们将学习指令微调（SFT）的完整训练流程，重点掌握 Loss Masking 技术和过拟合检测方法。
