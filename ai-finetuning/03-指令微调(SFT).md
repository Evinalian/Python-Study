# 第03章 指令微调 (SFT)

## 学习目标

完成本章后，你将能够：

1. 理解指令微调（SFT）的目标：让模型学会"理解并遵循指令"的交互模式
2. 掌握 Chat Template 的完整规范，正确构建训练和推理格式
3. 独立完成 SFT 训练流程：数据加载、tokenization、loss masking、训练、评估
4. 理解 Loss Masking 的原理和实现——为什么只对 assistant 回复计算 loss
5. 识别过拟合的早期信号，掌握验证集评估和生成质量抽查方法
6. 从多个 checkpoint 中选择最优模型，而不是盲目使用最后一个

## 前置知识

- LoRA/QLoRA 的基本概念和使用（第02章水平）
- HuggingFace Transformers 和 PEFT 库的基础使用（第02章已覆盖）
- Chat Template 的概念和结构（第01章已覆盖）
- PyTorch 训练循环的基本理解（pytorch-core 第04章水平）

---

## 1. 指令微调的目标

### 1.1 基座模型 vs 指令微调模型

让我们从最根本的问题开始：为什么需要指令微调？

你有一个预训练的因果语言模型（比如 LLaMA-7B）。这个模型在几万亿 token 的互联网文本上做了预训练，它的核心能力是"文本补全"——给定一段文本，预测下一个最可能出现的 token。

但"文本补全"和"执行指令"是完全不同的两件事。如果你给原始 LLaMA 输入：

```
请帮我写一首关于春天的诗。
```

预训练模型看到这段文字后，它可能会这样"补全"：

```
请帮我写一首关于春天的诗。请帮我写一首关于夏天的诗。请帮我写一首关于秋天的诗。请帮我写一首关于冬天的诗。
```

或者：

```
请帮我写一首关于春天的诗。要求：1. 不少于八句 2. 押韵 3. 包含意象描写
```

预训练模型的补全结果看起来像是在"模仿"它见过的类似网页文本，而不是"理解并完成这个指令"。它不知道自己的角色是"AI 助手"，不知道应该直接输出诗歌，不知道"请帮我写"意味着用户想要模型产出而不是继续列举更多指令。

指令微调（Supervised Fine-Tuning, SFT）的目标就是：**用"指令-回复对"的形式训练模型，让模型学会"看到指令后直接输出回复"的交互模式**。

### 1.2 SFT 教会模型的三个能力

经过 SFT 后，模型学会了三种关键能力：

**能力一：角色识别**

模型理解了对话结构中不同角色的含义——system 是给模型的背景指令，user 是需要回复的问题，assistant 是模型自己应该产出的回答。它学会了在看到 `<|im_start|>assistant` 后开始生成回复，看到 `<|im_end|>` 后停止生成。

**能力二：指令遵循**

模型学会了按照指令的要求来组织回复——如果用户说"用 JSON 格式输出"，它会输出 JSON；如果用户说"用小朋友能听懂的语言解释"，它会简化语言。这不是"记忆"了所有可能的指令格式，而是学会了"理解指令意图并调整输出行为"的元能力。

**能力三：格式化输出**

模型学会了在回复中保持一致的格式——稳定的段落结构、正确的 Markdown 格式、一致的语气和风格。这是通过大量格式一致的训练数据"训练"出来的习惯，而不是每次靠 Prompt 约束的结果。

### 1.3 SFT 不能做什么

理解 SFT 的边界同样重要。SFT 不是万能的：

- SFT 主要是教会模型"格式"和"风格"，而不是注入大量新的知识。SFT 的数据量通常只有几万到几十万条，远不足以改变模型的知识储备。
- SFT 不能纠正模型的基础能力缺陷。如果基座模型数学就不好，SFT 不会让它突然变成数学天才。
- SFT 不能保证安全性。如果训练数据中有不安全的内容，SFT 后的模型反而可能更不安全。
- SFT 后的模型可能会过拟合到训练数据的分布上，对分布外的指令泛化能力下降。

---

## 2. 训练数据格式：Chat Template 的完整规范

### 2.1 标准化格式

SFT 训练数据必须严格遵循统一的 Chat Template 格式。大多数现代的对话模型使用以下标准格式：

```
<|im_start|>system
你是一个有帮助的AI助手。
<|im_end|>
<|im_start|>user
请帮我翻译下面的句子：Hello World
<|im_end|>
<|im_start|>assistant
请把下面的句子翻译成中文：你好世界
<|im_end|>
```

每个样本在 tokenize 之后会变成一个长的 token 序列。关键的设计决策是：**对整个序列计算语言模型的 loss（预测下一个 token），但只对 assistant 部分计算 loss 的反向传播梯度**。system 和 user 部分的 token 虽然参与前向传播（提供上下文），但不参与 loss 计算。

### 2.2 多轮对话的模板

对于多轮对话，格式也很直观——就是多次重复 user/assistant 轮次：

```
<|im_start|>system
你是一个编程助手。
<|im_end|>
<|im_start|>user
Python 中如何读取文件？
<|im_end|>
<|im_start|>assistant
使用 open() 函数配合 with 语句可以安全地读取文件：
```python
with open("file.txt", "r") as f:
    content = f.read()
```
<|im_end|>
<|im_start|>user
那二进制文件呢？
<|im_end|>
<|im_start|>assistant
读取二进制文件只需要将模式改为 "rb"：
```python
with open("image.png", "rb") as f:
    data = f.read()
```
<|im_end|>
```

多轮对话数据非常有价值，因为它教会模型：
- 理解对话历史中的指代（"那二进制文件"指的是什么？）
- 在之前回答的基础上补充细节
- 纠正之前不完整或不准确的回答
- 保持对话风格和深度的一致性

### 2.3 特殊 token 的重要性

Chat Template 中的特殊 token（如 `<|im_start|>`、`<|im_end|>`）不是装饰品，它们是模型理解对话结构的关键信号。

这些特殊 token 在预训练或指令微调阶段被用作"分隔符"，模型学会了：
- `<|im_start|>` 表示一个新角色的开始
- `<|im_end|>` 表示当前角色发言的结束
- `<|im_start|>assistant` 是模型"该我说话了"的信号

如果在 SFT 中使用了错误的 Chat Template（比如把 `<|im_start|>` 写成了 `[INST]`），模型虽然也能训练（loss 会下降），但这些特殊 token 对模型来说失去了"分隔符"的语义，推理时模型可能不知道何时该停止、何时该切换角色。

---

## 3. 训练流程详解

### 3.1 数据加载：HuggingFace Datasets

SFT 训练的第一步是将准备好的 JSONL 数据加载为 HuggingFace Dataset 格式。为什么要用 Dataset 而不用 Python 原生列表？

- **内存效率**：Dataset 使用 Apache Arrow 格式存储，支持内存映射（memory-mapping），不需要将全部数据加载到 RAM 中。对于百万级别的数据集，这至关重要。
- **流式处理**：支持 streaming 模式，数据不需要全部下载就可以开始训练。
- **高效映射**：`dataset.map()` 方法支持多进程并行处理，显著加速 tokenization。

### 3.2 Tokenization + Chat Template 应用

Tokenization 是将文本转换为模型能理解的 token ID 序列的过程。在 SFT 中，这一步特别需要注意 Chat Template 的正确应用。

**为什么必须用 `apply_chat_template`？**

不同模型的 Chat Template 不仅仅是格式不同，还涉及：
- 特殊 token 的 token ID 是否正确
- 是否需要在 assistant 回复前添加生成提示（generation prompt）
- system prompt 的位置和格式
- 是否支持多轮对话

手动拼接字符串很容易出错。`apply_chat_template` 从 tokenizer 的配置文件中读取正确的模板，保证格式的一致性。

**关于 `add_generation_prompt` 参数**：
- 训练时设为 `False`：因为训练数据中已经包含了完整的 assistant 回复
- 推理时设为 `True`：在 assistant 的位置添加生成提示信号，告诉模型"现在开始生成回复"

### 3.3 Loss Masking：只对 assistant 回复计算 loss

Loss Masking 是 SFT 中最重要的技术细节，也是最容易被忽视的。让我们深入理解它的原理和必要性。

#### 3.3.1 为什么需要 Loss Masking

在因果语言模型中，训练目标是对序列中的每个位置 i，用前 i 个 token 预测第 i+1 个 token。loss 通常是对所有位置取平均。

但 SFT 的训练样本包含了 system、user 和 assistant 三部分的 token。如果我们对所有位置都计算 loss，模型会学习：
- 给定 system prompt，预测 user 的输入（不合理！）
- 给定 user 输入，预测 assistant 的回复（合理，这是我们想要的）
- 在 user 输入内部预测 user 的下一个 token（会浪费训练算力在无关的目标上）

更重要的是，如果模型学会了"预测 user 输入"，它在推理时可能会"替用户说话"——在某些情况下，模型输出完 assistant 回复后不停止，继续输出下一个 user 输入，形成一种奇怪的"自问自答"模式。

#### 3.3.2 Loss Masking 的实现原理

Loss Masking 的核心操作是：将 system 和 user 部分的 labels 设为 -100（PyTorch 的 `CrossEntropyLoss` 默认忽略 index=-100 的位置）。

实现步骤：
1. 先对整个序列创建 labels（labels = input_ids 的拷贝）
2. 然后找到 assistant 部分在序列中的起止位置
3. 将非 assistant 部分的 labels 替换为 -100
4. 只有 assistant 部分的 labels 保持原值，参与 loss 计算

```python
"""
Loss Masking 的核心实现。

对于 Chat Template 格式的序列:
<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>

我们只想对 assistant 部分计算 loss。
"""
import torch


def create_masked_labels(
    input_ids: torch.Tensor,
    assistant_start_positions: list[int],
    assistant_end_positions: list[int],
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    创建带 masking 的 labels。

    参数:
        input_ids: (seq_len,) 完整的 token ID 序列
        assistant_start_positions: 每个 assistant 回复段的起始位置列表
        assistant_end_positions: 每个 assistant 回复段的结束位置列表
        ignore_index: 被 mask 掉的位置的填充值（PyTorch 默认 -100）

    返回:
        labels: (seq_len,) 只在 assistant 位置保留 token ID，其余为 -100
    """
    # 初始化为全部忽略
    labels = torch.full_like(input_ids, ignore_index)

    # 只保留 assistant 部分的 token ID
    for start, end in zip(assistant_start_positions, assistant_end_positions):
        labels[start:end] = input_ids[start:end]

    return labels
```

**如何找到 assistant 部分的起止位置？**

这需要在 tokenization 的同步完成。一种方法是：在 tokenize 之前，分别 tokenize 非 assistant 部分和 assistant 部分，记录它们的长度，从而推断位置。

更精确的方法是使用 tokenizer 返回的 `offset_mapping` 或其他 attention 信息。但实践中，最常用的方法是：

```python
def tokenize_with_mask(messages: list[dict], tokenizer) -> dict:
    """
    将 messages 列表 tokenize 并同时生成 loss mask。

    策略: 分别构建 "prompt"（system + user 部分）和 "completion"（assistant 部分），
    配合 tokenizer 的 add_generation_prompt 来精确控制。

    注意: 这个方法利用了一个关键事实——
    在 Chat Template 中，assistant 部分总是排在 prompt 的后面。
    """
    # Step 1: 构建 prompt（system + user，加上 generation prompt 暗示接下来是 assistant）
    prompt_messages = [
        msg for msg in messages if msg["role"] in ("system", "user")
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,  # 添加引导 assistant 回复的 token
    )

    # Step 2: 构建完整文本（包含 assistant 回复）
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # 不需要，因为已经包含了 assistant 回复
    )

    # Step 3: Tokenize prompt 部分（不包含 assistant 回复）
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    # Step 4: Tokenize 全部
    full_tokens = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    # Step 5: prompt 部分的长度就是 assistant 部分的起始位置
    prompt_len = len(prompt_tokens)
    full_len = len(full_tokens)

    # Step 6: 创建 labels（prompt 部分为 -100，assistant 部分保留）
    labels = [-100] * full_len
    labels[prompt_len:] = full_tokens[prompt_len:]

    # 确保长度一致
    if len(labels) > full_len:
        labels = labels[:full_len]
    elif len(labels) < full_len:
        labels += [-100] * (full_len - len(labels))

    return {
        "input_ids": full_tokens,
        "labels": labels,
    }
```

### 3.4 训练超参设置

SFT 的超参数设置与预训练有显著不同。理解这些差异背后的原因，比记住具体数值更重要。

#### 3.4.1 学习率（Learning Rate）

SFT 的学习率通常比预训练小 10-100 倍。预训练的学习率通常是 1e-4 到 3e-3，而 SFT 的学习率通常是 1e-5 到 5e-5（全量微调）或 1e-4 到 5e-4（LoRA）。

**为什么更小？**
- 预训练时，模型参数是随机初始化的，需要较大的步长来快速找到好的解空间
- SFT 从一个已经训练好的模型出发，模型已经在很好的位置，大学习率会把它"踢出"这个好位置
- 过大的学习率会导致灾难性遗忘——模型迅速忘记预训练学到的通用知识

**为什么 LoRA 可以用稍大的学习率？**
- LoRA 只更新少量参数（<1%），而且 LoRA adapter 是随机初始化的
- 这些新参数需要稍大的学习率来快速适应
- 原始参数保持冻结，不会因为大学习率而偏离

#### 3.4.2 训练轮数（Epochs）

SFT 通常只需要 3-5 个 epoch。太少学不到指令模式，太多会过拟合。

**为什么不是越多越好？**

SFT 的数据量（通常几千到几万条）远小于预训练（几万亿 token）。这意味着每个样本在训练过程中被"看"的次数更多。在预训练中，一个网页通常只被模型看过 1-2 次。而在 SFT 中，一条指令可能在 3 个 epoch 中被看了 3 次。

如果训练太多 epoch（比如 10 个以上），模型会开始"背诵"训练数据——它对训练集中的指令回复得完美，但对未见过的指令泛化能力下降。这就是过拟合。

#### 3.4.3 Batch Size 和梯度累积

SFT 的有效 batch size 通常推荐在 64-128 之间。有效 batch size = per_device_batch_size * gradient_accumulation_steps * num_gpus。

如果单卡显存只够 batch_size=2，可以通过设置 gradient_accumulation_steps=32 来达到有效 batch_size=64。代价是训练变慢（每个优化步骤需要 32 次前向+反向传播）。

**为什么需要较大的 batch size？**
- 小 batch 的梯度噪声大，训练不稳定
- 较大的 batch 能提供更稳定的梯度估计，帮助模型更好地学习指令的共性模式

#### 3.4.4 学习率调度器

推荐使用余弦退火（cosine annealing）调度：
- 训练开始时线性 warmup，从 0 增长到目标学习率
- 然后按余弦曲线从目标学习率逐渐减小
- 训练结束时学习率接近 0

这种策略的好处：
- Warmup 阶段避免初始训练的不稳定
- 余弦退火在训练后期让学习率降到很低，帮助模型收敛到更好的局部最优

---

## 4. 过拟合检测

### 4.1 过拟合的本质

SFT 中的过拟合表现为：训练 loss 持续下降，但验证 loss 开始上升，或者生成质量开始下降。

这是一种典型的"memorization vs generalization"的权衡。模型在记忆训练数据中的特定模式（甚至是噪声），而这些模式在验证集中不存在。

### 4.2 检测方法

**Loss 曲线监控**：

最基本的检测手段是同时绘制训练 loss 和验证 loss 曲线。
- 两条曲线都在下降：正常训练，模型在进步
- 训练 loss 下降但验证 loss 上升：过拟合开始
- 训练 loss 下降但验证 loss 不变化：模型已经饱和，继续训练意义不大

注意：由于训练数据通常不完全 represent 验证数据的分布，训练 loss 低于验证 loss 是正常的。关键是看趋势——验证 loss 是否开始稳定上升，而不是绝对值。

**生成质量抽查**：

loss 只是代理指标。真正的评判标准是模型的实际生成质量。在每个 epoch 结束时，用当前 checkpoint 对一组固定的测试问题生成回复，人工检查：
- 回复是否准确回答了问题？
- 回复格式是否一致？
- 有没有出现重复、不连贯或奇怪的输出？（过拟合后的常见症状）
- 对未见过的指令类型，泛化能力如何？

**验证集的自动评估**：

对于有标准答案的任务（如翻译、摘要），可以用 BLEU、ROUGE 等指标自动评估。对于开放式对话，可以使用 LLM-as-Judge 评分（详见第05章）。

### 4.3 防止过拟合的方法

1. **增加数据多样性**：确保训练数据覆盖足够多样化的指令类型、主题和风格
2. **减少训练轮数**：如果验证 loss 在第 2 个 epoch 后就开始上升，那就只训练 2 个 epoch
3. **增加正则化**：增大 weight_decay（L2 正则化）、增大 dropout
4. **数据增强**：对训练数据进行同义改写、添加噪声、改变表达方式，增加数据多样性
5. **早停（Early Stopping）**：在验证 loss 不再下降时提前终止训练

---

## 5. Checkpoint 择优

### 5.1 不是最后一个 checkpoint 最好

初学者常犯的错误是自动使用训练结束时的最后一个 checkpoint。事实上，对於 SFT，"中间"的 checkpoint 往往更好。

原因在于：SFT 的训练过程通常是：
- 第 1 个 epoch：模型快速学习指令格式（loss 大幅下降）
- 第 2-3 个 epoch：模型优化回复质量（loss 缓慢下降）
- 第 4-5 个 epoch：loss 可能还在微降，但实际上模型开始过拟合（验证 loss 上升或生成质量下降）

如果在每个 epoch 结束时保存了 checkpoint，最佳的那个通常在第 2-3 个 epoch 之间。

### 5.2 Checkpoint 择优的方法

**基于验证 loss**：
选择验证 loss 最低的 checkpoint。这是最基本的方法，但不够精确——验证 loss 和实际生成质量的相关性不是完美的。

**基于生成质量**：
在一组固定的评估问题上，用每个 checkpoint 生成回复，用 LLM-as-Judge 评分，选择总分最高的 checkpoint。这个方法更可靠但更耗时。

**基于多指标综合**：
同时考虑验证 loss、BLEU/ROUGE 分数（如果适用）、LLM-as-Judge 评分、回复多样性等指标，综合判断。可以给每个指标分配权重，计算综合得分。

---

## 6. 完整的 SFT 训练代码

下面是一个完整的 SFT 训练脚本，涵盖了数据加载、tokenization、loss masking、LoRA 配置、训练和 checkpoint 评估的全流程。

```python
"""
完整的 SFT 训练脚本。

流程:
1. 加载基座模型（支持 LoRA/QLoRA）
2. 加载和预处理训练数据
3. 实现 Loss Masking（只对 assistant 回复计算 loss）
4. 配置 Trainer 并训练
5. 评估每个 checkpoint 的生成质量
6. 选择最佳 checkpoint

依赖:
    pip install transformers accelerate peft bitsandbytes datasets torch
"""
import os
import json
import torch
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from datasets import Dataset, load_dataset


# ============================================================
# 1. 模型和 Tokenizer 加载
# ============================================================

def load_model_and_tokenizer(
    model_name: str,
    use_qlora: bool = False,
) -> tuple:
    """
    加载基座模型和 tokenizer。

    参数:
        model_name: HuggingFace 模型名或本地路径
        use_qlora: 是否使用 4-bit QLoRA

    返回:
        (model, tokenizer)
    """
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # 某些模型还需要设置 pad_token_id
        # tokenizer.pad_token_id = tokenizer.eos_token_id

    # 加载模型
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if use_qlora:
        # QLoRA: 使用 NF4 量化加载
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        # LoRA: 直接以 BF16/FP16 加载
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # QLoRA 需要的额外步骤
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    # 启用梯度检查点（节省显存，用计算换内存）
    model.gradient_checkpointing_enable()

    return model, tokenizer


# ============================================================
# 2. 数据加载和预处理（含 Loss Masking）
# ============================================================

def load_sft_data(data_path: str) -> Dataset:
    """从 JSONL 文件加载 SFT 训练数据。"""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return Dataset.from_list(data)


def tokenize_with_loss_mask(
    example: dict,
    tokenizer,
    max_length: int = 2048,
) -> dict:
    """
    将 messages 格式的数据 tokenize 并创建 loss mask。

    核心思路:
    - 将 prompt (system + user) 和完整文本分别 tokenize
    - prompt 的长度就是 assistant 回复的起始位置
    - 将 prompt 部分的 labels 设为 -100（忽略 loss）
    - 只保留 assistant 部分的 labels（计算 loss）

    参数:
        example: 包含 "messages" 字段的字典
            messages 格式: [{"role": "system", "content": "..."},
                            {"role": "user", "content": "..."},
                            {"role": "assistant", "content": "..."}]
        tokenizer: HuggingFace tokenizer
        max_length: 最大序列长度

    返回:
        {"input_ids": [...], "attention_mask": [...], "labels": [...]}
    """
    messages = example["messages"]

    # 分离 prompt 和 assistant 部分
    # prompt 包含: system + user（可能有多个轮次）
    prompt_messages = []
    assistant_contents = []

    for msg in messages:
        if msg["role"] in ("system", "user"):
            prompt_messages.append(msg)
        elif msg["role"] == "assistant":
            assistant_contents.append(msg["content"])
            prompt_messages.append(msg)  # assistant 也要加入 prompt（作为多轮历史）

    # 构建完整的 prompt（包含所有历史消息 + 最后的 generation prompt）
    # add_generation_prompt=True 会在最后添加引导 assistant 开始回复的 token
    full_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 构建完整的训练文本（不添加 generation prompt，因为已经包含了 assistant 回复）
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize prompt 部分
    prompt_enc = tokenizer(full_prompt, add_special_tokens=False)
    prompt_len = len(prompt_enc["input_ids"])

    # Tokenize 完整文本
    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    # 创建 labels
    full_len = len(full_enc["input_ids"])
    labels = [-100] * full_len  # 初始化为全部忽略

    # 只对 assistant 部分保留真实的 token ID
    # assistant 部分从 prompt_len 开始到 full_len 结束
    # 注意：需要截断到 max_length 范围内
    assist_start = min(prompt_len, full_len)
    labels[assist_start:] = full_enc["input_ids"][assist_start:]

    # 确保 labels 长度与 input_ids 一致
    labels = labels[:full_len]

    return {
        "input_ids": full_enc["input_ids"],
        "attention_mask": full_enc["attention_mask"],
        "labels": labels,
    }


def preprocess_dataset(
    dataset: Dataset,
    tokenizer,
    max_length: int = 2048,
    num_proc: int = 4,
) -> Dataset:
    """
    预处理整个数据集：逐条应用 tokenize_with_loss_mask。

    参数:
        dataset: 原始数据集（每条包含 "messages" 字段）
        tokenizer: tokenizer 对象
        max_length: 最大序列长度
        num_proc: 并行处理的进程数

    返回:
        预处理后的数据集（包含 input_ids, attention_mask, labels）
    """
    def _tokenize(example):
        return tokenize_with_loss_mask(example, tokenizer, max_length)

    # 使用 map 进行并行处理
    # remove_columns 移除原始文本字段，只保留 token 化后的数据
    processed = dataset.map(
        _tokenize,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc="Tokenizing and masking",
    )

    return processed


# ============================================================
# 3. LoRA 配置
# ============================================================

def configure_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> PeftModel:
    """
    为模型配置 LoRA adapter。

    参数:
        model: 基座模型
        r: LoRA rank
        lora_alpha: LoRA alpha 缩放系数
        lora_dropout: LoRA dropout 概率

    返回:
        挂载了 LoRA adapter 的模型
    """
    # 自动检测模型的线性层名称
    # 对于大多数 LLaMA/Qwen 架构的模型，以下是常见的命名
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # 验证 target_modules 是否存在
    # 不同模型的命名可能不同（如 ChatGLM 使用 query_key_value）
    available_modules = set()
    for name, _ in model.named_modules():
        available_modules.add(name.split(".")[-1])

    valid_targets = [m for m in target_modules if m in available_modules]

    if not valid_targets:
        # 如果预设的都找不到，尝试自动检测所有线性层
        print("警告: 未找到预设的 target_modules，尝试自动检测...")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                valid_targets.append(name.split(".")[-1])
        valid_targets = list(set(valid_targets))  # 去重
        print(f"自动检测到的线性层: {valid_targets}")

    print(f"LoRA target_modules: {valid_targets}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=valid_targets,
        lora_dropout=lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# ============================================================
# 4. 自定义 Data Collator（支持 labels）
# ============================================================

class SFTDataCollator:
    """
    自定义 Data Collator，用于 SFT 训练。

    DataCollatorForLanguageModeling 会自动创建 labels（右移 input_ids），
    但我们已经在数据预处理中手动创建了带有 loss masking 的 labels。
    因此需要一个简单的 collator，只做 padding 而不覆盖 labels。
    """

    def __init__(self, tokenizer, padding: bool = True):
        self.tokenizer = tokenizer
        self.padding = padding

    def __call__(self, features: list[dict]) -> dict:
        """将不等长的样本 padding 到相同长度。"""
        # 找到当前 batch 中最长的序列
        max_len = max(len(f["input_ids"]) for f in features)

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len

            # Padding input_ids（用 pad_token_id 填充）
            batch["input_ids"].append(
                f["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )
            # Padding attention_mask（padding 部分为 0）
            batch["attention_mask"].append(
                f["attention_mask"] + [0] * pad_len
            )
            # Padding labels（用 -100 填充，PyTorch loss 忽略）
            batch["labels"].append(
                f["labels"] + [-100] * pad_len
            )

        # 转换为 tensor
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
        }


# ============================================================
# 5. 训练
# ============================================================

def train_sft(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    output_dir: str = "./sft_output",
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_length: int = 2048,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    save_total_limit: int = 3,
    logging_steps: int = 10,
):
    """
    执行 SFT 训练。

    参数:
        model: 配置了 LoRA 的模型
        tokenizer: tokenizer
        train_dataset: 训练数据集（已 tokenize）
        val_dataset: 验证数据集（已 tokenize），可选
        output_dir: 输出目录
        num_epochs: 训练轮数
        learning_rate: 学习率
        per_device_batch_size: 每设备 batch size
        gradient_accumulation_steps: 梯度累积步数
        max_length: 最大序列长度
        warmup_ratio: warmup 比例
        weight_decay: 权重衰减
        save_total_limit: 最多保存 checkpoint 数量
        logging_steps: 日志记录步数
    """
    effective_batch_size = per_device_batch_size * gradient_accumulation_steps
    print(f"有效 batch size: {effective_batch_size}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # 学习率配置
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",

        # 优化器配置
        optim="adamw_torch",
        weight_decay=weight_decay,

        # 精度
        bf16=True,
        # fp16=False,  # 如果 GPU 不支持 BF16，改用 FP16

        # 保存策略
        save_strategy="epoch",
        save_total_limit=save_total_limit,

        # 日志
        logging_steps=logging_steps,
        report_to="none",  # 如需 tensorboard，改为 "tensorboard"

        # 评估策略（如果有验证集）
        eval_strategy="epoch" if val_dataset else "no",
        eval_steps=None,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,

        # 数据加载
        dataloader_num_workers=2,
        remove_unused_columns=False,

        # 梯度检查点（节省显存）
        gradient_checkpointing=True,
    )

    # Data Collator
    data_collator = SFTDataCollator(tokenizer)

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 开始训练
    print(f"\n{'='*50}")
    print("  开始 SFT 训练")
    print(f"  训练样本: {len(train_dataset)}")
    if val_dataset:
        print(f"  验证样本: {len(val_dataset)}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  学习率: {learning_rate}")
    print(f"{'='*50}\n")

    trainer.train()

    # 保存最终模型
    final_output = f"{output_dir}/final"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"\n训练完成！模型已保存至 {final_output}")

    return trainer


# ============================================================
# 6. Checkpoint 评估
# ============================================================

def evaluate_checkpoints(
    checkpoints_dir: str,
    base_model_name: str,
    test_questions: list[str],
    tokenizer,
    system_prompt: str = "你是一个有帮助的AI助手。",
) -> dict:
    """
    评估所有 checkpoint 的生成质量。

    对每个 checkpoint，用测试问题生成回复并人工/自动评估。

    参数:
        checkpoints_dir: 包含多个 checkpoint 的目录
        base_model_name: 基座模型名（用于加载 checkpoint）
        test_questions: 测试问题列表
        tokenizer: tokenizer
        system_prompt: 系统提示

    返回:
        {"checkpoint_name": {"responses": [...], "avg_length": ...}, ...}
    """
    results = {}

    # 遍历所有 checkpoint 子目录
    import glob
    checkpoint_dirs = sorted(glob.glob(f"{checkpoints_dir}/checkpoint-*"))

    for ckpt_dir in checkpoint_dirs:
        ckpt_name = os.path.basename(ckpt_dir)
        print(f"\n评估 {ckpt_name}...")

        try:
            # 加载基座模型并挂载 checkpoint 的 LoRA adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base_model, ckpt_dir)
            model.eval()

            responses = []
            for question in test_questions:
                # 构建 prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(input_text, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                # 只取生成的部分（不包括输入）
                gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
                response = tokenizer.decode(gen_ids, skip_special_tokens=True)
                responses.append(response)

            results[ckpt_name] = {
                "responses": responses,
                "avg_length": sum(len(r) for r in responses) / len(responses),
            }

            # 清理显存
            del base_model
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  评估 {ckpt_name} 失败: {e}")
            results[ckpt_name] = {"error": str(e)}

    # 打印比较
    print(f"\n{'='*50}")
    print("  Checkpoint 对比")
    print(f"{'='*50}")
    for ckpt_name, info in results.items():
        if "error" not in info:
            print(f"\n--- {ckpt_name} (平均回复长度: {info['avg_length']:.0f} 字符) ---")
            for i, (q, r) in enumerate(zip(test_questions, info["responses"])):
                print(f"Q{i+1}: {q}")
                print(f"A{i+1}: {r[:200]}...")  # 只打印前 200 字符

    return results


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    # 配置参数
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"     # 基座模型（小模型便于测试）
    DATA_PATH = "sft_train_data.jsonl"     # SFT 训练数据
    VAL_DATA_PATH = "sft_val_data.jsonl"   # SFT 验证数据
    OUTPUT_DIR = "./sft_output"            # 输出目录
    USE_QLORA = False                      # 是否使用 QLoRA
    LORA_R = 16                            # LoRA rank
    LORA_ALPHA = 32                        # LoRA alpha
    MAX_LENGTH = 1024                      # 最大序列长度（小模型可设短一些）
    NUM_EPOCHS = 3                         # 训练轮数

    # 1. 加载模型
    print("加载模型和 tokenizer...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, use_qlora=USE_QLORA)

    # 2. 配置 LoRA
    print("配置 LoRA...")
    model = configure_lora(model, r=LORA_R, lora_alpha=LORA_ALPHA)

    # 3. 加载和预处理数据
    print("加载训练数据...")
    train_dataset = load_sft_data(DATA_PATH)
    print(f"训练数据: {len(train_dataset)} 条")

    print("预处理训练数据（tokenize + loss masking）...")
    train_dataset = preprocess_dataset(train_dataset, tokenizer, max_length=MAX_LENGTH)

    val_dataset = None
    if os.path.exists(VAL_DATA_PATH):
        print("加载验证数据...")
        val_dataset = load_sft_data(VAL_DATA_PATH)
        val_dataset = preprocess_dataset(val_dataset, tokenizer, max_length=MAX_LENGTH)
        print(f"验证数据: {len(val_dataset)} 条")

    # 4. 训练
    trainer = train_sft(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=OUTPUT_DIR,
        num_epochs=NUM_EPOCHS,
        max_length=MAX_LENGTH,
    )

    print("\nSFT 训练完成！")
    print(f"模型保存位置: {OUTPUT_DIR}")
```

---

## 基础练习

1. **Chat Template 应用**：编写一个脚本，对比同一组 messages 在三种不同模型（LLaMA-3、Qwen2.5、ChatGLM-4）的 Chat Template 下的实际输出格式，理解不同模板的差异。

2. **Loss Masking 实现**：基于 tokenize_with_loss_mask 函数，手动实现 loss masking 逻辑。验证你的实现：确保 system/user 部分的 labels 全为 -100，assistant 部分不为 -100。

3. **过拟合检测**：使用训练过程中的 loss 日志数据，编写可视化脚本，绘制训练 loss 和验证 loss 曲线，自动检测过拟合的起始点（训练 loss 下降但验证 loss 上升的那个 epoch）。

## 进阶练习

1. **完整 SFT Pipeline**：将本章的代码整合为一个可配置的命令行工具，支持自定义模型、数据路径、超参数。实现断点恢复训练功能。

2. **Checkpoint 自动择优**：编写一个自动化脚本，在训练完成后自动评估所有 checkpoint，综合考虑验证 loss 和生成质量（可以用更高级的模型作为 Judge），选出最佳 checkpoint。

---

## 常见错误

1. **忘记做 Loss Masking**：对所有 token 都计算 loss，包括 system prompt 和 user 输入。这会导致模型学习"预测用户输入"而非"回复用户"。推理时模型可能"自问自答"。

2. **训练和推理时 Chat Template 不一致**：训练时用 `add_generation_prompt=False`（因为包含了 assistant 回复），推理时用 `add_generation_prompt=True`（因为需要模型生成 assistant 回复）。如果搞反了会导致生成质量极差。

3. **学习率太大导致灾难性遗忘**：模型迅速丧失了预训练学到的通用能力，只会机械地模仿训练数据。在验证集上可能表现为：回复格式正确但内容空洞、缺乏常识。

4. **Overfitting 到小数据集上**：只在几百条数据上训练了 10 个 epoch，模型完美背诵训练集但对新问题回答得很差。解决方法：增加数据、减少 epoch、增加 dropout。

5. **忽略验证集的代表性**：验证集和训练集分布差异太大，导致验证 loss 无法真实反映泛化能力。验证集应该和实际使用场景的分布一致。

6. **不对 labels 做 padding**：labels 的 padding 必须用 -100 而不是 0。如果用 0（可能对应某个真实 token），模型会在 padding 位置也计算 loss，混淆训练信号。

7. **序列截断时截断了 assistant 回复**：truncation 可能把重要的回复部分截掉。应该确保截断策略不会把 assistant 回复的后半部分截掉（只能截前面的部分）。

8. **多轮对话的 Loss Masking 不正确**：在多轮对话中，所有 assistant 轮次都应该计算 loss，所有 system/user 轮次都应该 mask。初学者可能只 mask 了第一个轮次。

9. **未区分 eos_token 和 pad_token**：将 eos_token 用作 pad_token 时，attention mask 可能在应该忽略的位置出现非零值。建议在 tokenizer 中显式设置 pad_token 和 pad_token_id。

10. **在 QLoRA 训练中使用了不兼容的 DeepSpeed 配置**：某些 DeepSpeed ZeRO 阶段与 bitsandbytes 量化不兼容。在使用 QLoRA 时，建议使用简单的 DDP 或单卡训练，而不是复杂的分布式策略。

---

## 本章小结

本章我们深入探讨了指令微调（SFT）的完整训练流程。我们从 SFT 的目标出发——让模型学会角色识别、指令遵循和格式化输出——理解了 SFT 能做什么和不能做什么。

在训练数据格式方面，我们详细拆解了 Chat Template 的结构和特殊 token 的作用，理解了为什么统一格式和正确使用 `apply_chat_template` 至关重要。

在训练流程中，我们逐步讲解了数据加载、tokenization、Loss Masking、超参设置等关键环节。Loss Masking 是本章的技术重点——我们深入解释了为什么只对 assistant 回复计算 loss，以及如何正确实现。

过拟合检测和 Checkpoint 择优是确保最终模型质量的实践技能。我们介绍了多种检测方法，并强调了"最优 checkpoint 不一定在最后一个 epoch"这个重要认知。

最后，我们提供了一套完整的 SFT 训练代码，覆盖了从模型加载到 checkpoint 评估的全流程。

下一章，我们将学习偏好对齐（RLHF/DPO），让模型不仅会遵循指令，还能产生更符合人类偏好的回复。
