# 第04章 偏好对齐 (RLHF/DPO)

## 学习目标

完成本章后，你将能够：

1. 理解为什么 SFT 之后还需要偏好对齐——SFT 让模型"会做"，偏好对齐让模型"做得好"
2. 完整描述 RLHF 的三个阶段：SFT、奖励模型训练、PPO 强化学习
3. 理解奖励模型的数据结构（chosen vs rejected）和训练原理
4. 用直觉掌握 PPO 的核心机制——KL 散度约束和策略更新
5. 理解 DPO 的简化哲学——直接从偏好对中学习，无需单独的奖励模型
6. 使用 TRL 库实现 DPO 训练，并对齐自己的微调模型

## 前置知识

- 指令微调（SFT）的完整流程（第03章水平）
- LoRA 和 PEFT 库的使用（第02章水平）
- 基础概率论：条件概率、交叉熵、KL 散度（不需要精通，理解概念即可）
- 强化学习的基本概念（智能体、环境、奖励、策略，了解即可）

---

## 1. 为什么需要偏好对齐

### 1.1 SFT 的边界

在第03章中，我们通过 SFT 让模型学会了"理解指令并给出回复"。但如果你实际使用过 SFT 后的模型，你可能会发现一些令人困扰的问题：

- **敷衍了事**：模型知道要回复，但回复可能很短、没有深度。比如问"量子计算是什么"，SFT 模型可能只给一两句话，而用户期望的是一个有结构的长篇解释。
- **过于"讨好人"**：模型有时会给不出错但毫无帮助的回答——"这个问题很复杂，取决于具体情况"——这样的回答技术上没错但对用户毫无价值。
- **安全性问题**：如果训练数据中包含了不安全的对话，SFT 模型可能在应该拒绝回答的时候仍然给出有害的回复。
- **风格不统一**：有些回复过于啰嗦，有些过于简略。模型似乎不理解"什么时候该详细，什么时候该简洁"。

这些问题的根源在于：SFT 的训练目标是最小化 token 级别的预测误差（cross-entropy loss）。这个目标隐含地假设所有训练数据中的回复都是"同样好"的——模型不知道哪个回复更好，只知道"这些回复都是正确的"。

但在真实世界中，对于同一个问题，显然存在"更好的回复"和"更差的回复"。我们需要一种训练方法，让模型学会区分好坏——这就是偏好对齐。

### 1.2 偏好对齐的核心思想

偏好对齐的核心思想很直观：**不是告诉模型"这个回复是正确的"，而是告诉模型"回复 A 比回复 B 更好"**。

为什么这种"相对比较"比"绝对正确"更有效？

第一，相对判断比绝对判断更容易标注。让人打一个绝对分数（"这个回复 7 分"）非常主观且不稳定，但让人从两个回复中选择更好的一个（"A 更好"）就容易得多，标注员之间的一致性也更高。

第二，偏好数据包含更丰富的信息。一个训练样本不再是"问题 X 的正确回答是 Y"，而是"问题 X 有回答 Y_chosen（被选中的好回答）和 Y_rejected（被拒绝的差回答），Y_chosen 比 Y_rejected 更好"。模型不仅学到了"什么是对的"，还学到了"什么是更好的"。

第三，偏好对齐可以注入多元化的价值观。不同场景下"好"的定义不同——客服场景偏好礼貌委婉，教育场景偏好详细解释，创意写作场景偏好有趣和新颖。通过对不同维度的偏好数据加权，可以定制模型的"价值观"。

---

## 2. RLHF 的完整流程

RLHF（Reinforcement Learning from Human Feedback）是 OpenAI 在 InstructGPT 论文中提出的一套完整流程。它分为三个阶段，每个阶段都有独特的目标和方法。

### 2.1 阶段一：监督微调（SFT）

这是 RLHF 的起点，我们在第03章已经详细讲过。RLHF 框架中的 SFT 与独立的 SFT 没有本质区别，都是用高质量的"指令-回复对"训练模型，让它学会基本的指令遵循能力。

在这个阶段，收集的数据通常由人类标注员编写——直接根据指令写出高质量的回复。这些数据量通常不大（1-5 万条），但质量需要很高。

完成 SFT 后，模型已经有了基本的对话和指令遵循能力。但它还不知道"什么样的回复更好"。

### 2.2 阶段二：训练奖励模型（Reward Model）

这是 RLHF 最独特的环节。我们需要训练一个"评分模型"——给它一个输入（指令）和一个回复，它能输出一个分数，表示这个回复有多好。

**奖励模型的数据结构**：

奖励模型的训练数据是"偏好对"（preference pairs）：
- 同一个指令，有两个（或多个）回复
- 人类标注员选择哪个更好（chosen）和哪个更差（rejected）
- 数据集格式：`(prompt, chosen_response, rejected_response)`

**为什么需要单独的奖励模型？**

你可能会问：既然我们已经有 SFT 模型了，为什么不直接在 SFT 数据上继续训练，而是要去训练一个单独的奖励模型？

答案是：在 PPO 强化学习阶段，我们需要在每一个 token 生成后都获得一个"即时奖励"信号。但人类只能对完整的回复打分，无法对生成过程中的每个中间步骤打分。奖励模型的作用就是**将人类对完整回复的偏好信号"分解"为生成过程中的 token 级别的奖励**。

**奖励模型的训练**：

奖励模型的训练本质上是一个二分类（或排序）问题。对于偏好对 (prompt, chosen, rejected)，模型应该给 chosen 更高的分数，给 rejected 更低的分数。常用的训练目标是 Bradley-Terry 模型：

```
P(chosen > rejected) = exp(r_chosen) / (exp(r_chosen) + exp(r_rejected))
```

其中 r_chosen 和 r_rejected 是奖励模型对两个回复的评分。训练目标是最小化：

```
loss = -log(sigmoid(r_chosen - r_rejected))
```

这个 loss 函数的直觉很简单：让 r_chosen 和 r_rejected 之间的差距尽量大，让模型明确地认为 chosen 更好。

**奖励模型的结构**：

奖励模型通常从 SFT 模型初始化——把语言模型的最后一层（lm_head）替换为一个输出标量的线性层，这个标量就是奖励分数。这样做的原因是：评分能力需要建立在语言理解能力之上，从 SFT 模型出发可以复用已有的语言理解能力。

### 2.3 阶段三：PPO 强化学习

PPO（Proximal Policy Optimization）是 RLHF 的核心强化学习算法。在这个阶段，我们使用第二阶段训练好的奖励模型作为"环境"，用 PPO 来进一步优化语言模型的策略。

**为什么需要 PPO？**

在 SFT 之后，模型已经能生成不错的回复。但 SFT 有一个隐含的问题：它假设训练数据中的所有回复都是同等优秀的。现实中，对于同一个问题，不同回复的质量差异很大。PPO 通过奖励模型提供的细粒度反馈，让模型学会生成"更好的回复"。

**PPO 的核心要素**：

把语言模型视为一个"策略"（policy）——给定状态（即到目前为止生成的文本），策略决定了下一个动作（生成哪个 token）。PPO 的目标是最大化期望奖励，同时保持策略不要偏离初始策略（SFT 模型）太远。

**KL 散度约束**：

为什么需要"不要偏离太远"？因为奖励模型是"近似的"——它只在训练数据覆盖的范围内准确。如果语言模型在大幅偏离原始分布的文本上生成，奖励模型的评分可能非常不准确（甚至给一些无意义但"看起来好"的文本打高分）。这被称为"奖励黑客"（reward hacking）问题。

KL 散度（Kullback-Leibler Divergence）衡量两个概率分布的差异。PPO 的目标函数中加入了 KL 惩罚项：

```
objective = E[reward(response)] - beta * KL(π_new || π_old)
```

其中：
- E[reward(response)] 是奖励模型给生成回复的期望评分（我们希望它越高越好）
- KL(π_new || π_old) 是新策略和旧策略之间的 KL 散度（我们希望它越小越好，即不要偏离太远）
- beta 是一个超参数，控制"优化奖励"和"保持稳定"之间的权衡

**PPO 的训练流程**：

1. 用当前策略对一批 prompt 生成多个回复
2. 用奖励模型对每个回复打分
3. 计算每个 token 的优势函数（advantage）——这个 token 生成的"方向"比平均好多少
4. 用 PPO 的裁剪目标函数更新策略
5. 同时计算并监控 KL 散度，确保策略不会偏离太远
6. 重复以上步骤

**PPO 的挑战**：

RLHF + PPO 虽然效果好，但也有明显的工程挑战：
- 需要同时维护多个模型（策略模型、参考模型、奖励模型、价值模型），显存需求巨大
- PPO 训练本身不稳定，对超参数敏感
- 调试困难——当训练出问题时，很难定位是 SFT 的问题、奖励模型的问题还是 PPO 的问题

这些挑战催生了 DPO——一种更简单的替代方案。

---

## 3. DPO：直接偏好优化

### 3.1 DPO 的简化哲学

DPO（Direct Preference Optimization）的核心洞察是：**你不需要显式地训练一个奖励模型，然后用 PPO 去最大化这个奖励。你可以直接在偏好数据上优化语言模型，让它隐式地学会人类的偏好。**

这个洞察来自一个数学推导。RLHF 的 PPO 优化目标在理论上有一个闭式解（closed-form solution）——最优策略可以表达为奖励函数的函数。将这个关系代入偏好数据的 Bradley-Terry 模型，可以得到一个直接优化语言模型的 loss 函数，无需单独训练奖励模型。

DPO 的 loss 函数为：

```
L_DPO = -E[log(sigmoid(beta * (log(π(chosen)/π_ref(chosen)) - log(π(rejected)/π_ref(rejected)))))]
```

这个公式看起来很复杂，但直觉很简单：
- π(chosen) 是当前模型给 chosen 回复的概率（我们希望它越大越好）
- π_ref(chosen) 是参考模型（SFT 模型）给 chosen 回复的概率
- log(π(chosen)/π_ref(chosen)) 表示当前模型相对于参考模型对 chosen 回复的偏好提升
- sigmoid(beta * (...)) 将偏好差映射到 [0,1] 的概率
- 整体目标是让模型对 chosen 回复的偏好明显高于对 rejected 回复的偏好

### 3.2 DPO 相比 RLHF 的优势

**简洁性**：
- RLHF：SFT → 训练奖励模型 → PPO（三个步骤，需要维护多个模型）
- DPO：SFT → DPO 训练（两个步骤，只需要参考模型和训练模型）

**稳定性**：
- DPO 是标准的监督学习，loss 收敛稳定
- PPO 是强化学习，训练可能不稳定、对超参数敏感

**可解释性**：
- DPO 的 loss 函数直接反映了"模型是否更偏好 chosen 回复"
- RLHF 中，问题可能出在奖励模型的偏差上，而奖励模型的内部行为是黑箱的

**数据效率**：
- DPO 需要的偏好数据量与 RLHF 相当或更少
- 不需要额外的奖励模型训练数据

### 3.3 DPO 的局限性

DPO 并非在所有方面都优于 RLHF：

- **在线 vs 离线**：DPO 是离线方法——它只在已有的偏好数据上训练。RLHF 的 PPO 是在线的——它可以在训练过程中生成新的回复并由奖励模型实时打分，理论上能探索更大的策略空间。
- **对偏好数据的质量更敏感**：DPO 直接学习偏好对中的差异，如果偏好标注质量不高（噪声大），DPO 的性能会显著下降。RLHF 中的奖励模型可以一定程度上平滑噪声。
- **缺少显式的探索**：PPO 在训练中会主动探索不同的生成策略，而 DPO 完全依赖已有的偏好数据。当偏好数据覆盖的"回复空间"有限时，DPO 的泛化能力可能不如 PPO。

---

## 4. DPO 实战：使用 TRL 库训练

### 4.1 TRL 库简介

TRL（Transformer Reinforcement Learning）是 HuggingFace 生态中的一个库，专门用于 LLM 的强化学习和对齐训练。它提供了 DPO、PPO、KTO 等对齐方法的即用型 Trainer。

安装:
```bash
pip install trl
```

TRL 的 `DPOTrainer` 提供了与 Transformers 的 `Trainer` 类似的 API，但专门为 DPO 训练做了优化：
- 自动处理 chosen 和 rejected 的 tokenization
- 内置参考模型的加载和管理
- 自动计算 DPO loss

### 4.2 DPO 数据格式

DPO 训练数据需要每条包含三个字段：
- `prompt`：用户指令（通常包含 Chat Template 格式的 system + user 部分）
- `chosen`：被标注员选中的更好的回复
- `rejected`：被标注员拒绝的更差的回复

```json
{
    "prompt": "<|im_start|>system\n你是一个有帮助的AI助手。<|im_end|>\n<|im_start|>user\n解释一下什么是机器学习。<|im_end|>\n<|im_start|>assistant\n",
    "chosen": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习和改进，而无需进行明确的编程...（详细且深入的回答）",
    "rejected": "机器学习就是让机器学习的。"
}
```

注意 `prompt` 字段的格式：它应该包含到 assistant 开始标记为止的所有内容。这意味着：
1. System prompt（如果有）
2. User 输入
3. 特殊 token（如 `<|im_start|>assistant\n`）作为生成提示

`chosen` 和 `rejected` 字段只需要包含 assistant 的实际回复内容（不需要特殊 token）。

### 4.3 完整的 DPO 训练代码

```python
"""
使用 TRL 库进行 DPO 训练的完整示例。

DPO 训练流程:
1. 加载 SFT 模型作为参考模型（ref_model）
2. 准备偏好数据（prompt + chosen + rejected）
3. 配置 DPOTrainer
4. 训练

依赖: pip install trl transformers peft accelerate datasets bitsandbytes
"""
import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)
from trl import DPOTrainer, DPOConfig
from datasets import Dataset


# ============================================================
# 1. 加载模型
# ============================================================

def load_models(model_name: str, use_qlora: bool = False):
    """
    加载 SFT 模型（作为训练起点）和 tokenizer。

    注意: DPO 需要两个模型——训练模型和参考模型。
    参考模型是冻结的 SFT 模型，用于计算 KL 散度。
    在 DPOTrainer 中，如果 ref_model 为 None，
    它会自动创建一个训练模型的副本作为参考模型。

    返回: (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["torch_dtype"] = torch.bfloat16
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


# ============================================================
# 2. 偏好数据准备
# ============================================================

def format_dpo_sample(example: dict, tokenizer, system_prompt: str = "") -> dict:
    """
    将原始偏好数据格式化为 DPO 训练所需的格式。

    输入格式（原始数据）:
    {
        "instruction": "用户问题",
        "chosen": "更好的回复",
        "rejected": "更差的回复",
    }

    输出格式（DPO 需要）:
    {
        "prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n",
        "chosen": "更好的回复",
        "rejected": "更差的回复",
    }
    """
    instruction = example["instruction"]
    chosen = example["chosen"]
    rejected = example["rejected"]

    # 构建 prompt 部分的 messages（不包含 assistant 回复）
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": instruction})

    # 使用 apply_chat_template 构建 prompt
    # add_generation_prompt=True 会在末尾添加生成提示（如 <|im_start|>assistant\n）
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def load_dpo_data(data_path: str, tokenizer, system_prompt: str = "") -> Dataset:
    """加载并格式化 DPO 偏好数据。"""
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    dataset = Dataset.from_list(data)
    dataset = dataset.map(
        lambda x: format_dpo_sample(x, tokenizer, system_prompt),
        remove_columns=dataset.column_names,
        desc="Formatting DPO data",
    )
    return dataset


# ============================================================
# 3. LoRA 配置
# ============================================================

def configure_lora(model, r: int = 16, lora_alpha: int = 32) -> PeftModel:
    """为模型配置 LoRA adapter。"""
    # 自动检测线性层
    from peft import LoraConfig, get_peft_model

    # 常见的 target modules
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    # 验证哪些模块实际存在
    available = {name.split(".")[-1] for name, _ in model.named_modules()
                 if isinstance(_, torch.nn.Linear)}
    valid_targets = [m for m in target_modules if m in available]

    if not valid_targets:
        valid_targets = list(available)
        print(f"自动检测到的线性层: {valid_targets}")

    print(f"DPO LoRA target_modules: {valid_targets}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=valid_targets,
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ============================================================
# 4. DPO 训练
# ============================================================

def train_dpo(
    model,
    tokenizer,
    train_dataset: Dataset,
    output_dir: str = "./dpo_output",
    beta: float = 0.1,                    # DPO 温度参数（控制偏离参考模型的程度）
    learning_rate: float = 5e-5,           # DPO 学习率通常比 SFT 稍小
    num_epochs: int = 1,                   # DPO 通常 1-3 个 epoch（数据量小时可稍多）
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    max_length: int = 2048,
    max_prompt_length: int = 512,          # prompt 部分的最大长度
    warmup_ratio: float = 0.1,
    logging_steps: int = 10,
):
    """
    执行 DPO 训练。

    参数:
        model: 配置了 LoRA 的训练模型
        tokenizer: tokenizer
        train_dataset: DPO 偏好数据集
        output_dir: 输出目录
        beta: DPO 温度参数
            - beta=0: 不做 KL 约束（相当于只用偏好数据做 SFT）
            - beta 小（0.01-0.1）: 允许较大偏离，偏好信号更强
            - beta 大（0.5-1.0）: 严格约束，保持更接近参考模型
            推荐从 0.1 开始实验
        learning_rate: 学习率
        num_epochs: 训练轮数
    """
    print(f"DPO 训练参数: beta={beta}, lr={learning_rate}, epochs={num_epochs}")

    # DPO 训练参数
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # 学习率
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",

        # 优化器
        optim="adamw_torch",
        weight_decay=0.01,

        # 精度
        bf16=True,

        # DPO 特定参数
        beta=beta,                           # KL 惩罚系数
        max_length=max_length,               # 总序列最大长度
        max_prompt_length=max_prompt_length, # prompt 最大长度

        # 保存和日志
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=logging_steps,
        report_to="none",

        # 数据加载
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    # 创建 DPOTrainer
    # 注意: ref_model 参数可选。如果不提供，DPOTrainer 会
    # 自动创建一个训练模型的冻结副本作为参考模型。
    # 这意味着需要约 2 倍的模型显存（训练模型 + 参考模型）。
    # 如果显存不够，可以:
    # 1. 使用更小的模型
    # 2. 使用 QLoRA（两个模型都以 4-bit 加载）
    # 3. 使用 ref_model=None 的分阶段加载策略
    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # None 表示自动创建参考模型副本
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    # 开始训练
    print(f"\n{'='*50}")
    print(f"  开始 DPO 训练")
    print(f"  训练样本: {len(train_dataset)}")
    print(f"  beta: {beta}")
    print(f"{'='*50}\n")

    trainer.train()

    # 保存
    final_output = f"{output_dir}/final"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"\nDPO 训练完成！模型已保存至 {final_output}")

    return trainer


# ============================================================
# 5. 评估 DPO 效果
# ============================================================

def evaluate_dpo(
    sft_model_path: str,
    dpo_model_path: str,
    base_model_name: str,
    test_prompts: list[str],
    system_prompt: str = "你是一个有帮助的AI助手。",
) -> dict:
    """
    对比 SFT 模型和 DPO 模型在同一批测试 prompt 上的生成质量。

    返回:
        {
            "sft_responses": [...],
            "dpo_responses": [...],
            "comparisons": [{"prompt": ..., "sft": ..., "dpo": ..., "winner": "dpo/sft/tie"}, ...]
        }
    """
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def generate_for_model(model_path: str, prompts: list[str]) -> list[str]:
        """用指定模型生成回复。"""
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()

        responses = []
        for prompt_text in prompts:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
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

            gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(response)

        del base_model
        del model
        torch.cuda.empty_cache()
        return responses

    print("生成 SFT 模型回复...")
    sft_responses = generate_for_model(sft_model_path, test_prompts)

    print("生成 DPO 模型回复...")
    dpo_responses = generate_for_model(dpo_model_path, test_prompts)

    # 打印对比
    comparisons = []
    for i, (prompt, sft_r, dpo_r) in enumerate(zip(test_prompts, sft_responses, dpo_responses)):
        print(f"\n{'='*50}")
        print(f"对比 #{i+1}")
        print(f"Prompt: {prompt}")
        print(f"\n--- SFT 回复 ---")
        print(sft_r[:300])
        print(f"\n--- DPO 回复 ---")
        print(dpo_r[:300])

        # 可以在这里加入人工判断或 LLM-as-Judge
        winner = input("\n哪个更好？(sft/dpo/tie): ").strip().lower()
        comparisons.append({
            "prompt": prompt,
            "sft_response": sft_r,
            "dpo_response": dpo_r,
            "winner": winner,
        })

    dpo_wins = sum(1 for c in comparisons if c["winner"] == "dpo")
    sft_wins = sum(1 for c in comparisons if c["winner"] == "sft")
    ties = sum(1 for c in comparisons if c["winner"] == "tie")

    print(f"\n===== 对比结果 =====")
    print(f"DPO 更好: {dpo_wins}/{len(comparisons)}")
    print(f"SFT 更好: {sft_wins}/{len(comparisons)}")
    print(f"平分:     {ties}/{len(comparisons)}")

    return {
        "sft_responses": sft_responses,
        "dpo_responses": dpo_responses,
        "comparisons": comparisons,
    }


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    # 配置
    BASE_MODEL = "Qwen/Qwen2.5-0.5B"      # 基座模型
    SFT_ADAPTER = "./sft_output/final"      # SFT 后的 adapter 路径
    DPO_DATA = "dpo_train_data.jsonl"       # DPO 偏好数据
    DPO_OUTPUT = "./dpo_output"             # DPO 输出目录

    # 1. 加载 SFT 模型（如果已有 SFT adapter，从这里出发）
    print("加载 SFT 模型...")
    model, tokenizer = load_models(BASE_MODEL, use_qlora=False)
    model = PeftModel.from_pretrained(model, SFT_ADAPTER)
    # 注意: PeftModel.from_pretrained 加载后 adapter 参数默认是可训练的
    print("SFT adapter 已加载")

    # 2. 加载偏好数据
    print("加载 DPO 偏好数据...")
    train_dataset = load_dpo_data(DPO_DATA, tokenizer,
                                  system_prompt="你是一个有帮助的AI助手。")
    print(f"偏好数据: {len(train_dataset)} 对")

    # 3. DPO 训练
    trainer = train_dpo(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=DPO_OUTPUT,
        beta=0.1,
        learning_rate=5e-5,
        num_epochs=1,
    )

    print("\nDPO 训练完成！")
```

### 4.4 DPO 训练的关键参数解读

**beta（KL 惩罚系数）**：

这是 DPO 中最关键的参数。它控制模型可以多大程度地偏离参考模型（SFT 模型）。

- beta 很小（如 0.01）：模型可以大幅调整策略，偏好信号的影响很强。在偏好数据质量高的情况下可以使用较小的 beta。
- beta 较大（如 0.5-1.0）：模型更倾向于保持参考模型的原始行为，偏好信号的影响较弱。在担心过拟合或偏好数据质量不确定时使用。
- 推荐起点：beta = 0.1。然后根据训练过程中的指标（chosen/rejected 的概率差异、KL 散度）进行调整。

**如何判断 beta 是否合适？**

- 如果 chosen 的 log probability 比 rejected 高很多（差异 > 5），且 KL 散度很小（< 1），说明偏好信号足够强但模型变化不大，beta 可以减小一些
- 如果 KL 散度快速增长（> 10），模型在剧烈偏离，beta 可能需要增大

**max_prompt_length vs max_length**：

- `max_prompt_length`：prompt 部分的最大 token 数，超过部分会被截断
- `max_length`：prompt + chosen/rejected 的总序列最大长度

合理的设置是：max_prompt_length 大约为 max_length 的 25-33%。这样留给 chosen/rejected 回复的长度空间大约是 prompt 的 2-3 倍，符合大多数对话中回复比指令长的实际情况。

---

## 基础练习

1. **偏好数据集构建**：设计一个标注协议，对同一批 50 个指令每个生成 2 个回复（一个高质量、一个低质量），构建偏好对数据集。要求定义清晰的标注标准。

2. **DPO Loss 手动计算**：手动实现 DPO 的 loss 函数。输入 chosen 和 rejected 的 log probabilities（模拟值），计算 DPO loss。对比不同 beta 值下 loss 的变化。

3. **理解 Bradley-Terry 模型**：用代码模拟 Bradley-Terry 模型的概率分布。给两个选手设定不同的"实力"参数，计算模型预测的胜率，理解奖励分数和胜率的关系。

## 进阶练习

1. **DPO vs SFT 效果对比实验**：在同一批测试 prompt 上，对比 SFT 模型和 DPO 模型的生成质量。使用 LLM-as-Judge 进行 A/B 盲评，统计 DPO 在帮助性、安全性、真实性三个维度的提升。

2. **偏好数据质量对 DPO 的影响**：故意构造不同质量的偏好数据（随机标注、弱偏好、强偏好），对比 DPO 训练效果，分析偏好数据质量如何影响最终模型。

---

## 常见错误

1. **混淆 SFT 数据和 DPO 数据**：SFT 数据是"指令-回复对"，DPO 数据是"指令-(好回复, 差回复)对"。两者的数据格式完全不同，不能混用。

2. **beta 设置过大或过小**：beta 太大（>1.0）导致偏好信号几乎没用，模型基本不变；beta 太小（<0.01）导致模型过度追赶偏好信号，丧失通用能力。推荐从 0.1 开始，观察训练日志调整。

3. **忽略参考模型的显存开销**：DPO 需要同时加载训练模型和参考模型，显存需求约是 SFT 的 2 倍。如果使用 QLoRA，显存压力可以缓解。如果显存不够，考虑使用 `ref_model=None` 但减小 batch size。

4. **偏好数据中 chosen 和 rejected 差异太小**：如果两个回复几乎差不多，标注员靠"掷骰子"决定哪个是 chosen，这样的数据对 DPO 几乎没有价值。数据质量的前提是 chosen 和 rejected 有明确的差异。

5. **偏好数据分布与实际使用场景不匹配**：偏好数据全是"长篇解释"场景下的偏好，但实际用户更多问简短的事实性问题。DPO 后的模型可能在长篇场景下表现很好，但短回复场景反而退步。

6. **prompt 格式错误**：DPO 的 prompt 字段必须以生成提示结束（如 `<|im_start|>assistant\n`），但 chosen 和 rejected 字段不应包含这个提示。如果格式错误，模型可能无法正确区分"输入"和"输出"的边界。

7. **DPO 之后不再需要 SFT 数据**：虽然 DPO 可以直接从 SFT 模型出发，但 DPO 主要调整的是"偏好"而非"能力"。如果 SFT 模型本身基础不够好，DPO 也救不了。

8. **将 DPO 当作 SFT 的替代品**：DPO 的目标是让模型学会"A 比 B 更好"，而不是"如何正确执行指令"。如果模型连基本的指令遵循都做不好，应该先用 SFT 解决基础能力问题。

9. **忽略 DPO 训练中的 chosen/rejected 概率监控**：训练过程中，chosen 的 log probability 应该在上升，rejected 的在下降。如果两者都在下降（或上升），说明训练出了问题——可能是学习率太大或数据质量有问题。

10. **在偏好数据中混入了模型自己生成的回复**：如果用同一个模型生成 chosen 和让人类标注，chosen 可能只是"模型本来就擅长生成的"，而不是"人类真正偏好的"。偏好数据应该尽可能反映人类真实偏好，而非模型分布。

---

## 本章小结

本章我们从 SFT 的局限性出发，理解了为什么需要偏好对齐——SFT 让模型"会做"，偏好对齐让模型"做得好"。我们用一个贯穿始终的直觉——"相对判断比绝对判断更容易标注也更富含信息"——理解了偏好对齐的核心哲学。

在 RLHF 部分，我们详细描述了三个阶段的流程和目标：SFT 建立基础能力、奖励模型学习评分函数、PPO 用强化学习优化策略。我们特别探讨了 KL 散度约束的必要性——防止模型为了追逐奖励分数而偏离太远、丧失通用能力。

在 DPO 部分，我们展示了这种更简洁的方法如何从数学推导出发，消除了对独立奖励模型的需求，直接将偏好信号注入语言模型。我们通过完整的 TRL 库代码展示了 DPO 训练的端到端流程。

下一章，我们将学习如何科学地评估微调模型的效果——从自动化 benchmark 到 LLM-as-Judge 到 A/B 对比统计检验。
