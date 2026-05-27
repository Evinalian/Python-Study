"""
练习 1: LoRA 配置与加载 —— 理解 PEFT 库的基本用法

场景:
    你需要为 Qwen2.5-0.5B 模型配置 LoRA 微调。
    在此之前，你需要先理解 LoRA 配置中的每个参数如何影响可训练参数量。

要求:
    1. 加载一个因果语言模型（Qwen2.5-0.5B 或其他支持的小模型）
    2. 尝试不同的 LoRA 配置，观察可训练参数量的变化
    3. 打印每种配置的详细信息

TODO:
    1. 实现 load_base_model(model_name) 函数:
       - 加载模型和 tokenizer
       - 设置 pad_token
       - 返回 model, tokenizer

    2. 实现 apply_lora(model, lora_config) 函数:
       - 使用 get_peft_model 应用 LoRA
       - 打印可训练参数信息（总数、可训练数、占比）
       - 返回 PeftModel

    3. 实现 compare_lora_configs(model, configs) 函数:
       - configs 是一个列表，每项为 {"name": str, "config": LoraConfig}
       - 对每个配置应用 LoRA 并统计:
         * 可训练参数量
         * 可训练参数占比
         * 理论上每种 target_module 贡献的参数量
       - 打印对比表格

    4. 实现 analyze_target_modules(model) 函数:
       - 打印所有 nn.Linear 层的名称和形状
       - 帮助你理解哪些层适合应用 LoRA

    5. 准备至少 4 种不同的 LoRA 配置进行对比:
       - 配置 A: r=4, alpha=8, 只对 q_proj 和 v_proj
       - 配置 B: r=8, alpha=16, 对 q/k/v/o_proj
       - 配置 C: r=16, alpha=32, 对 q/k/v/o_proj + gate/up/down_proj
       - 配置 D: r=64, alpha=128, 对 q/k/v/o_proj + gate/up/down_proj

    6. 思考题（注释回答）:
       - 为什么 target_modules 的选择对参数量影响更大（而不是 rank）？
       - 在什么情况下你应该选择只对 attention 层应用 LoRA，而不是加上 FFN 层？
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# TODO 1: 加载基座模型
# ============================================================
def load_base_model(model_name: str):
    """
    加载基座模型和 tokenizer。

    返回: (model, tokenizer)
    """
    # TODO: 加载 tokenizer（trust_remote_code=True）
    # TODO: 设置 pad_token = eos_token（如果 pad_token 为 None）
    # TODO: 加载模型（torch_dtype=torch.bfloat16, device_map="auto"）
    pass


# ============================================================
# TODO 2: 应用 LoRA
# ============================================================
def apply_lora(model, lora_config: LoraConfig):
    """
    应用 LoRA 配置到模型。

    返回: PeftModel
    """
    # TODO: 调用 get_peft_model(model, lora_config)
    # TODO: 调用 model.print_trainable_parameters()
    # TODO: 返回 model
    pass


# ============================================================
# TODO 3: 对比不同 LoRA 配置
# ============================================================
def compare_lora_configs(base_model_name: str, configs: list[dict]):
    """
    对比多种 LoRA 配置的参数效率。

    configs 格式:
    [
        {
            "name": "基础配置 (r=4, q+v)",
            "config": LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], ...)
        },
        ...
    ]
    """
    # TODO: 对每种配置:
    #   1) 重新加载基座模型
    #   2) 应用 LoRA
    #   3) 获取可训练参数量和占比
    #   4) 将结果记录到列表中
    # TODO: 打印对比表格（格式对齐）
    #   列: 配置名 | rank | alpha | target_modules | 可训练参数 | 占比
    pass


# ============================================================
# TODO 4: 分析目标模块
# ============================================================
def analyze_target_modules(model):
    """
    打印模型中所有 nn.Linear 层的名称和形状。

    输出示例:
    model.layers.0.self_attn.q_proj:    torch.Size([896, 896])
    model.layers.0.self_attn.k_proj:    torch.Size([896, 128])
    ...
    """
    # TODO: 遍历 model.named_modules()
    # TODO: 过滤出 isinstance(module, torch.nn.Linear)
    # TODO: 打印模块名称和 weight.shape
    pass


# ============================================================
# TODO 5: 准备对比配置
# ============================================================
def get_comparison_configs() -> list[dict]:
    """返回多种 LoRA 配置用于对比。"""
    # TODO: 创建至少 4 种不同的 LoraConfig
    # TODO: 每种配置有明确的名称和参数
    # TODO: 注意 task_type=TaskType.CAUSAL_LM
    configs = [
        # {"name": "配置A: ...", "config": LoraConfig(...)},
    ]
    return configs


# ============================================================
# TODO 6: 思考题
# ============================================================
"""
Q1: 为什么 target_modules 的选择对参数量影响更大（而不是 rank）？
A1: TODO

Q2: 在什么情况下你应该选择只对 attention 层应用 LoRA，而不是加上 FFN 层？
A2: TODO

Q3: 如果两个配置的可训练参数量相同，但一个 rank=8 覆盖全模块，另一个 rank=64 只覆盖 attention——哪种通常更好？
A3: TODO
"""


if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # 使用小模型快速测试

    print("=" * 50)
    print("  LoRA 配置对比实验")
    print("=" * 50)

    # 1. 加载模型并分析模块
    # model, tokenizer = load_base_model(MODEL_NAME)
    # analyze_target_modules(model)

    # 2. 对比不同配置
    # configs = get_comparison_configs()
    # compare_lora_configs(MODEL_NAME, configs)

    print("\n请完成以上 TODO 后取消注释运行。")
