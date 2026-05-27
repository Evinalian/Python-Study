"""
练习 1: LoRA 权重合并与验证

场景:
    你完成了一个模型的 LoRA 微调，现在需要将 adapter 合并回基座模型。
    你需要确保合并后的模型与"基座模型 + adapter"产生相同的输出。

要求:
    1. 使用 PEFT 库实现权重合并
    2. 验证合并前后的输出一致性
    3. 对比合并前后的模型大小和推理速度

TODO:
    1. 实现 merge_and_save(base_model_name, adapter_path, output_path):
       - 加载基座模型
       - 挂载 LoRA adapter
       - 调用 merge_and_unload()
       - 保存合并后的模型和 tokenizer
       - 返回合并后的模型

    2. 实现 compare_outputs(model_before_merge, model_after_merge, tokenizer, test_inputs):
       - model_before_merge: 基座模型 + adapter（未合并）
       - model_after_merge: 合并后的模型
       - 对相同的 test_inputs 生成回复
       - 计算输出 token 序列的完全匹配率
       - 计算输出文本的差异度（编辑距离或 BLEU）
       - 如果差异 > 阈值（如 1e-5），打印警告

    3. 实现 compare_model_size(base_model, adapter, merged_model):
       - 打印三个"模型"的信息:
         * 基座模型参数量 / 文件大小
         * LoRA adapter 参数量 / 文件大小
         * 合并后模型参数量 / 文件大小
       - 验证: 合并后的参数量 == 基座模型参数量

    4. 实现 compare_inference_speed(before, after, tokenizer, test_inputs, n_runs=10):
       - 测量合并前（基座 + adapter）和合并后的推理时间
       - 每个模型运行 n_runs 次取平均
       - 打印对比表格（平均时间、标准差、加速比）

    5. 思考题（注释回答）:
       - 为什么合并后推理速度会略微变快？
       - 合并后的模型能否再"拆分"回基座 + adapter？
       - 如果你计划频繁切换 adapter（如多任务），应该合并吗？
"""
import os
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ============================================================
# TODO 1: 权重合并
# ============================================================
def merge_and_save(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
) -> tuple:
    """
    将 LoRA adapter 合并到基座模型并保存。

    返回: (merged_model, tokenizer)
    """
    # TODO: 加载基座模型（torch_dtype=torch.bfloat16, device_map="auto"）
    # TODO: 加载 tokenizer
    # TODO: 挂载 LoRA adapter: PeftModel.from_pretrained(base_model, adapter_path)
    # TODO: 调用 model.merge_and_unload()
    # TODO: 保存合并后的模型: model.save_pretrained(output_path)
    # TODO: 保存 tokenizer
    # TODO: 返回 (merged_model, tokenizer)
    pass


# ============================================================
# TODO 2: 输出一致性验证
# ============================================================
def compare_outputs(
    base_model,       # 基座模型（未挂载 adapter 时）
    model_with_adapter,  # 基座 + adapter（未合并时）
    merged_model,     # 合并后的模型
    tokenizer,
    test_inputs: list[str],
    max_new_tokens: int = 64,
) -> dict:
    """
    验证合并前后三个模型的输出一致性:
    1. 基座模型（原始，无 adapter）
    2. 基座 + adapter（合并前）
    3. 合并后模型

    三者对相同的输入应该给出（几乎）相同的输出吗？
    - 1 和 2 应该不同（adapter 改变了行为）
    - 2 和 3 应该完全相同（合并只是改变了计算方式，不应改变结果）
    """
    results = {
        "base_vs_with_adapter": [],
        "with_adapter_vs_merged": [],
        "all_match": True,
    }

    # TODO: 对每个 test_input:
    #   1) 用三个模型分别生成
    #   2) 比较 2 和 3 的 token 序列是否完全一致
    #   3) 如果不一致，计算字符级别的差异
    #   4) 打印不一致的具体内容
    # TODO: 综合判断: 2 和 3 的差异应该在 1e-5 以下

    return results


# ============================================================
# TODO 3: 模型大小对比
# ============================================================
def compare_model_size(
    base_model_name: str,
    adapter_path: str,
    merged_model_path: str,
):
    """
    对比三种"模型"的参数量和文件大小。
    """
    # TODO: 加载基座模型，统计参数量
    # TODO: 加载 LoRA adapter，统计可训练参数量
    # TODO: 加载合并后模型，统计参数量
    # TODO: 获取三种模型的文件大小（os.path.getsize 或遍历目录）
    # TODO: 打印对比表格
    #   | 组件 | 参数量 | 文件大小 |
    #   | 基座模型 | 7.0B | 14.0 GB |
    #   | LoRA adapter | 16M | 64 MB |
    #   | 合并后 | 7.0B | 14.0 GB |
    # TODO: 验证合并后参数量 == 基座模型参数量
    pass


# ============================================================
# TODO 4: 推理速度对比
# ============================================================
def compare_inference_speed(
    model_with_adapter,
    merged_model,
    tokenizer,
    test_inputs: list[str],
    n_runs: int = 10,
) -> dict:
    """
    对比合并前后的推理速度。
    """
    # TODO: 预热（各跑一次，不计入统计）
    # TODO: 对每个模型，跑 n_runs 次，记录每次的时间
    # TODO: 计算平均时间和标准差
    # TODO: 打印对比表格
    #   | 模型 | 平均延迟 (ms) | 标准差 (ms) | 加速比 |
    #   | 基座+adapter | 120.5 | 3.2 | 1.0x |
    #   | 合并后 | 115.3 | 2.8 | 1.05x |
    pass


# ============================================================
# TODO 5: 思考题
# ============================================================
"""
Q1: 为什么合并后推理速度会略微变快？
A1: TODO
    提示: 合并后的前向传播只需要 W*x 一次矩阵乘法；
    未合并的需要 W*x + B*A*x 两次矩阵乘法（虽然 B*A*x 很小但也需要开销）

Q2: 合并后的模型能否再"拆分"回基座 + adapter？
A2: TODO
    提示: merge_and_unload 会删除 LoRA 的 A 和 B 矩阵。这是不可逆的。
    如果你想保留拆分能力，在合并前备份 adapter 权重。

Q3: 如果你计划频繁切换 adapter（如多任务），应该合并吗？
A3: TODO
    提示: 不合并。保留基座模型 + 多个 adapter 的方案更灵活：
    一个基座模型（14GB） + 10 个 adapter（各 64MB）只需 ~14.6GB，
    而 10 个合并后的模型需要 140GB。
"""


if __name__ == "__main__":
    print("=" * 50)
    print("  LoRA 权重合并与验证")
    print("=" * 50)

    # 配置
    BASE_MODEL = "Qwen/Qwen2.5-0.5B"
    ADAPTER_PATH = "./sft_output/final"  # 需要先有微调的 adapter
    OUTPUT_PATH = "./merged_model"
    TEST_INPUTS = [
        "解释一下什么是机器学习",
        "Python 中如何读取文件？",
    ]

    print(f"\n基座模型: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"输出路径: {OUTPUT_PATH}")

    # TODO: 取消注释完成流程
    # if os.path.exists(ADAPTER_PATH):
    #     merged_model, tokenizer = merge_and_save(BASE_MODEL, ADAPTER_PATH, OUTPUT_PATH)
    #     compare_model_size(BASE_MODEL, ADAPTER_PATH, OUTPUT_PATH)
    # else:
    #     print(f"Adapter 路径不存在: {ADAPTER_PATH}")
    #     print("请先完成第03章的 SFT 训练，获得一个 LoRA adapter。")

    print("\n请完成以上 TODO 后取消注释运行。")
