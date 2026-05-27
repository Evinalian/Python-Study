"""
练习 2: LoRA 参数量手动计算

场景:
    PEFT 库的 print_trainable_parameters() 告诉你可训练参数量，
    但你想深入理解这些参数是从哪来的——每个 target_module 贡献了多少参数。

要求:
    1. 手动计算理论上 LoRA 对每个模块贡献的参数量
    2. 将理论计算与 PEFT 库的实际输出进行对比
    3. 分析误差来源

TODO:
    1. 实现 get_linear_layer_shapes(model, target_modules) 函数:
       - 遍历模型的所有模块
       - 找到名称为 target_modules 中的 nn.Linear 层
       - 返回 {module_name: (in_features, out_features)} 字典

    2. 实现 calculate_lora_params(shapes, r) 函数:
       - 对于每个 (in_features, out_features):
         LoRA 参数 = r * in_features (A 矩阵) + r * out_features (B 矩阵)
       - 汇总所有模块的参数量
       - 返回 {"per_module": {...}, "total": N} 字典

    3. 实现 get_actual_lora_params(model) 函数:
       - 应用 LoRA 之后，遍历 model.named_parameters()
       - 只统计 requires_grad=True 的参数
       - 按模块聚合参数量
       - 返回 {"per_module": {...}, "total": N} 字典

    4. 实现 compare_theory_vs_actual(model, target_modules, r) 函数:
       - 分别调用 calculate_lora_params 和 get_actual_lora_params
       - 打印对比表格，计算偏差
       - 分析偏差来源（bias? LayerNorm? 嵌入层?）

    5. 实现 estimate_params_for_diff_ranks(model, target_modules, ranks) 函数:
       - 对不同的 r 值（如 [1, 2, 4, 8, 16, 32, 64, 128]）计算理论 LoRA 参数量
       - 绘制 ASCII 柱状图或用表格展示 r 与参数量的关系
       - 标记参数量的"收益递减拐点"

    6. 思考题（注释回答）:
       - 为什么 r 从 8 增加到 16 参数量翻倍，但效果通常不会翻倍？
       - target_modules 中有些模块的 in/out features 不对称（如 q_proj 和 k_proj 可能有不同形状），这如何影响 LoRA 的参数分配效率？
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# TODO 1: 获取线性层形状
# ============================================================
def get_linear_layer_shapes(model, target_module_names: list[str]) -> dict:
    """
    获取指定 target_modules 的线性层的形状信息。

    返回:
        {
            "model.layers.0.self_attn.q_proj": {"in_features": 896, "out_features": 896},
            ...
        }
    """
    # TODO: 遍历 model.named_modules()
    # TODO: 对每个 nn.Linear 层，检查其名称后缀是否在 target_module_names 中
    #   提示：模块全名可能是 "model.layers.0.self_attn.q_proj"
    #   target_module_names 中可能是 "q_proj"
    #   需要检查名称是否以 target_module 结尾
    # TODO: 记录 in_features 和 out_features
    pass


# ============================================================
# TODO 2: 理论计算 LoRA 参数量
# ============================================================
def calculate_lora_params(shapes: dict, r: int) -> dict:
    """
    根据线性层形状和 rank 理论计算 LoRA 参数量。

    LoRA 对每个线性层 W (d_out, d_in) 添加:
    - A 矩阵: (r, d_in) 参数 → r * d_in 个
    - B 矩阵: (d_out, r) 参数 → d_out * r 个
    总计: r * (d_in + d_out) 个参数

    返回:
        {
            "per_module": {"q_proj": N1, "v_proj": N2, ...},
            "total": N_total
        }
    """
    # TODO: 对 shapes 中的每个模块计算参数量
    # TODO: 按模块名称聚合（同名模块可能有多个，如 24 个 layer 各有一个 q_proj）
    # TODO: 返回分模块统计和总计
    pass


# ============================================================
# TODO 3: 获取 PEFT 库实际的可训练参数量
# ============================================================
def get_actual_lora_params(model) -> dict:
    """
    从应用了 LoRA 的模型中统计实际可训练参数。

    返回:
        {
            "per_module": {"lora_A": N1, "lora_B": N2, ...},
            "total": N_total
        }
    """
    # TODO: 遍历 model.named_parameters()
    # TODO: 只统计 requires_grad=True 的参数
    # TODO: 按参数名的前缀聚合（如所有 "lora_A" 归为一组）
    #   提示: LoRA 参数名通常包含 "lora_A" 和 "lora_B"
    # TODO: 返回分模块统计和总计
    pass


# ============================================================
# TODO 4: 理论 vs 实际对比
# ============================================================
def compare_theory_vs_actual(model, target_modules: list[str], r: int):
    """
    对比理论计算的 LoRA 参数量和 PEFT 库实际输出。

    打印对比表格:
    模块名 | 理论参数 | 实际参数 | 偏差 | 偏差%
    """
    # TODO: 获取 shapes
    # TODO: 计算理论值
    # TODO: 应用 LoRA 到 model
    # TODO: 获取实际值
    # TODO: 对比并打印表格
    # TODO: 分析偏差来源（可能的额外参数：bias? LayerNorm? 其他?）
    pass


# ============================================================
# TODO 5: 不同 rank 的参数量估算
# ============================================================
def estimate_params_for_diff_ranks(model, target_modules: list[str],
                                   ranks: list[int] = None):
    """
    估算不同 rank 下的 LoRA 参数量，分析"收益递减拐点"。

    参数:
        ranks: 要测试的 rank 列表

    输出:
        - 表格: rank | 可训练参数 | 占原始模型比例
        - ASCII 柱状图
        - 标记"拐点": 参数量增长开始加速（线性增长）的位置
    """
    if ranks is None:
        ranks = [1, 2, 4, 8, 16, 32, 64, 128]

    # TODO: 获取 shapes
    # TODO: 对每个 rank 计算理论参数量
    # TODO: 获取原始模型总参数量
    # TODO: 计算占比
    # TODO: 打印表格
    # TODO: 绘制 ASCII 柱状图
    #   rank  1: ▏    81920 ( 0.02%)
    #   rank  2: ▎   163840 ( 0.04%)
    #   ...
    # TODO: 分析拐点（rank 增大但参数占比仍然很低，说明 LoRA 效率高）
    pass


# ============================================================
# TODO 6: 思考题
# ============================================================
"""
Q1: 为什么 r 从 8 增加到 16 参数量翻倍，但效果通常不会翻倍？
A1: TODO

Q2: target_modules 中有些模块的 in/out features 不对称（如 q_proj 和 k_proj 可能有不同形状），这如何影响 LoRA 的参数分配效率？
A2: TODO

Q3: 如果你发现理论计算值和 PEFT 实际值有偏差（例如差了 0.1%），可能是什么原因？
A3: TODO
"""


if __name__ == "__main__":
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"

    print("=" * 50)
    print("  LoRA 参数量计算练习")
    print("=" * 50)

    # TODO: 取消注释以下代码完成实验
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    # )
    #
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # shapes = get_linear_layer_shapes(model, target_modules)
    # print(f"找到 {len(shapes)} 个线性层:")
    # for name, shape in list(shapes.items())[:5]:
    #     print(f"  {name}: in={shape['in_features']}, out={shape['out_features']}")
    #
    # theory = calculate_lora_params(shapes, r=8)
    # print(f"\n理论 LoRA 参数量 (r=8): {theory['total']:,}")
    #
    # estimate_params_for_diff_ranks(model, target_modules)

    print("\n请完成以上 TODO 后取消注释运行。")
