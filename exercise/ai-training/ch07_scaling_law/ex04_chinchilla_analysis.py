"""
练习 4（进阶）：Chinchilla 最优分析
===================================

分析公开模型的训练数据量和规模，判断它们离 Chinchilla 最优有多远。

你需要完成:
1. 收集公开模型的参数量和训练 token 量数据
2. 计算每个模型的"Chinchilla 最优数据量":
   D_opt = 20 × N（每个参数约 20 个训练 token）
3. 计算"训练充裕度":
   sufficiency = D_actual / D_opt
   - > 1: 过度训练（数据充裕型）
   - ≈ 1: Chinchilla 最优
   - < 1: 训练不足（模型过大型）
4. 绘制散点图: 参数量 vs 训练 token 量
   标注 Chinchilla 最优线
   标注各模型的位置

公开模型数据（2023-2024年）:
    模型          | 参数量 (N) | 训练 Token 数 (D) | D/N 比
    GPT-3         | 175B       | 300B              | 1.7
    Chinchilla    | 70B        | 1.4T              | 20
    PaLM          | 540B       | 780B              | 1.4
    LLaMA 7B      | 6.7B       | 1.0T              | 149
    LLaMA 13B     | 13B        | 1.0T              | 77
    LLaMA 33B     | 33B        | 1.4T              | 42
    LLaMA 65B     | 65B        | 1.4T              | 21.5
    Falcon 40B    | 40B        | 1.0T              | 25
    Mistral 7B    | 7.3B       | ???               | ??
    LLaMA 2 7B    | 6.7B       | 2.0T              | 299

思考题:
- LLaMA 系列为什么选择"过度训练"？（提示: 推理成本 vs 训练成本）
- 如果你想训练一个 1B 参数、10B 参数或 100B 参数的模型，
  按照 Chinchilla 最优，各需要多少训练 token？
- 在实际项目中，"最优"的计算分配是否一定是"最好"的选择？
"""

import numpy as np
import matplotlib.pyplot as plt


# TODO: 模型数据
KNOWN_MODELS = [
    # (name, params_billions, tokens_billions)
    ("GPT-3", 175, 300),
    ("Chinchilla", 70, 1400),
    ("PaLM", 540, 780),
    ("LLaMA 7B", 6.7, 1000),
    ("LLaMA 13B", 13, 1000),
    ("LLaMA 33B", 33, 1400),
    ("LLaMA 65B", 65, 1400),
    ("Falcon 40B", 40, 1000),
    ("LLaMA 2 7B", 6.7, 2000),
    ("LLaMA 2 13B", 13, 2000),
    ("LLaMA 2 70B", 69, 2000),
]


# TODO: 分析 Chinchilla 最优
def analyze_chinchilla_optimal(models):
    """
    TODO: 计算每个模型的 Chinchilla 富裕度。

    对每个模型:
    1. D_opt = 20 × N（Chinchilla 建议的最优数据量）
    2. sufficiency = D_actual / D_opt
    3. 分类: sufficiency > 2 → "数据充裕"
            sufficiency ≈ 1 → "Chinchilla 最优"
            sufficiency < 0.5 → "训练不足"

    打印分析表格。
    """
    pass


# TODO: 可视化
def plot_chinchilla_analysis(models):
    """
    TODO: 绘制 log-log 散点图。

    X 轴: 参数量 (log scale)
    Y 轴: 训练 token 数 (log scale)

    绘制:
    - 每个模型的数据点（标注名称）
    - Chinchilla 最优线: D = 20 × N
    - Kaplan 建议线: D ∝ N^0.74
    """
    pass


# TODO: 计算训练所需的 token 量
def recommend_training_tokens(target_params, sufficiency=20):
    """
    TODO: 给目标参数量推荐训练 token 量。

    参数:
        target_params: 目标参数量（Billion）
        sufficiency: D/N 比，默认 20（Chinchilla 最优）

    返回:
        recommended_tokens: 推荐的训练 token 数（Billion）
    """
    pass


if __name__ == "__main__":
    analyze_chinchilla_optimal(KNOWN_MODELS)
    plot_chinchilla_analysis(KNOWN_MODELS)
