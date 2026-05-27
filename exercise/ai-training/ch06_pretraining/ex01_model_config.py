"""
练习 1：修改模型配置
====================

基于第06章教程中的 MiniGPT，修改模型配置参数并观察影响。

你需要完成:
1. 使用教程中的 MiniGPT 训练代码
2. 选择 3 种不同配置（小型、中型、大型），训练并比较:

   小型 (约 10M 参数):
       d_model=256,  n_layers=4,  n_heads=4
   中型 (约 34M 参数):
       d_model=512,  n_layers=8,  n_heads=8
   大型 (约 85M 参数):
       d_model=768,  n_layers=12, n_heads=12

3. 记录每种配置的:
   - 总参数量和非 embedding 参数量
   - 训练速度 (tokens/sec)
   - 显存占用
   - 最终验证 perplexity
4. 在 log-log 坐标上绘制 perplexity vs params 的关系

计算参数量的近似公式:
    P ≈ n_layers × (4×d_model² + 3×d_model×d_ff) + vocab_size×d_model
    d_ff ≈ 8/3 × d_model（SwiGLU）

思考题:
- 参数量从 10M 到 85M 增加了约 8.5 倍，验证 perplexity 降低了多少？
- 增加 n_layers 和增加 d_model 哪个对 perplexity 的影响更大？
- 训练速度和显存占用与参数量的关系是线性的吗？
"""

import torch
import math
import time
import matplotlib.pyplot as plt


# TODO: 复制教程中的 MiniGPT 和训练函数


# TODO: 定义不同配置
CONFIGS = {
    'small':  {'d_model': 256,  'n_layers': 4,  'n_heads': 4},
    'medium': {'d_model': 512,  'n_layers': 8,  'n_heads': 8},
    'large':  {'d_model': 768,  'n_layers': 12, 'n_heads': 12},
}


# TODO: 实验函数
def experiment_model_sizes():
    """
    TODO: 对每种配置，训练模型并记录指标。

    对每种配置:
    1. 创建模型
    2. 统计参数量
    3. 训练固定步数（如 500 步）
    4. 测量速度和显存
    5. 评估验证 perplexity
    6. 记录结果
    """
    pass


# TODO: 可视化
def plot_scaling_curve(results):
    """
    TODO: 在双对数坐标上绘制 Perplexity vs Params 的关系。
    同时绘制 Tokens/sec vs Params 的关系。
    """
    pass


if __name__ == "__main__":
    experiment_model_sizes()
