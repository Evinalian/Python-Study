"""
练习 1：Scaling Law 实验
=========================

使用 MiniGPT 训练不同规模的模型，拟合 Scaling Law。

你需要完成:
1. 训练 4 个不同规模的模型（如 5M、10M、20M、40M 参数）
   在相同数据上训练相同 token 数
2. 记录每个模型的最终验证 Loss
3. 在双对数坐标上绘制 Loss vs Params
4. 拟合幂律关系: L(N) = a × N^b
   - 取对数: log(L) = log(a) + b × log(N)
   - 用线性回归拟合 b（幂律指数）
5. 报告拟合参数和 R² 值
6. 外推: 预测 100M、500M、1B 参数模型的 Loss

拟合技巧:
    from scipy.optimize import curve_fit
    def power_law(x, a, b): return a * x**b

    或使用 sklearn:
    from sklearn.linear_model import LinearRegression
    在 log-log 数据上拟合

思考题:
- 你的幂律指数 b 是多少？和文献中报告的值（约 -0.05 到 -0.08）接近吗？
- 外推的可靠性如何？假设外推 10 倍规模，误差可能有多大？
- 为什么 Scaling Law 在 log-log 坐标上是直线？
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# TODO: 复制 MiniGPT 和训练代码（从第06章教程）


# TODO: 定义不同规模的模型配置
MODEL_CONFIGS = {
    '5M':  {'d_model': 128,  'n_layers': 4,  'n_heads': 4},
    '10M': {'d_model': 256,  'n_layers': 4,  'n_heads': 4},
    '20M': {'d_model': 256,  'n_layers': 8,  'n_heads': 8},
    '40M': {'d_model': 512,  'n_layers': 8,  'n_heads': 8},
}


# TODO: 训练实验
def run_scaling_experiment(configs, total_tokens=1_000_000):
    """
    TODO: 对每种配置训练模型，记录最终验证 Loss。

    参数:
        configs: dict {name: config_dict}
        total_tokens: 每个模型训练的总 token 数（保持一致）

    返回:
        params_list: list[float]，参数量
        loss_list: list[float]，对应的验证 Loss
    """
    pass


# TODO: 拟合 Scaling Law
def fit_scaling_law(params, losses):
    """
    TODO: 拟合幂律关系 L(N) = a × N^b。

    1. 定义幂律函数
    2. 使用 curve_fit 拟合
    3. 计算 R²
    4. 打印拟合参数

    返回:
        a, b: 拟合参数
        r_squared: R²
    """
    pass


# TODO: 外推
def extrapolate(a, b, target_params):
    """
    TODO: 用拟合的 Scaling Law 预测更大模型的 Loss。

    参数:
        a, b: 拟合参数
        target_params: list[float]，目标参数量

    返回:
        predicted_losses: list[float]
    """
    pass


# TODO: 可视化
def plot_scaling_law(params, losses, a, b, extrapolated_params, extrapolated_losses):
    """
    TODO: 绘制:
    1. 实验数据点
    2. 拟合曲线
    3. 外推预测点（用虚线连接）
    4. 标注拟合公式和 R²
    """
    pass


if __name__ == "__main__":
    pass
