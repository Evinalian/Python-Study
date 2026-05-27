"""
练习 3：Loss 曲线可视化与异常检测
=================================

读取训练日志，用 matplotlib 可视化 loss 曲线，并实现基本的异常检测。

你需要完成:
1. 解析训练日志（CSV 或 WandB 导出的 JSON），提取 train_loss 和 val_loss
2. 绘制 loss 曲线
3. 实现异常检测:
   - 震荡检测: 相邻 N 步的 loss 标准差 > threshold
   - NaN 检测: loss is NaN
   - 平坦检测: 最近 N 步的 loss 变化 < threshold
   - 分叉检测: val_loss - train_loss 的差距持续增大
4. 在图上标注检测到的异常区域
5. 生成异常报告

异常检测的阈值设置:
- 震荡: rolling std > 0.5（可能需要根据你的数据调整）
- 平坦: 最近 100 步的 max_loss - min_loss < 0.01
- 分叉: (val_loss - train_loss) 最近 200 步增加 > 50%

思考题:
- 为什么仅依赖 loss 曲线诊断问题有时候不够？
- 除了 loss 曲线，还有哪些指标可以帮助诊断训练问题？
- 在实际训练中，遇到异常时应该先检查什么？
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt


# TODO: 解析训练日志
def load_training_log(log_file):
    """
    TODO: 从 CSV 或 JSON 加载训练日志。

    支持的格式:
    - CSV: step, train_loss, val_loss, ...
    - JSON: WandB 导出的 JSON 格式

    返回:
        steps: ndarray
        train_losses: ndarray
        val_losses: ndarray (可能比 train_losses 稀疏)
    """
    pass


# TODO: 绘制 loss 曲线
def plot_loss_curve(steps, train_losses, val_losses=None, anomalies=None):
    """
    TODO: 绘制 loss 曲线，可选地标注异常区域。

    参数:
        steps: 步数数组
        train_losses: 训练 loss
        val_losses: 验证 loss（可选）
        anomalies: 异常区域列表 [(start, end, type), ...]
    """
    pass


# TODO: 异常检测
def detect_anomalies(steps, train_losses, val_losses):
    """
    TODO: 检测 loss 曲线中的异常。

    检测以下模式:
    1. NaN: any(np.isnan(losses))
    2. 震荡: rolling_std > 2 * global_std
    3. 平坦: rolling_range < 0.01
    4. 分叉: (val_loss - train_loss) 持续增长

    返回:
        anomalies: list[dict]，每个异常包含 start_step, end_step, type, severity
    """
    pass


# TODO: 生成异常报告
def generate_anomaly_report(anomalies, output_file):
    """
    TODO: 将检测到的异常格式化为可读报告。

    报告内容:
    - 异常数量总结
    - 每个异常的详细信息（类型、步数范围、严重程度）
    - 建议的排查方向
    """
    pass


# TODO: 生成模拟数据用于测试
def generate_test_log(num_steps=1000):
    """
    TODO: 生成包含各种异常的模拟训练日志。

    - 前 200 步: 正常下降
    - 200-300: 剧烈震荡
    - 500-520: NaN
    - 700-800: 平坦
    """
    pass


if __name__ == "__main__":
    # TODO: 生成测试数据并运行异常检测
    pass
