"""
章节：第18章 Matplotlib 可视化
题目：画一个简单的折线图
类型：基础练习

题目描述：
用 plt.subplots() 画一条折线图：
- x 轴：0 到 6.28（2*pi），100 个点
- y 轴：sin(x)
- 设置标题为 "正弦波"，x 轴标签为 "角度"，y 轴标签为 "sin(x)"
- 图片保存为 "basic_01_sin_wave.png"

提示：
- 使用面向对象（OO）风格：fig, ax = plt.subplots()
- np.linspace(0, 2 * np.pi, 100) 生成 x 轴数据
- 用 fig.savefig() 保存图片
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 用 np.linspace(0, 2*np.pi, 100) 生成 x 数据，np.sin(x) 生成 y 数据
# 2. 用 OO 风格创建 fig, ax，调用 ax.plot() 画折线图
# 3. 设置标题、轴标签、网格、图例
# 4. 设置 x 轴刻度为 pi 的倍数（更直观）
# 5. 用 fig.savefig() 保存为 PNG 图片
#
# 提示：
# - 使用面向对象风格（OO），不要用 pyplot 函数式风格
# - 用 plt.savefig() / fig.savefig() 保存图片
# - 参考第18章教程中折线图部分
#
# 完成后运行: python basic_01_简单折线图.py
