"""
章节：第16章 NumPy 数值计算
题目：创建不同类型的数组
类型：基础练习

题目描述：
创建以下数组：
1. 一个从 10 到 49 的一维整数数组（使用 arange）
2. 一个 3x4 的全零浮点数组（使用 zeros）
3. 一个 2x3 的 [0, 1) 均匀分布随机数组（使用 default_rng）

提示：
- np.arange(10, 50) 生成 [10, 11, ..., 49]
- np.zeros((3, 4)) 生成 3 行 4 列的全零数组
- 使用 rng = np.random.default_rng(seed=42) 创建随机数生成器
"""

import numpy as np

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 使用 np.arange(10, 50) 创建一维数组，打印 shape 和 dtype
# 2. 使用 np.zeros((3, 4)) 创建全零数组
# 3. 使用 rng.random((2, 3)) 创建随机数组（先创建 rng = np.random.default_rng(42)）
# 4. 额外练习：np.ones()、np.full()、np.eye()、np.linspace()
#
# 提示：
# - 参考第16章教程中数组创建部分
#
# 完成后运行: python basic_01_创建不同类型数组.py
