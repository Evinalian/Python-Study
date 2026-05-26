"""
章节：第17章 Pandas 数据分析
题目：从字典创建 DataFrame 并查看基本信息
类型：基础练习

题目描述：
用字典创建一个包含 3 个学生信息的 DataFrame（姓名、年龄、成绩），
然后输出 info() 和 describe() 的结果。

提示：
- DataFrame 从字典创建时，key 为列名，value 为该列的值
- info() 显示每列的非空数量、数据类型
- describe() 显示数值列的统计摘要（count/mean/std/min/25%/50%/75%/max）
"""

import pandas as pd

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 用 pd.DataFrame({...}) 创建包含 name/age/score 三列的 DataFrame
# 2. 打印 df.shape、df.columns、df.index
# 3. 调用 df.info() 查看列信息和非空数量
# 4. 调用 df.describe() 查看数值统计摘要
# 5. 计算单列统计：df['score'].mean()、.max()、.min()
#
# 提示：
# - 参考第17章教程中 DataFrame 创建和基本查看部分
#
# 完成后运行: python basic_01_创建DataFrame并查看.py
