"""
章节：第17章 Pandas 数据分析
题目：添加新列并删除旧列
类型：基础练习

题目描述：
基于以下 DataFrame，添加一列 bonus（= salary * 0.1），然后删除原来的 salary 列。

df = pd.DataFrame({
    "name": ["小王", "小李", "小张", "小陈", "小刘"],
    "department": ["技术", "销售", "技术", "销售", "技术"],
    "salary": [15000, 12000, 18000, 11000, 20000],
})

提示：
- 直接赋值添加列：df["bonus"] = df["salary"] * 0.1
- 删除列：df.drop("salary", axis=1)，注意 axis=1 表示删列
- 也可以用 df.drop(columns=["salary"])
"""

import pandas as pd

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 创建题目中的 DataFrame
# 2. 添加 bonus 列：df["bonus"] = df["salary"] * 0.1
# 3. 删除 salary 列：df.drop("salary", axis=1) 或 df.drop(columns=["salary"])
# 4. 练习 assign() 链式操作：一次添加多列并删除旧列
#
# 提示：
# - axis=1 表示列操作，axis=0 表示行操作
# - assign() 返回新 DataFrame，不修改原数据
# - 参考第17章教程中列操作部分
#
# 完成后运行: python basic_03_添加和删除列.py
