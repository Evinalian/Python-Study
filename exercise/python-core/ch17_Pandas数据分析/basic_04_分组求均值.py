"""
章节：第17章 Pandas 数据分析
题目：按某列分组求均值
类型：基础练习

题目描述：
按 department 分组，计算各组的平均薪资。

数据：
df = pd.DataFrame({
    "name": ["小王", "小李", "小张", "小陈", "小刘", "小赵"],
    "department": ["技术", "销售", "技术", "销售", "技术", "人事"],
    "salary": [15000, 12000, 18000, 11000, 20000, 13000],
})

提示：
- groupby 的核心模式是 split-apply-combine
- 使用 df.groupby("department")["salary"].mean()
- 也可以使用 agg() 同时计算多个统计量
"""

import pandas as pd

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 创建题目中的 DataFrame
# 2. 使用 df.groupby("department")["salary"].mean() 求各组平均薪资
# 3. 使用 .agg(["mean", "min", "max", "count"]) 同时计算多指标
# 4. 使用 transform() 计算每个员工薪资与部门均值的偏差
# 5. 使用 .size() 统计各部门人数
#
# 提示：
# - groupby 模式: split-apply-combine
# - 参考第17章教程中分组聚合部分
#
# 完成后运行: python basic_04_分组求均值.py
