"""
章节：第17章 Pandas 数据分析
题目：完整的数据分析流程——从清洗到导出
类型：进阶练习

题目描述：
模拟一份销售数据，完成"清洗 → 分组统计 → 排序 → 导出"的完整流程。

步骤：
1. 创建包含缺失值和重复行的模拟数据
2. 清理：去除缺失值、重复行
3. 添加 total_amount = quantity * unit_price 列
4. 按产品和月份分组，计算总销售额
5. 按总销售额降序排列
6. 导出为 CSV

提示：
- 用 pd.date_range 生成日期范围
- 用 dropna() 删除缺失值，drop_duplicates() 删除重复行
- groupby 后可以用 .sum() 聚合
- to_csv 时建议 encoding="utf-8-sig" 方便 Excel 打开
"""

import pandas as pd
import numpy as np
import os

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 创建随机销售数据（100行，含 product/quantity/unit_price），插入缺失值和重复行
# 2. 清洗：dropna() 去缺失 → drop_duplicates() 去重复
# 3. 添加 total_amount 列，提取月份列
# 4. 按 product+month 分组，对 total_amount 求和，按销售额降序排序
# 5. 导出为 CSV（encoding="utf-8-sig"），验证导出文件
#
# 提示：
# - 用 rng.choice() 随机选择日期/产品/价格
# - 完整流程: 创建→检查→清洗→变换→聚合→排序→导出
# - 参考第17章教程中数据分析流程部分
#
# 完成后运行: python advanced_01_清洗分组排序导出.py
