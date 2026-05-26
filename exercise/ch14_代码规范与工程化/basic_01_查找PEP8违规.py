"""
章节：第14章 代码规范与工程化
题目：找出 PEP 8 违规
类型：基础练习

题目描述：
下面这段代码违反了至少 7 条 PEP 8 规则，请逐一找出并说明：

from math import *
import os, sys

class dataProcessor:
    def __init__(Self, name, Age):
        Self.name=name
        Self.age=Age

    def getInfo(Self):
        return f"{Self.name} is {Self.age} years old"

MAX_RETRY=3

提示：
- 对照 PEP 8 规则逐一检查：import 方式、命名规范、空格、self 写法等
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 阅读题目中的代码，逐一找出至少 7 条 PEP 8 违规
# 2. 用注释或文档字符串列出每条违规及对应的 PEP 8 规则
# 3. 在下方写出修复后的合规代码（可写在 if __name__ == "__main__" 块中）
#
# 提示：
# - 检查通配符导入、多模块同行导入、类名、self 大小写、参数命名、等号空格等
# - 参考 PEP 8 官方文档或第14章教程
#
# 完成后运行: python basic_01_查找PEP8违规.py
