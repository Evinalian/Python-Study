"""
章节：第14章 代码规范与工程化
题目：安装 black 并格式化文件
类型：基础练习

题目描述：
1. 在终端执行 `pip install black`
2. 下面有一段格式混乱的 Python 代码，将它保存为 messy.py，然后运行 `black messy.py` 查看格式化后的效果
3. 运行 `black --check messy.py` 只检查不修改（适用于 CI 环境）

混乱代码（供测试用）：
def calc( a,b,c ):
    result=a+b*c
    if result>100:
       return "big"
    else:
        return "small"

print(calc( 10,20,3 ))

提示：
- black 的默认行宽是 88
- 缩进统一为 4 个空格
- 运算符两侧自动加空格
- 文件末尾自动加空行
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 将题目中的混乱代码保存到同目录下的 messy.py 文件
# 2. 在终端执行: pip install black
# 3. 运行: black messy.py (格式化) 和 black --check messy.py (检查模式)
# 4. 观察格式化前后差异，理解 black 的格式化规则
#
# 提示：
# - 本文件是练习说明，实际操作在终端和 messy.py 中进行
# - 参考第14章教程中 black 格式化部分
#
# 完成后运行: python basic_03_black格式化.py
