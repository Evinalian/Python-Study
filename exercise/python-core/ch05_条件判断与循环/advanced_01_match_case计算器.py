"""
章节：第5章 条件判断与循环
题目：用 match/case 实现简单计算器
类型：进阶练习

题目描述：
写一个函数 calculate(a, op, b)，用 match/case 支持加减乘除四则运算。要求：
- op 为 "+", "-", "*", "/" 之一
- 除法时处理除零错误
- 不支持的运算符返回错误提示

示例输入/输出：
calculate(10, "+", 5) → 15
calculate(10, "-", 5) → 5
calculate(10, "*", 5) → 50
calculate(10, "/", 3) → 3.333...
calculate(10, "/", 0) → "错误：除数不能为零"
calculate(10, "^", 5) → "错误：不支持的运算符 '^'"

提示：
- 使用 match/case 模式匹配（Python 3.10+）
- 用 _ 通配符匹配不支持的运算符
- match/case 不需要 break，不会 fall-through
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 calculate(a, op, b) 函数
# 2. 使用 match/case 匹配 "+", "-", "*", "/"
# 3. 除法时检查 b == 0，用 _ 匹配不支持的运算符
#
# 提示：
# - match/case 无需 break，不会贯穿执行
#
# 完成后，运行 python advanced_01_match_case计算器.py 测试你的代码。


def calculate(a, op, b):
    """
    简单计算器，支持加减乘除。

    参数:
        a, b: 两个操作数
        op: 运算符（"+", "-", "*", "/"）

    返回:
        计算结果或错误信息
    """
    pass  # TODO: 实现此函数
