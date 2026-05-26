"""
章节：第13章 类型注解与静态检查
题目：给函数添加类型注解
类型：基础练习

题目描述：
下面是一个没有类型注解的函数，请为它添加完整的类型注解（包括参数和返回值）：

def calculate_total(prices, quantities, tax_rate, discount=0):
    subtotal = 0
    for p, q in zip(prices, quantities):
        subtotal += p * q
    total = subtotal * (1 + tax_rate) - discount
    return round(total, 2)

提示：
- prices 是浮点数列表，quantities 是整数列表
- tax_rate 是浮点数（如 0.13 表示 13%）
- discount 是浮点数，默认值为 0
- 返回值是浮点数
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 将上方题目描述中的函数复制下来
# 2. 为所有参数和返回值添加类型注解（使用 list[float]、list[int]、float 等）
# 3. 在 if __name__ == "__main__": 块中编写测试代码验证
#
# 提示：
# - Python 3.9+ 支持 list[int] 语法，无需从 typing 导入 List
#
# 完成后运行: python basic_01_函数添加类型注解.py
