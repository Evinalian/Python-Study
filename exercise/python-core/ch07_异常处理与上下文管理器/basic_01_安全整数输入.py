"""
章节：第7章 异常处理与上下文管理器
题目：安全整数输入
类型：基础练习

题目描述：
写一个函数 `safe_input_int(prompt)`，反复提示用户输入直到输入一个合法整数为止。
如果用户输入非数字，提示错误并重新询问。

示例输入/输出：
    请输入年龄: abc
    'abc' 不是一个整数，请重新输入。
    请输入年龄: 25
    你的年龄是: 25

提示：
使用 while True 循环 + try/except ValueError 实现。
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 safe_input_int(prompt) 函数
# 2. 使用 while True + try/except ValueError
# 3. 在 if __name__ == "__main__": 块中编写测试
#
# 提示：参考第7章 try/except 示例
#
# 完成后运行: python basic_01_安全整数输入.py


def safe_input_int(prompt):
    """反复提示用户，直到输入一个合法整数为止"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
