"""
章节：第6章 函数
题目：带默认参数的问候函数
类型：基础练习

题目描述：
写一个函数 greet(name, greeting="你好", punctuation="！")，让它能灵活定制问候语。

示例输入/输出：
greet("小明") → "你好，小明！"
greet("小红", greeting="早上好") → "早上好，小红！"
greet("小刚", greeting="Hello", punctuation=".") → "Hello，小刚."

提示：
- 使用默认参数值让函数更灵活
- 注意默认参数的顺序：有默认值的参数放在无默认值的参数之后
- f-string 格式化返回问候语
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 greet(name, greeting="你好", punctuation="！") 函数
# 2. 用 f-string 拼接返回问候语
# 3. 测试三种调用方式：默认参数 / 部分覆盖 / 全部覆盖
#
# 提示：
# - 默认参数需放在无默认值参数之后
#
# 完成后，运行 python basic_03_带默认参数的问候.py 测试你的代码。


def greet(name, greeting="你好", punctuation="！"):
    """问候某人"""
    pass  # TODO: 实现此函数
