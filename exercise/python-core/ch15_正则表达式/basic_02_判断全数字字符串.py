"""
章节：第15章 正则表达式
题目：判断字符串是否全是数字
类型：基础练习

题目描述：
写一个函数 `is_all_digits(s)`，判断字符串是否全部由数字组成（使用正则）。

要求：
- 使用 `re.fullmatch()` 确保整个字符串都匹配
- 空字符串返回 False
- 包含非数字字符返回 False

示例输入/输出：
- is_all_digits("12345") → True
- is_all_digits("12a45") → False
- is_all_digits("") → False
"""

import re

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 is_all_digits(s: str) -> bool 函数，使用 re.fullmatch(r"\\d+", s)
# 2. 在 main 块中编写测试用例，覆盖：全数字、含字母、空字符串、含空格/符号
# 3. 验证每个测试用例的期望结果
#
# 提示：
# - re.fullmatch() 返回 Match 对象或 None，判断只需 is not None
# - 参考第15章教程中 fullmatch 部分
#
# 完成后运行: python basic_02_判断全数字字符串.py
