"""
章节：第12章 内置函数与 functools
题目：用 zip 合并两个列表为字典
类型：基础练习

题目描述：
给定两个列表（键列表和值列表），用 `dict(zip(keys, values))` 将它们合并为字典。

示例输入/输出：
    keys = ["name", "age", "city"]
    values = ["小明", 20, "北京"]

    结果: {'name': '小明', 'age': 20, 'city': '北京'}

提示：
- zip(keys, values) 返回 (key, value) 元组的迭代器
- dict() 可以直接从 (key, value) 元组序列创建字典
- 如果两个列表长度不一致，zip 以短的那个为准
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 main() 函数
# 2. 用 zip(keys, values) + dict() 合并为字典
# 3. 打印结果
#
# 提示：参考第12章 zip() 函数示例
#
# 完成后运行: python basic_03_zip合并列表为字典.py


def main():
    """zip 合并两个列表为字典"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    main()
