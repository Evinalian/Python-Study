"""
章节：第14章 代码规范与工程化
题目：将不 Pythonic 的代码改成 Pythonic
类型：基础练习

题目描述：
将下面的代码改写为 Pythonic 风格：

def process_data(data_list):
    result = []
    for i in range(len(data_list)):
        if data_list[i] != None and data_list[i] > 0:
            result.append(data_list[i] * 2)
    return result

要求：
- 使用列表推导式代替循环 + append
- 用 `is not None` 代替 `!= None`
- 直接遍历元素，不要用 `range(len(...))`
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 将上方函数改写为 Pythonic 风格（列表推导式 + 直接遍历 + is not None）
# 2. 添加类型注解
# 3. 在 main 块中用测试数据验证结果是否正确
#
# 提示：
# - 列表推导式：[x * 2 for x in data_list if x is not None and x > 0]
# - 参考第14章教程中 Pythonic 写法部分
#
# 完成后运行: python basic_02_代码改Pythonic.py
