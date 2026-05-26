"""
章节：第7章 异常处理与上下文管理器
题目：安全字典取值
类型：基础练习

题目描述：
写一个函数 `safe_get(data, key, default)`，从字典中安全地取出值。
- 如果 key 不存在，返回 default
- 如果 data 本身不是字典（如 None 或 list），也返回 default 并提示错误

示例输入/输出：
    d = {"name": "小明", "age": 20}
    safe_get(d, "name", "未知")     → "小明"
    safe_get(d, "score", 0)         → 0  （key 不存在）
    safe_get(None, "anything", 0)   → 0  （并打印警告）

提示：
需要捕获 KeyError（键不存在）和 TypeError（data 不是字典）两种异常。
"""


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 safe_get(data, key, default=None) 函数
# 2. 捕获 KeyError 和 TypeError 两种异常
# 3. 在 if __name__ == "__main__": 块中编写测试
#
# 提示：参考第7章异常处理示例
#
# 完成后运行: python basic_02_安全字典取值.py


def safe_get(data, key, default=None):
    """安全地从字典取值"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
