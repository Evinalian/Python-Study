"""
章节：第13章 类型注解与静态检查
题目：用 TypedDict 定义用户信息
类型：基础练习

题目描述：
定义一个 `UserProfile` TypedDict，包含以下字段：
- `name`: str（必填）
- `age`: int（必填）
- `email`: str（必填）
- `phone`: str（可选，Python 3.11+ 用 NotRequired）

然后用它标注一个函数的返回类型，并编写测试代码。

提示：
- Python 3.11+ 使用 `from typing import NotRequired`
- Python 3.10 及以下可使用 `total=False` 类（但所有字段都会变成可选，不太精确）
"""

from typing import TypedDict, NotRequired

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义 UserProfile TypedDict，包含 name/age/email (必填) 和 phone (NotRequired)
# 2. 编写 get_user_profile() 函数，返回 UserProfile 类型
# 3. 在 main 块中测试：创建带 phone 和不带 phone 的实例，验证 NotRequired 行为
#
# 提示：
# - 参考第13章教程中 TypedDict 和 NotRequired 的用法
#
# 完成后运行: python basic_02_TypedDict定义用户信息.py
