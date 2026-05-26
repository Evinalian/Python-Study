"""
章节：第13章 类型注解与静态检查
题目：用 Pydantic 定义数据模型
类型：基础练习

题目描述：
用 Pydantic 定义一个 `Student` 模型：
- `name`: str，最少 2 个字符
- `student_id`: str，必须匹配模式 "S\\d{6}"（S 后面 6 位数字）
- `score`: float，在 0-100 之间
- `grade`: Literal["A", "B", "C", "D", "F"]，默认 "C"

创建几个实例测试校验是否生效（合法数据与非法数据都要测试）。

提示：
- 需要使用 `from pydantic import BaseModel, Field`
- 使用 `from typing import Literal`
- 使用 `model_dump()` 方法查看模型数据
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Literal

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 用 Pydantic 定义 Student 模型，给每个字段添加适当的 Field 约束
# 2. 在 main 块中测试合法数据（正确格式+自动类型转换）
# 3. 使用 try/except ValidationError 测试非法数据（短名字、错误学号、超范围分数）
#
# 提示：
# - pattern 参数使用正则表达式 r"^S\\d{6}$"
# - 参考第13章教程中 Pydantic 数据校验部分
#
# 完成后运行: python basic_03_Pydantic数据模型.py
