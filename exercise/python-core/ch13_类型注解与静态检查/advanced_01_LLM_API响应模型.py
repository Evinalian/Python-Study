"""
章节：第13章 类型注解与静态检查
题目：用 Pydantic 设计 LLM API 的请求和响应模型
类型：进阶练习

题目描述：
设计一套 Pydantic 模型来描述一个"情感分析 API"的交互：

请求模型 (AnalysisRequest)：
- `text`: str，最少 1 个字符
- `model`: str，默认 "default"
- `language`: str，默认 "zh"
- `max_tokens`: int，默认 512，必须大于 0

响应模型 (AnalysisResponse)：
- `sentiment`: Literal["positive", "negative", "neutral"]
- `confidence`: float，0-1 之间
- `tokens_used`: int，非负
- `details`: 嵌套模型 SentimentDetails，包含三个 0-1 范围的 float 字段：
  `positive_score`、`negative_score`、`neutral_score`

完成后用一段模拟的 JSON 数据测试 `model_validate_json()`。

提示：
- 嵌套模型就是在一个 Pydantic 模型的字段中使用另一个 Pydantic 模型
- `model_validate_json()` 接受 JSON 字符串，在解析的同时做校验
"""

from pydantic import BaseModel, Field, ValidationError
from typing import Literal

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义 SentimentDetails 模型（3 个 0-1 范围的 float 字段）
# 2. 定义 AnalysisRequest 模型（text/model/language/max_tokens）
# 3. 定义 AnalysisResponse 模型（嵌套 SentimentDetails）
# 4. 用 model_validate_json() 测试合法 JSON + 非法 JSON
#
# 提示：
# - 嵌套模型：字段类型直接写另一个 Pydantic 类名
# - 参考第13章教程中 Pydantic 嵌套模型部分
#
# 完成后运行: python advanced_01_LLM_API响应模型.py
