"""
章节：第23章 AI 大模型应用开发实战
题目：用 system prompt 设定 AI 的角色
类型：基础练习

题目描述：
写一个程序，用 system prompt 把 AI 设定为"苏格拉底式导师"——不直接给答案，而是用提问引导你思考。
然后问它"我应该怎么学好编程？"，观察它的回复风格。

要求：
1. 编写一个详细的 system prompt，定义苏格拉底式导师的行为准则
2. 将 system prompt 作为第一条消息传入
3. 发送用户问题，观察回复是否遵循了苏格拉底风格
4. 对比：同样的 prompt 去掉 system 角色，看回复风格有什么不同（可选）

前置准备：
需要安装的包：openai, python-dotenv
需要设置的环境变量：OPENAI_API_KEY

提示：
- system prompt 在整个对话中持续生效，是设定 AI"长期人设"的核心手段
- 好的 system prompt 应包含：原则（做什么）、禁止项（不做什么）、格式要求
- 苏格拉底式教学法的核心：通过提问引导学生自己发现答案，而不是直接给答案
- 可以用分隔符（如 1. 2. 3.）让 system prompt 更结构化
"""

import os
import sys

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 编写 SOCRATES_PROMPT 常量，定义苏格拉底式导师的行为准则
#    - 原则：永远不直接给答案，用提问引导学生自己发现
#    - 禁止项：不说客套话、不直接给建议、不一次提多个问题
# 2. 用 os.environ.get("OPENAI_API_KEY") 读取 API Key（不硬编码）
# 3. 分别测试有/无 system prompt，对比回复风格的差异
# 4. 封装 chat_with_role() 函数，传入不同的 system_prompt
#
# 前置准备：
# - 需要安装的包：openai, python-dotenv
# - 需要设置的环境变量：OPENAI_API_KEY
#
# 提示：
# - system prompt 在对话历史第一条（role: "system"）
# - 用分隔符（1. 2. 3.）让 system prompt 结构化
# - 不要硬编码 API Key，始终用 os.environ.get("OPENAI_API_KEY") 读取
