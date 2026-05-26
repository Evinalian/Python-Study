"""
章节：第23章 AI 大模型应用开发实战
题目：实现流式输出打印
类型：基础练习

题目描述：
将基本对话改为流式输出模式。写一个 `stream_ask(prompt: str)` 函数，接收用户问题，流式打印回复，返回完整文本。

要求：
1. 使用 `stream=True` 开启流式模式
2. 逐 chunk 读取并立即打印（用 `flush=True` 确保即时输出）
3. 同时将每个字符片段收集起来，最后返回完整文本
4. 函数签名：`stream_ask(prompt: str) -> str`

前置准备：
需要安装的包：openai, python-dotenv
需要设置的环境变量：OPENAI_API_KEY

提示：
- 不设置 flush=True 的话，Python 的输出缓冲会导致字符攒到一定量才输出，流式效果就没了
- chunk.choices[0].delta.content 可能为 None（首尾 chunk 不含内容）
- 每个 chunk 通常只包含 1-3 个字符，不要假设一个 chunk 就是一个词
- 用 end="" 避免 print 自动换行
"""

import os
import sys

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义 stream_ask(prompt: str, system_prompt: str | None = None) -> str
# 2. 用 os.environ.get("OPENAI_API_KEY") 读取 API Key（不硬编码）
# 3. 用 stream=True 发起请求，遍历 chunk 逐字打印
# 4. 用 print(content, end="", flush=True) 即时输出
# 5. 收集所有 text 片段，用 "".join() 拼接后返回完整文本
#
# 前置准备：
# - 需要安装的包：openai, python-dotenv
# - 需要设置的环境变量：OPENAI_API_KEY
#
# 提示：
# - delta.content 可能为 None，需要判空
# - flush=True 确保即时输出，不等待缓冲区
# - 不要硬编码 API Key，始终用 os.environ.get("OPENAI_API_KEY") 读取
