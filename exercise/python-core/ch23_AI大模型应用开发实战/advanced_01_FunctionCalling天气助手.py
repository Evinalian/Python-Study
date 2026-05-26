"""
章节：第23章 AI 大模型应用开发实战
题目：实现一个带 Function Calling 的天气查询助手
类型：进阶练习

题目描述：
实现一个完整的命令行天气助手，具备以下工具：

1. `get_weather(city: str)` -- 查询城市天气（用 mock 数据）
2. `get_time(city: str)` -- 查询城市当前时间（用 mock 数据）

要求：
- 支持命令行交互（input() 读用户输入，/exit 退出，/clear 清空对话）
- 自动判断是否需要调用工具（使用 tool_choice="auto"）
- 支持多轮对话（维护 messages 历史列表）
- 如果用户问的问题不需要工具，直接对话
- 工具调用过程和结果在控制台可见

前置准备：
需要安装的包：openai, python-dotenv
需要设置的环境变量：OPENAI_API_KEY（在项目根目录 .env 文件中设置）

提示：
- Function Calling 的四步流程：
  1. 定义工具（JSON Schema 描述函数）
  2. 模型决定是否调用工具
  3. 你的代码执行工具函数
  4. 把结果返回给模型，模型生成最终回复
- messages 是一个列表，每次调用需要传入完整对话历史
- assistant 消息中包含 tool_calls 时，需要追加 tool 角色消息来返回结果
- tool_call_id 必须匹配，否则模型无法关联
"""

import json
import os
import sys
from datetime import datetime, timedelta

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 用 os.environ.get("OPENAI_API_KEY") 读取 API Key（不硬编码）
# 2. 定义 TOOLS 列表，用 JSON Schema 描述 get_weather 和 get_time 两个工具
# 3. 实现 get_weather(city) 和 get_time(city)，用 mock 数据返回结果
# 4. 实现 call_ai(messages, client) 处理 Function Calling 循环
# 5. 实现命令行交互主循环（input / /exit / /clear / /help）
# 6. 维护 messages 历史列表，支持多轮对话
#
# 前置准备：
# - 需要安装的包：openai, python-dotenv
# - 需要设置的环境变量：OPENAI_API_KEY
#
# 提示：
# - tool_choice="auto" 让模型自动决定是否调用工具
# - tool 角色消息必须包含 tool_call_id，与 assistant 消息中的 id 匹配
# - 不要硬编码 API Key，始终用 os.environ.get("OPENAI_API_KEY") 读取
# - 可参考第 23.6 节"Function Calling"
