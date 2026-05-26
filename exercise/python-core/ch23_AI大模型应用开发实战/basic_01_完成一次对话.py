"""
章节：第23章 AI 大模型应用开发实战
题目：完成一次 AI 对话
类型：基础练习

题目描述：
用 openai SDK 完成一次对话，要求：
1. 从 `.env` 文件读取 API Key（不硬编码）
2. 向模型发送 "用 Python 写一个冒泡排序函数，包含注释"
3. 打印模型的完整回复内容
4. 打印 token 用量（输入 token、输出 token、总计 token）

前置准备：
需要安装的包：openai, python-dotenv
需要设置的环境变量：OPENAI_API_KEY（在项目根目录创建 .env 文件，写入 OPENAI_API_KEY=你的密钥）

提示：
- 加载顺序：先 load_dotenv()，再读取 os.environ
- model 推荐用 gpt-4o-mini（便宜且快），也可以用 deepseek-chat 等兼容模型
- temperature 控制随机性：0.0=确定性最高，1.0=最随机。代码生成建议 0.3 以下
- response.usage 包含 prompt_tokens（输入）、completion_tokens（输出）、total_tokens（总计）
- 如果不用 OpenAI，改 base_url 即可切换平台（见第 23.8 节）
"""

import os
import sys

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 用 dotenv.load_dotenv() 加载 .env，用 os.environ.get("OPENAI_API_KEY") 读取 API Key
# 2. 用 OpenAI(api_key=api_key) 创建客户端
# 3. 调用 client.chat.completions.create() 发送对话
# 4. 读取 response.choices[0].message.content 打印回复
# 5. 读取 response.usage 打印 token 用量和 finish_reason
#
# 前置准备：
# - 需要安装的包：openai, python-dotenv
# - 需要设置的环境变量：OPENAI_API_KEY
#
# 提示：
# - 不要硬编码 API Key，始终用 os.environ.get("OPENAI_API_KEY") 读取
# - temperature=0.3 适合代码生成
# - 注意处理 API 调用异常和未设置 API Key 的情况
