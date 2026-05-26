"""
章节：第20章 开发环境与工具链
题目：用 python-dotenv 读取环境变量
类型：基础练习

题目描述：
在 `learn-venv` 目录下，创建 `.env` 文件写入测试变量，安装 `python-dotenv`，写一个程序读取并打印这些变量。

要求：
1. 创建 `.env` 文件，写入 MY_NAME 和 CLASS_NAME 两个变量
2. 安装 python-dotenv：`pip install python-dotenv`
3. 写一个 Python 程序，用 `load_dotenv()` 加载 `.env`
4. 分别用 `os.environ[key]` 和 `os.getenv(key, default)` 两种方式读取变量
5. 打印读取到的值

前置准备：
需要安装的包：python-dotenv

提示：
- .env 文件绝对不能提交到 git（应该加入 .gitignore）
- 建议同时创建 .env.example 文件作为模板（可提交到 git）
- load_dotenv() 必须放在读取环境变量之前调用
- os.environ[key] 在 key 不存在时抛出 KeyError
- os.getenv(key, default) 在 key 不存在时返回默认值，更安全
"""

import os
import sys

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 在 learn-venv 目录创建 .env 文件，写入 MY_NAME 和 CLASS_NAME
# 2. 用 dotenv.load_dotenv() 加载 .env
# 3. 用 os.environ["MY_NAME"] 读取（处理 KeyError）
# 4. 用 os.getenv("CLASS_NAME", "未知") 读取（带默认值）
# 5. 打印读取结果，对比两种读取方式的区别
#
# 提示：
# - pip install python-dotenv
# - .env 内容示例：MY_NAME=张三
# - .env.example 是模板文件，可以提交到 git
# - 可参考第 20.3 节"环境变量管理"
