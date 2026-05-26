"""
章节：第20章 开发环境与工具链
题目：从零初始化一个完整的 Python 项目
类型：进阶练习

题目描述：
创建一个完整的 Python 项目 `my-weather-app`，包含以下内容：

1. 合理的目录结构
2. 虚拟环境 `.venv/`
3. `pyproject.toml`（含项目元数据、依赖、ruff 配置）
4. `.env` 和 `.env.example`
5. `.gitignore`
6. 一个 `main.py` 从 `.env` 读取 API Key 并打印"配置加载成功"

目标目录结构：
```
my-weather-app/
├── .venv/                  # 虚拟环境（不提交到 git）
├── .env                    # 敏感信息（不提交到 git）
├── .env.example            # 模板（提交到 git）
├── .gitignore              # git 忽略规则
├── pyproject.toml          # 项目配置
├── main.py                 # 主程序入口
└── README.md               # 项目说明
```

前置准备：
需要安装的包：python-dotenv
需要设置的环境变量：OPENWEATHER_API_KEY（在 .env 中设置）

提示：
- 本练习包含命令行操作和文件编写两部分
- 命令行步骤在注释中详细列出
- 代码部分展示如何编写 main.py 验证项目配置
- 如果 API Key 还是默认值，程序会提示用户去配置
"""

# ========== 命令行操作步骤 ==========

"""
=== 第 1 步：创建项目目录和虚拟环境 ===

mkdir my-weather-app
cd my-weather-app
python -m venv .venv

# Windows 激活
.venv\\Scripts\\Activate.ps1

# Linux/Mac 激活
# source .venv/bin/activate

# 安装依赖
pip install requests python-dotenv


=== 第 2 步：创建 pyproject.toml ===

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "my-weather-app"
version = "0.1.0"
description = "一个查询天气的命令行小工具"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "requests>=2.28",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
]

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.pytest.ini_options]
testpaths = ["tests"]


=== 第 3 步：创建 .env 和 .env.example ===

# .env
echo OPENWEATHER_API_KEY=abc123yourkeyhere > .env

# .env.example
echo OPENWEATHER_API_KEY=请在这里填入你的 API Key > .env.example


=== 第 4 步：创建 .gitignore ===

# 虚拟环境
.venv/

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# 敏感信息
.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# 操作系统
.DS_Store
Thumbs.db


=== 第 5 步：运行 main.py ===
python main.py
"""

import os

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 按以上步骤创建 my-weather-app 项目目录和虚拟环境
# 2. 创建 pyproject.toml、.env、.env.example、.gitignore 文件
# 3. 编写 main.py，用 dotenv 加载 .env 并读取 OPENWEATHER_API_KEY
# 4. 检查虚拟环境、依赖安装、API Key 是否配置正确
#
# 前置准备：
# - 需要安装的包：requests, python-dotenv
# - 需要设置的环境变量：OPENWEATHER_API_KEY
#
# 提示：
# - 检查虚拟环境：sys.prefix != sys.base_prefix
# - .env 不要提交到 git（已在 .gitignore 中排除）
# - 可参考第 20 章综合示例
