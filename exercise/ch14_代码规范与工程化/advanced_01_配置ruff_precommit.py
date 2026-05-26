"""
章节：第14章 代码规范与工程化
题目：为一个 Python 项目配置 ruff + pre-commit
类型：进阶练习

题目描述：
假设你有一个 Python 项目，请完成以下配置：

1. 在项目根目录创建 pyproject.toml，配置 ruff：
   - line-length = 88
   - target-version = "py311"
   - 启用 E, W, F, I, N, UP 规则
   - 格式化使用双引号、空格缩进

2. 创建 .pre-commit-config.yaml，配置两条 hook：
   - 基础 hook：trailing-whitespace、end-of-file-fixer
     （使用 pre-commit/pre-commit-hooks v4.6.0）
   - ruff hook：ruff check --fix 和 ruff-format
     （使用 astral-sh/ruff-pre-commit v0.3.0）

3. 初始化 git 仓库，执行 `pre-commit install` 安装 hook

4. 试着 `git add` 一个文件，然后 `git commit`——观察 pre-commit 是否自动运行检查

提示：
- 需要先安装：pip install ruff pre-commit
- 必须在 git 仓库中才能安装 pre-commit hooks
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 安装工具: pip install ruff pre-commit
# 2. 创建 pyproject.toml，配置 [tool.ruff] 和 [tool.ruff.lint]
# 3. 创建 .pre-commit-config.yaml，配置基础 hook 和 ruff hook
# 4. 初始化 git: git init → pre-commit install → git add → git commit
# 5. 观察提交时自动运行的检查结果
#
# 提示：
# - 参考第14章教程中 ruff 和 pre-commit 配置部分
# - 本文件为指导说明，配置文件需另行创建
#
# 完成后运行: python advanced_01_配置ruff_precommit.py
