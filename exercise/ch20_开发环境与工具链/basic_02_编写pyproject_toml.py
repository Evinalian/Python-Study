"""
章节：第20章 开发环境与工具链
题目：写一个最简 pyproject.toml
类型：基础练习

题目描述：
在你的 `learn-venv` 目录下创建一个 `pyproject.toml`，包含基本的项目元数据和 `requests` 依赖。

pyproject.toml 是现代 Python 项目的统一配置文件，相当于 Java 项目里 pom.xml 的地位。

要求包含以下字段：
- project.name: 项目名（用 - 分隔）
- project.version: 版本号（语义化版本）
- project.description: 一句话描述
- project.requires-python: 最低 Python 版本
- project.dependencies: 运行时依赖列表

前置准备：
需要了解 TOML 的基本语法（类似 INI 文件，但更规范）。

提示：
- pyproject.toml 放在项目根目录（learn-venv/ 目录下）
- 依赖版本建议使用宽松范围（如 >=2.28），而不是精确版本（==2.31.0）
- [project] 是必须的部分
- [project.optional-dependencies] 用于可选依赖（如 dev 工具）
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 在 learn-venv 目录下创建 pyproject.toml 文件
# 2. 填写 [project] 部分：name、version、description、requires-python、dependencies
# 3. （可选）添加 [project.optional-dependencies] 用于 dev 工具
# 4. （可选）添加 [tool.ruff] 和 [tool.pytest.ini_options] 配置
#
# 提示：
# - TOML 语法类似 INI，使用 [section] 和 key = value
# - 依赖使用宽松版本范围：dependencies = ["requests>=2.28"]
# - 可选依赖组如 dev = ["pytest>=8.0", "ruff>=0.5"]
# - 可参考第 20.2 节"pyproject.toml 详解"
