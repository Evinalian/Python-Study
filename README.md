# Python-Study

面向 CS 本科生的 Python 系统学习教程，覆盖从基础语法到 AI 大模型应用开发的完整链路。

## 项目结构

```
Python-Study/
├── python-core/                  # 📚 23 章结构化教程
│   ├── 01-认识Python与变量.md
│   ├── 02-字符串与文本处理.md
│   ├── 03-列表与序列.md
│   ├── 04-字典与集合.md
│   ├── 05-条件判断与循环.md
│   ├── 06-函数.md
│   ├── 07-异常处理与上下文管理器.md
│   ├── 08-文件读写与序列化.md
│   ├── 09-面向对象编程.md
│   ├── 10-模块、包与项目管理.md
│   ├── 11-推导式、迭代器与itertools.md
│   ├── 12-内置函数与functools.md
│   ├── 13-类型注解与静态检查.md
│   ├── 14-代码规范与工程化.md
│   ├── 15-正则表达式.md
│   ├── 16-NumPy数值计算.md
│   ├── 17-Pandas数据分析.md
│   ├── 18-Matplotlib可视化.md
│   ├── 19-Seaborn统计可视化.md
│   ├── 20-Python开发环境与工具链.md
│   ├── 21-HTTP请求与API调用.md
│   ├── 22-异步编程基础.md
│   └── 23-AI大模型应用开发实战.md
│
├── exercise/                    # 🏋️ 102 道配套练习题
│   ├── ch01_变量与数据类型/       (4 道)
│   ├── ch02_字符串与文本处理/     (4 道)
│   ├── ...                      (共 23 个章节目录)
│   └── ch23_AI大模型应用开发实战/ (4 道)
│
└── .gitignore
```

## 章节概览

| 章节 | 内容 | 定位 |
|------|------|------|
| 01-06 | 变量、字符串、序列、字典集合、条件循环、函数 | **Python 核心语法** |
| 07-12 | 异常处理、文件IO/序列化、OOP、模块/包、推导式/itertools、内置函数/functools | **中级特性** |
| 13-19 | 类型注解/Pydantic、工程化、正则、NumPy、Pandas、Matplotlib、Seaborn | **数据科学生态** |
| 20-23 | 环境/工具链、HTTP/API、asyncio、LLM SDK 实战 | **AI 开发必备** |

## 教程特点

- **每章独立**：可按顺序学习，也可按需跳转到任意章节
- **递进式讲解**：每个概念从最简单示例开始，逐步深入
- **代码可运行**：所有示例代码复制即可运行
- **分层练习**：每章含基础练习（巩固语法）+ 进阶练习（综合运用），共 102 道
- **常见错误**：每章列出 4-6 个高频陷阱，配错误/正确代码对比
- **对比 Java/C**：适当对比帮助有 C/Java 基础的学习者快速迁移

## 使用方法

### 1. 阅读教程

```bash
# 按顺序学习，从第 1 章开始
# 也可直接打开感兴趣的章节

# 推荐用 VS Code 打开，支持 Markdown 预览
code python-core/01-认识Python与变量.md
```

### 2. 完成练习

```bash
# 每章教程对应一个练习目录
# 每个 .py 文件顶部有题目描述，底部有 TODO 指引
cd exercise/ch01_变量与数据类型

# 阅读题目，编写代码，运行测试
python basic_01_变量命名判断.py
```

### 3. 学习路线建议

```
零基础入门        → 01-05 章（核心语法）
进阶提升          → 06-12 章（函数/OOP/模块）
数据处理          → 13-19 章（类型注解/数据科学生态）
AI 开发实战       → 20-23 章（环境/HTTP/异步/LLM SDK）
```

## 前置要求

- Python 3.10+
- 基本的编程概念（变量、循环、函数）——不需要精通，了解即可
- 推荐 IDE：VS Code + Python 扩展

## 安装依赖

部分章节需要第三方库：

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux

# 基础依赖
pip install numpy pandas matplotlib seaborn

# 进阶章节依赖
pip install openai python-dotenv httpx aiohttp pydantic mypy ruff
```

## 许可

MIT License
