# Python-Study

面向 CS 本科生的 **Python + AI 大模型全栈学习教程**——从基础语法到训练自己的大模型。

## 全局架构（6 模块 60 章）

```
python-core (23章)        Python 基础 → 中级特性 → 数据生态 → AI 入门
pytorch-core (7章)        张量计算 → 神经网络 → GPU 训练 → 完整项目
ai-application (9章)      Prompt 工程 → RAG → 多模态 → 生产部署
ai-agent (8章)            Agent 架构 → 工具系统 → ReAct → 多智能体
ai-finetuning (6章)       数据工程 → LoRA 微调 → 偏好对齐 → 模型部署
ai-training (7章)         DL 基础 → Transformer → 分布式训练 → Scaling Law
```

## 项目结构

```
Python-Study/
├── python-core/            # 📚 23 章 Python 基础教程
├── pytorch-core/           # 🔥 7 章 PyTorch 深度学习
├── ai-application/         # 🤖 9 章 AI 应用开发
├── ai-agent/               # 🧠 8 章 智能体开发
├── ai-finetuning/          # 🎯 6 章 大模型微调
├── ai-training/            # ⚙️ 7 章 大模型训练
├── exercise/               # 🏋️ 配套练习题（TODO 格式）
├── project.md              # 📋 项目全局规划
└── README.md
```

## 教程特点

- **每章独立**：可按顺序学习，也可按需跳转
- **文字驱动**：先用文字建立心智模型，代码佐证（不是 API 速查表）
- **代码可运行**：所有示例代码复制即可运行
- **分层练习**：每章含基础练习 + 进阶练习
- **常见错误**：每章 6-10 个高频陷阱，配错误/正确代码对比
- **CS 本科生定位**：有基本编程能力和计算机理论基础即可学习

## 学习路线

```
第一阶段：Python 核心
  python-core/ 01-14 章  → 掌握 Python 语法和常用库

第二阶段：数据科学与 AI 入门
  python-core/ 15-23 章  → NumPy/Pandas/可视化/LLM SDK
  ai-application/ 01-05  → Prompt 工程/Function Calling/RAG

第三阶段：深度学习基础
  pytorch-core/ 全部7章   → 从张量到完整图像分类项目

第四阶段：AI 应用与 Agent
  ai-application/ 06-09  → 多模态/Web 部署/生产实践
  ai-agent/ 全部8章       → 从单 Agent 到多 Agent 协作

第五阶段：模型微调与训练
  ai-finetuning/ 全部6章  → LoRA/指令微调/偏好对齐
  ai-training/ 全部7章    → Transformer 原理到预训练实战
```

## 使用方法

```bash
git clone https://github.com/Evinalian/Python-Study.git
cd Python-Study

# 按顺序学习教程（推荐 VS Code + Markdown 预览）

# 完成每章对应练习
cd exercise/python-core/ch01_变量与数据类型
python basic_01_变量命名判断.py
```

## 环境要求

- Python 3.10+
- 推荐 IDE：VS Code + Python 扩展

```bash
pip install torch numpy pandas matplotlib seaborn openai python-dotenv httpx aiohttp pydantic chromadb
```

## 许可

MIT License
