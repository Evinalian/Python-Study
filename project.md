# Python+AI 大模型全栈学习教程项目

## 项目仓库

- **本地路径**: `d:\proj\Python-exer`
- **GitHub**: https://github.com/Evinalian/Python-Study
- **Skill**: `C:\Users\9008536\.voyah\skills\python-tutor\SKILL.md`

## 全局架构（6 模块，实际目标约 60 章）

```
python-core (23章)    ✅ 已完成   10,500 行
pytorch-core (7章)    ✅ 已完成    7,568 行
ai-application (9章)  ✅ 已完成   13,552 行
ai-agent (8章)        ⏳ 待启动
ai-finetuning (6章)   ⏳ 待启动
ai-training (7章)     ⏳ 待启动

已完成合计: 39 章, ~31,600 行
```

---

## 完成状态明细

### python-core/ ✅ (23 章, ~10,500 行)

全部完成，涵盖从变量到 LLM SDK 实战的完整 Python 学习路线。

配套练习: `exercise/` 下 23 个目录, 102 个 `.py` 文件 (TODO 格式, 无答案)

### pytorch-core/ ✅ (7 章, 7,568 行)

| 章节 | 行数 |
|------|------|
| 01-张量与自动求导 | 1255 |
| 02-数据处理管线 | 1149 |
| 03-神经网络组件 | 1095 |
| 04-训练引擎 | 1250 |
| 05-模型管理与部署 | 820 |
| 06-GPU与混合精度训练 | 731 |
| 07-实战：图像分类完整项目 | 1268 |

配套练习: `exercise/pytorch-core/` 下 7 个目录, ~34 个 `.py` 文件

### ai-application/ ✅ (9 章, 13,552 行)

| 章节 | 行数 |
|------|------|
| 01-Prompt工程体系 | 1657 |
| 02-Function Calling深入 | 1659 |
| 03-流式处理与SSE | 1191 |
| 04-RAG架构深入 | 1425 |
| 05-向量数据库实战 | 1231 |
| 06-多模态模型应用 | 1835 |
| 07-LangChain与LlamaIndex | 1575 |
| 08-AI应用Web化 | 1305 |
| 09-生产环境最佳实践 | 1674 |

配套练习: `exercise/ai-application/` 各章节已有

### ai-agent/ ⏳ (8 章, 未启动)
### ai-finetuning/ ⏳ (6 章, 未启动)
### ai-training/ ⏳ (7 章, 未启动)

---

## ⚠️ 已知问题：代码驱动 → 文字驱动

pytorch-core 和 ai-application 的文件存在相同问题：**代码占比过高，文字解释太少**——更像 API 速查表而不是教程。

好教程应该：文字建立心智模型（主力）→ 代码佐证（辅助）→ 对比陷阱分析 → 小结。

**明天第一步：对 pytorch-core（7章）和 ai-application（9章）进行"文字驱动"风格重构**——保留全部代码和知识点，大幅增加文字解释、原理说明、使用场景、对比分析。

---

## 文件规范（所有模块通用）

### 教程 .md 文件
- 风格：**文字驱动，代码佐证**。每个概念先用文字讲清 WHY 和 WHAT，再用代码展示 HOW
- 结构：学习目标 → 前置知识 → 递进式讲解 → 基础练习(2-3道) + 进阶练习(1-2道) → 常见错误(6-10个) → 本章小结
- 越详细越好，无行数上限
- 目标读者：CS 本科生，有基本编程能力

### 练习 .py 文件
- 位置：`exercise/<module-name>/chXX_XXX/`
- 格式：顶部 docstring 题目描述 + 底部 TODO 步骤指引
- **无参考答案，无实现代码**

### 项目结构
```
Python-exer/
├── python-core/        # 教程 .md ✅
├── pytorch-core/       # 教程 .md ✅ (需风格重构)
├── ai-application/     # 教程 .md ✅ (需风格重构)
├── ai-agent/           # 教程 .md ⏳
├── ai-finetuning/      # 教程 .md ⏳
├── ai-training/        # 教程 .md ⏳
├── exercise/           # 练习题 .py
├── project.md          # 本文件
└── README.md
```

---

## 待建模块详细规划

### ai-agent/ (8 章)
01-Agent架构原理: 感知→规划→执行→观察循环、Agent分类、框架选型
02-工具系统设计: 工具注册发现、Tool Schema、动态选择、执行沙箱、错误重试
03-ReAct推理模式: Thought-Action-Observation、vs CoT对比、Self-Ask
04-记忆与状态管理: 短期(对话缓冲)、长期(向量持久化)、上下文窗口管理
05-任务规划与分解: Plan-and-Execute、子任务依赖图、动态重规划
06-多智能体协作: 角色分工、通信协议、任务委派、CrewAI实战
07-LangGraph工作流: StateGraph、条件分支、循环节点、Human-in-the-Loop
08-Agent评估与安全: 任务完成率、工具调用准确率、prompt injection防护

### ai-finetuning/ (6 章)
01-微调策略与数据工程: 全量vs高效、指令数据收集清洗质检、数据配比
02-LoRA与QLoRA实战: 低秩适配、量化(GPTQ/AWQ/bitsandbytes)、PEFT库、调参
03-指令微调(SFT): Chat Template、loss masking、过拟合检测、checkpoint择优
04-偏好对齐(RLHF/DPO): 奖励模型、PPO、DPO、偏好数据构建
05-模型评估体系: 自动化评测、LLM-as-Judge、A/B对比
06-模型合并与部署: LoRA合并、GGUF量化(Ollama)、vLLM推理、API服务

### ai-training/ (7 章)
01-深度学习训练基础: 前向/反向传播数学、损失函数、正则化、BatchNorm/LayerNorm
02-Transformer完全拆解: 自注意力数学、多头注意力、RoPE/ALiBi、FFN、Pre-Norm vs Post-Norm
03-分词器: BPE/WordPiece/Unigram、Tokenizers训练、特殊token
04-数据工程: 数据源、质量过滤、去重(MinHash/SimHash)、数据配比
05-分布式训练: DDP、模型并行、流水线并行、ZeRO-1/2/3、FSDP
06-预训练实战: 从头训练小型GPT、训练配置、Loss监控、中断恢复
07-Scaling Law: Chinchilla最优计算、Flash Attention、torch.compile

---

## 明天工作清单

1. **风格重构（最高优先）**: pytorch-core 7章 + ai-application 9章 → 从代码驱动改为文字驱动
2. **新模块**: 启动 ai-agent 8章 + ai-finetuning 6章 + ai-training 7章
3. **配套练习**: 为新模块创建 exercise 练习题
4. 更新 README.md 和 skill 路由表
5. git add + commit + push
