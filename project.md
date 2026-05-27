# Python+AI 大模型全栈学习教程项目

## 项目仓库

- **本地路径**: `d:\proj\Python-exer`
- **GitHub**: https://github.com/Evinalian/Python-Study
- **Skill**: `C:\Users\9008536\.voyah\skills\python-tutor\SKILL.md`

## 全局架构（6 模块 60 章）✅ 全部完成

```
python-core (23章)    ✅   ~10,500 行    基础→中级→数据生态→AI入门
pytorch-core (7章)    ✅    ~8,000 行    张量→训练→GPU→实战
ai-application (9章)  ✅   ~15,000 行    Prompt→RAG→多模态→生产
ai-agent (8章)        ✅       新建      架构→工具→ReAct→多Agent
ai-finetuning (6章)   ✅       新建      数据→LoRA→SFT→对齐→评估→部署
ai-training (7章)     ✅       新建      DL基础→Transformer→分布式→ScalingLaw
```

---

## 完成状态明细

### python-core/ ✅ (23 章)

从 Python 基础语法到 LLM SDK 实战的完整学习路线。

### pytorch-core/ ✅ (7 章，已文字驱动重构)

| 章节 | 内容 |
|------|------|
| 01-张量与自动求导 | Tensor 创建/运算、GPU 迁移、autograd 自动求导、计算图 |
| 02-数据处理管线 | Dataset/DataLoader 自定义、transform、collate_fn、Sampler |
| 03-神经网络组件 | nn.Module、Linear/Conv/LSTM/Transformer 层、损失函数、参数初始化 |
| 04-训练引擎 | 训练循环、优化器演进(SGD→AdamW)、学习率调度、过拟合应对 |
| 05-模型管理与部署 | state_dict/checkpoint、TorchScript/ONNX 导出、量化/剪枝/蒸馏 |
| 06-GPU与混合精度训练 | CUDA 基础、多 GPU(DDP)、AMP 混合精度(FP16/BF16) |
| 07-实战图像分类器 | 完整 CIFAR-10 项目：数据→模型→训练→评估→部署 |

### ai-application/ ✅ (9 章，已文字驱动重构)

| 章节 | 内容 |
|------|------|
| 01-Prompt工程体系 | Few-shot、CoT 思维链、System Prompt、输出格式控制、A/B 测试 |
| 02-Function Calling深入 | Tool Schema 设计、多工具编排、并行调用、错误恢复 |
| 03-流式处理与SSE | SSE 协议、chunk 解析、流式+FC 结合、FastAPI 流式 API |
| 04-RAG架构深入 | 文档加载、切块策略、稠密/稀疏检索、ReRank、生成与引用 |
| 05-向量数据库实战 | Embedding 模型选型、ChromaDB/Milvus、语义搜索 |
| 06-多模态模型应用 | 视觉理解、文生图(DALL-E/SD)、语音识别/合成(Whisper/TTS) |
| 07-LangChain与LlamaIndex | LCEL 管道、RAG Chain、LlamaIndex 索引/查询引擎、框架选型 |
| 08-AI应用Web化 | FastAPI/Gradio/Streamlit、Docker 容器化、部署方案 |
| 09-生产环境最佳实践 | Token 成本追踪、缓存策略、限流降级、安全防护(Prompt Injection) |

### ai-agent/ ✅ (8 章)

| 章节 | 内容 |
|------|------|
| 01-Agent架构原理 | Agent 本质、感知-规划-执行-观察循环、分类体系、框架选型 |
| 02-工具系统设计 | Tool Schema 设计、ToolRegistry、多工具/并行/安全沙箱 |
| 03-ReAct推理模式 | Thought-Action-Observation、ReAct vs CoT、Self-Reflection |
| 04-记忆与状态管理 | 工作/短期/长期记忆、上下文窗口管理、向量记忆 |
| 05-任务规划与分解 | Plan-and-Execute、子任务依赖图、动态重规划 |
| 06-多智能体协作 | 主从/协商/流水线模式、通信协议、CrewAI 实战 |
| 07-LangGraph工作流引擎 | StateGraph、条件分支、循环、Human-in-the-Loop |
| 08-Agent评估与安全 | 任务完成率、工具调用准确率、Prompt Injection 防护 |

### ai-finetuning/ ✅ (6 章)

| 章节 | 内容 |
|------|------|
| 01-微调策略与数据工程 | 全量vs高效微调、基座模型选型、指令数据收集/清洗/配比 |
| 02-LoRA与QLoRA实战 | 低秩分解原理、NF4 量化、PEFT 库、显存消耗对比 |
| 03-指令微调(SFT) | Chat Template、Loss Masking、过拟合检测、Checkpoint 择优 |
| 04-偏好对齐(RLHF/DPO) | RLHF 三阶段、DPO 简化原理与数学、TRL 实战 |
| 05-模型评估体系 | Benchmark 评测、LLM-as-Judge、A/B 统计检验、评估陷阱 |
| 06-模型合并与部署 | LoRA 合并、GGUF 量化(Ollama)、vLLM 推理、成本分析 |

### ai-training/ ✅ (7 章)

| 章节 | 内容 |
|------|------|
| 01-深度学习训练基础 | 前向/反向传播数学、损失函数、正则化、NumPy 手写 MLP |
| 02-Transformer完全拆解 | Q/K/V Attention、RoPE/ALiBi、SwiGLU、Micro-Transformer 实现 |
| 03-分词器原理与训练 | BPE/WordPiece/Unigram/SentencePiece、中文 BPE 训练 |
| 04-数据工程 | 质量过滤、MinHash/SimHash 去重、数据配比/打包、DVC |
| 05-分布式训练 | DDP/TP/PP、ZeRO-1/2/3(DeepSpeed)、FSDP、3D 并行 |
| 06-预训练实战 | CLM vs MLM、MiniGPT 训练(34M)、loss 曲线诊断 |
| 07-Scaling Law与效率优化 | Kaplan/Chinchilla Law、FlashAttention、KV Cache、torch.compile |

---

## 文件规范

### 教程 .md 文件
- 风格：**文字驱动，代码佐证**（已对 pytorch-core 和 ai-application 完成风格重构）
- 结构：学习目标 → 前置知识 → 递进式讲解 → 基础/进阶练习 → 常见错误 → 本章小结

### 练习 .py 文件
- 位置：`exercise/<module>/chXX_XXX/`
- 格式：docstring 题目描述 + TODO 指引，无答案

### 项目结构
```
Python-exer/
├── python-core/       # 教程 .md ✅
├── pytorch-core/      # 教程 .md ✅
├── ai-application/    # 教程 .md ✅
├── ai-agent/          # 教程 .md ✅
├── ai-finetuning/     # 教程 .md ✅
├── ai-training/       # 教程 .md ✅
├── exercise/          # 练习题 .py
├── project.md         # 本文件
└── README.md
```
