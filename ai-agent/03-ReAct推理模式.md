# 第3章 ReAct 推理模式

## 学习目标

完成本章后，你将能够：

1. 深刻理解 ReAct（Reasoning + Acting）的交替循环机制
2. 掌握 Thought-Action-Observation 循环中每个步骤的详细过程
3. 对比 ReAct 与 Chain-of-Thought（CoT）的区别和适用场景
4. 理解 Self-Reflection 和 Self-Ask 等高级推理模式
5. 从零实现一个完整的 ReAct Agent
6. 分析 ReAct 循环中的常见问题和优化策略

## 前置知识

- 已完成第 1 章 Agent 架构原理，理解 Agent 的核心循环
- 已完成第 2 章工具系统设计，理解工具 Schema 和执行机制
- 有 LLM API 调用的实际经验
- 理解基本的提示工程概念

---

## 第一部分：ReAct 是什么

### 1.1 问题的起源

在 ReAct 被提出之前，LLM 的使用方式主要有两种流派：

**流派一：纯推理（Reasoning-only）**
让 LLM 一步步推理，最终给出答案。典型的方法是 Chain-of-Thought（CoT）——给 LLM 一个"让我们一步步思考"的 prompt，LLM 就会输出推理过程，然后给出最终答案。这种方法的优点是 LLM 能展现出令人印象深刻的推理能力，缺点是它的推理完全建立在训练数据的"记忆"之上——如果它不知道某个事实，或者它的训练数据是错的，推理过程再漂亮也是错的。更糟糕的是，LLM 可能会"一本正经地胡说八道"——推理过程看起来逻辑严密，但每一步的"事实前提"都是编造的。

**流派二：纯行动（Acting-only）**
让 LLM 在环境中一步步行动，通过试错来完成任务。典型的例子是早期的 WebGPT 类项目——LLM 可以搜索网页、点击链接、滚动页面。这种方法的优点是 LLM 能获取真实的外部信息，缺点是缺少系统性的推理——LLM 可能会漫无目的地搜索、点开不相关的链接、在大量信息中迷失方向。

**ReAct 的核心洞察**：把推理和行动结合起来。推理让行动更有目的性（不是盲目搜索，而是先想清楚"我需要什么信息"再搜），行动让推理建立在事实之上（不是凭空推理，而是基于工具返回的真实数据）。

### 1.2 ReAct 的形式化定义

ReAct = **Rea**soning + A**c**ting。它是一种交替进行推理步骤和行动步骤的 Agent 模式。形式上，ReAct 循环可以表示为：

```
Thought → Action → Observation → Thought → Action → Observation → ... → Final Answer
```

每一步的含义：

- **Thought（思考）**：LLM 分析当前状态，思考"现在我在哪里？我还需要什么信息？下一步最佳行动是什么？"Thought 是内部的，不产生外部效果。它是一段自然语言文字，记录 LLM 的推理过程。

- **Action（行动）**：基于上一步的 Thought，LLM 执行一个具体的操作。在 LLM Agent 语境中，Action 通常表现为一个工具调用（function call）。Action 是外部的，会产生实际效果——查询数据库、调用 API、执行搜索等。

- **Observation（观察）**：Action 的结果反馈回来。Observation 是外部的信息输入——工具返回的 JSON、搜索结果、计算结果等。

然后是下一轮的 Thought，LLM 基于新的 Observation 重新思考，决定下一步。循环持续直到 LLM 认为任务完成，输出 Final Answer。

### 1.3 为什么这个循环模式如此有效

ReAct 之所以比纯推理或纯行动都更有效，可以从信息论的角度来理解：

**纯推理的问题**：LLM 的推理质量完全受限于其训练数据中包含的信息。当你问"2025 年诺贝尔物理学奖颁给了谁"，LLM 的训练数据截止日期如果早于 2025 年 10 月，它的推理就完全建立在"不知道"的基础上，只能胡猜。

**纯行动的问题**：如果没有推理引导，"行动"会变得低效甚至盲目。想象一个人被蒙上眼睛，手里只有一根探路棍。他可以不停地用棍子探路（行动），但没有方向感（推理），他可能在原地打转。

**ReAct 的协同效应**：推理为行动提供了方向（"我应该搜什么"），行动为推理提供了地基（"搜到了这个，那我接下来应该..."）。两者互相增强：
- 推理使得每一步行动都是有目的的
- 行动使得每一步推理都建立在真实数据之上
- 观察结果可能引发新的推理（意外发现、方向调整）
- 推理结果指导行动的终止时机（"信息够了，可以给答案了"）

---

## 第二部分：ReAct 循环详解

### 2.1 第一轮 Thought：启动分析

在接收到用户问题后的第一轮，LLM 还没有任何外部信息（除了系统 prompt 和用户问题本身）。第一轮 Thought 的重点是：

**理解问题**：用户的真实意图是什么？这个问题的本质是什么？是信息检索问题？是计算问题？是比较分析问题？还是闲聊？

**评估自身知识边界**：关于这个问题，我（LLM）自己知道什么？我不确定什么？我需要外部信息来确认什么？

**制定获取信息的策略**：我应该先做什么？搜索？计算？查询数据库？还是分解成子问题？

让我们用一个具体例子来展示每一轮 Thought：

**用户问题**："请帮我查一下特斯拉和比亚迪 2024 年的全球交付量，并比较两者的增长趋势。"

**第一轮 Thought（LLM 的内部推理）**：
> "用户想要特斯拉和比亚迪 2024 年的全球交付量数据，并进行比较。这两个数据都是实时/近期的信息，我无法从训练数据中准确获取。我需要先搜索这两个数据。我可以在一次搜索中查两个公司的数据，但分开搜索可能获得更精确的结果。让我先搜索特斯拉 2024 年交付量。"

### 2.2 第一轮 Action：首次工具调用

基于第一轮的 Thought，LLM 决定执行第一个 Action。这个 Action 通常是一个工具调用。

**Action**：调用 `search_web(query="特斯拉 2024年 全球交付量")`

在这个阶段，Agent 系统执行实际的工具调用——调用搜索引擎 API（或模拟搜索），等待结果返回。

### 2.3 第一轮 Observation：获取反馈

工具返回结果后，LLM 获得了一个 Observation。

**Observation**：
```
"特斯拉 2024 年全球交付量为 178.9 万辆，同比下降约 1%。这是特斯拉自 2011 年上市以来首次出现年度销量下滑。其中 Model 3 和 Model Y 合计交付约 170 万辆。"
```

Observation 是一个新的事实信息注入。这个信息可能：
- 直接回答了用户的部分问题（特斯拉的交付量数据拿到了）
- 包含了意料之外的信息（首次年度下滑——这个值得在比较中提到）
- 暗示了下一步行动（现在需要比亚迪的数据了）

### 2.4 第二轮 Thought：分析观察，调整计划

拿到特斯拉的数据后，LLM 进入第二轮 Thought。

**第二轮 Thought**：
> "我拿到了特斯拉 2024 年交付量：178.9 万辆，同比下降 1%。这个数据包含了意外的信息——是首次下滑。现在我需要比亚迪的数据来进行比较。搜索比亚迪 2024 年交付量。"

**第二轮 Action**：调用 `search_web(query="比亚迪 2024年 全球交付量")`

**第二轮 Observation**：
```
"比亚迪 2024 年全球新能源车销量达到 427 万辆，同比增长 41.3%，连续第三年位居全球新能源车销量第一。其中纯电动车型约 176 万辆，插电混动车型约 251 万辆。"
```

### 2.5 第三轮 Thought：信息汇总与对比分析

拿到双方的数据后，LLM 拥有了进行比较分析的所有必要信息。第三轮 Thought 负责整合和比较：

**第三轮 Thought**：
> "现在我有两个公司的数据了：
> - 特斯拉：178.9 万辆，同比下降约 1%
> - 比亚迪：427 万辆，同比增长 41.3%
>
> 比亚迪的交付量是特斯拉的约 2.4 倍。但需要注意，比亚迪的数据包含纯电动和插电混动两种类型，而特斯拉只有纯电动。如果只看纯电动，比亚迪约 176 万辆，与特斯拉的 178.9 万辆非常接近。
>
> 我已经有了回答用户问题所需的所有信息。不需要进一步搜索了。我现在可以直接回答用户的问题，给出数据、比较分析、以及对增长趋势的解读。"

### 2.6 Final Answer：综合输出

LLM 判断任务已完成，输出 Final Answer——一段综合了所有信息的自然语言回答：

> "根据搜索结果，以下是特斯拉和比亚迪 2024 年全球交付量的对比：
>
> **特斯拉**：2024 年全球交付 178.9 万辆，同比下降约 1%。这是特斯拉自上市以来首次出现年度销量下滑，主要受到全球电动车市场竞争加剧和部分地区补贴退坡的影响。
>
> **比亚迪**：2024 年全球新能源车销量 427 万辆，同比增长 41.3%，连续第三年位居全球第一。其中纯电动车型约 176 万辆，插电混动约 251 万辆。
>
> **对比分析**：
> 1. 总量对比：比亚迪的总销量是特斯拉的约 2.4 倍。但比亚迪的销量包含插电混动车型，而特斯拉仅销售纯电动车。
> 2. 纯电对比：如果只看纯电动车型，比亚迪（约 176 万辆）与特斯拉（约 178.9 万辆）几乎持平，竞争非常接近。
> 3. 增长趋势：比亚迪保持高速增长（+41.3%），而特斯拉出现了历史性的销量下滑（-1%），反映了两个公司在产品策略和市场布局上的不同路径。
> 4. 比亚迪的插电混动产品线是其增长的主要驱动力，这一定位使其在充电基础设施尚不完善的市场上获得了更大的用户群。"

---

## 第三部分：ReAct vs CoT 深度对比

### 3.1 Chain-of-Thought（CoT）的本质

CoT 是一种通过提示 LLM 逐步推理来提高答案质量的技巧。它的典型 prompt 是：

```
"Q: 一个农场有 15 只鸡和 8 头牛，一共有多少条腿？
Let's think step by step.
鸡有 2 条腿，15 只鸡 = 30 条腿。
牛有 4 条腿，8 头牛 = 32 条腿。
总计 = 30 + 32 = 62 条腿。
答案：62 条腿。"
```

CoT 的核心价值在于：
- 将复杂问题分解为小步骤，降低每一步的推理难度
- 中间步骤可检查（人类可以追踪 LLM 的推理链）
- 防止 LLM 跳过推理直接"猜"答案

### 3.2 CoT 的致命缺陷：没有外部信息输入

CoT 的推理过程完全封闭在 LLM 的"大脑"内部。每一步推理的依据都是 LLM 已经知道（或自认为知道）的信息。当推理的前提是"错误的事实"时，整个推理链就成了建立在沙滩上的城堡。

例如："Q: 苹果公司 2024 年 Q4 的营收是多少亿美元？同比增长多少？Let's think step by step."

LLM 可能会这样"推理"：
> "根据我了解的信息，苹果 2023 年 Q4 营收约为 895 亿美元。考虑到 iPhone 15 系列的发布和市场表现，以及服务业务的持续增长，苹果 2024 年 Q4 营收可能在 900-950 亿美元之间。同比增长可能在 3-6% 左右..."

这个过程看起来是在"推理"，但实际上每一步都是 LLM 在"编造"。它不知道 2023 年 Q4 的真实营收是不是 895 亿，也不知道 2024 年 Q4 的数字是多少。所谓的"推理"不过是根据训练数据中的统计规律，生成一个"看起来像推理"的文字序列。

### 3.3 ReAct 如何解决这个问题

ReAct 的解法是：**当你需要事实时，不要推理——去获取事实。**

同样的问题，ReAct 的处理方式：
> Thought: "我需要苹果 2024 年 Q4 的营收数据。这个数据是我不知道的实时信息。"
> Action: search_web("苹果 2024年Q4 营收 财报")
> Observation: "苹果 2024 年 Q4 营收为 949 亿美元，同比增长 6%..."
> Thought: "我拿到了准确数据：949 亿美元，同比增长 6%。... Final Answer。"

区别一目了然。CoT 在"猜"，ReAct 在"查"。当问题涉及真实世界的事实信息时，这是本质性的差异。

### 3.4 各自的适用场景

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| 数学推理题 | CoT | 不需要外部信息，纯逻辑推理 |
| 逻辑谜题 | CoT | 同上，纯推理问题 |
| 代码理解/调试 | CoT（或 CoT + 代码执行工具） | 可以纯推理，也可以加工具 |
| 实时信息查询 | ReAct | 必须依赖外部数据 |
| 数据分析（未知数据） | ReAct | 数据需要从外部获取 |
| 多步信息检索 | ReAct | 需要多次搜索和整合 |
| 创意写作 | 直接生成 | 不需要推理也不需要行动 |
| 综合调研 | ReAct（可能是多轮长循环） | 搜索+分析+搜索的迭代 |

**一个重要的事实**：ReAct 和 CoT 不是互斥的。你可以在 ReAct 的 Thought 阶段使用 CoT 式的逐步推理。实际上，一个好的 ReAct Agent 的 Thought 就应该包含 CoT 式的推理过程——先分析状态，再逐步推理出下一步行动。

---

## 第四部分：Self-Reflection 自我反思

### 4.1 什么是 Self-Reflection

Self-Reflection 是一种让 LLM 评估自身输出质量并自我修正的技术。它的核心思想是：**在给出最终答案之前，让 LLM 自己检查一遍自己的输出，看看有没有需要改进的地方。**

人类的写作过程通常包含"草稿→检查→修改"的循环。Self-Reflection 就是把这个循环引入到 LLM 的生成过程中。

### 4.2 Self-Reflection 在 Agent 中的应用

在 ReAct 循环中，Self-Reflection 可以集成在多个时机：

**时机一：每轮 Thought 之后**
在决定下一步 Action 之前，先反思："我上一步的思路对吗？有没有遗漏什么？有没有更好的方向？"

**时机二：获取 Observation 之后**
拿到工具返回的结果后，反思："这个结果可信吗？完整吗？我需要用不同的关键词再搜一次吗？"

**时机三：准备输出 Final Answer 之前**
在给用户最终回答之前，反思："我的回答完整吗？数据准确吗？逻辑通顺吗？有没有遗漏用户问题中的某个部分？"

### 4.3 实现 Self-Reflection 的几种方式

**方式一：反思 Prompt**
最直接的方式——在 prompt 中加入反思指令。例如在 system prompt 中加入：
```
"在每次做出决策前，请先反思：
1. 我当前拥有哪些信息？
2. 我是否遗漏了什么重要信息？
3. 是否有其他角度或方法来处理这个问题？
4. 我的推理过程中有没有逻辑漏洞？"
```

**方式二：独立的反思步骤**
在 ReAct 循环中插入一个专门的 Reflection 步骤。让 LLM 输出一段反思文字，但这段反思不被视为一个"Action"，而是用于调整后续的 Thought。这种方式更结构化，反思的质量通常更高。

**方式三：Reflexion 模式**
Reflexion 是 Self-Reflection 的进阶版本——LLM 不仅反思当前步骤，还将反思的结果存储为"长期记忆"，以便在未来的类似任务中避免重复犯错。这涉及记忆管理，将在第 4 章深入讨论。

### 4.4 Self-Reflection 的局限

**自我检查的盲区**：LLM 检查自己的输出时，和生成输出时使用的是同一个"大脑"。它可能和生成时犯完全一样的错误，因为它缺少独立的"审阅视角"。

**过度反思**：某些 LLM 在加入反思指令后会变得过于谨慎——反复确认已经正确的信息，浪费 token 和时间。需要在 prompt 中明确"反思一次就够，不要反复检查"。

**反思质量的波动**：LLM 有时会"敷衍式"反思——生成一段看起来像反思但实际没有价值的文字。例如"我仔细检查了，没有问题"，而实际上确实有问题。需要在 prompt 中给出反思的具体维度和检查清单。

---

## 第五部分：Self-Ask 自问自答

### 5.1 什么是 Self-Ask

Self-Ask 是 Google Research 在 2022 年提出的另一种推理模式。它的核心思想是：**LLM 在面对复杂问题时，主动将问题分解为一系列子问题，然后逐一回答每个子问题。**

Self-Ask 的典型流程：
```
用户问: "阿尔伯特·爱因斯坦是什么时候出生的？他出生的那年有什么重大历史事件？"

LLM Self-Ask:
"我需要回答两个子问题："
"子问题1: 阿尔伯特·爱因斯坦是什么时候出生的？"
"子问题2: 爱因斯坦出生的那年（1879年）有什么重大历史事件？"

[逐一回答子问题]

"综合回答: ..."
```

### 5.2 Self-Ask vs ReAct

Self-Ask 和 ReAct 都涉及"分解和逐步解决"，但有一个关键区别：

**Self-Ask 中的子问题由 LLM 自己回答**（通过自身的知识），不需要调用外部工具。

**ReAct 中的 Action 是调用外部工具**，从外部世界获取 LLM 自身不具备的信息。

在实践中，两者常常结合使用——Self-Ask 用于分解问题，ReAct 用于获取子问题所需的真实数据。这种组合模式被称为"Self-Ask + ReAct"或"Plan-and-Solve"。

### 5.3 Self-Ask 的实现

```python
"""
Self-Ask 模式实现 —— LLM 自主分解问题并逐一解决
"""

SELF_ASK_SYSTEM_PROMPT = """
你是一个善于分析复杂问题的助手。当面对复杂问题时，请遵循以下步骤：

1. 首先，将问题分解为几个独立的子问题
2. 逐一回答每个子问题（如果需要搜索信息，使用搜索工具）
3. 最后，综合所有子问题的答案，给出完整的最终回答

在分解问题时，确保每个子问题都是明确且可独立回答的。
不要提出无法回答或模糊不清的子问题。

输出格式：
### 问题分解
1. [子问题1]
2. [子问题2]
...

### 逐一回答
**子问题1**: [答案]
**子问题2**: [答案]
...

### 综合回答
[结合所有子问题答案的完整回答]
"""


def self_ask_agent(question: str, client, registry) -> str:
    """
    实现 Self-Ask + ReAct 的组合 Agent。

    流程:
    1. LLM 将用户问题分解为子问题
    2. 对每个子问题，使用 ReAct 循环来获取所需的真实信息
    3. 综合所有子问题的答案，给出最终回答
    """
    messages = [
        {"role": "system", "content": SELF_ASK_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    # 第一步：让 LLM 分解问题
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content
```

---

## 第六部分：从零实现一个完整的 ReAct Agent

现在我们来实现一个完整的 ReAct Agent。和第 1 章的最简 Agent 不同，这里我们重点展示 ReAct 的 Thought-Action-Observation 循环，让每一步的推理过程都可见。

```python
"""
完整的 ReAct Agent 实现

本实现展示了 ReAct (Reasoning + Acting) 的完整循环过程。
每个循环都清晰地展示 Thought → Action → Observation 三个步骤。
与第 1 章的最简 Agent 不同，这里我们没有依赖 OpenAI 的 native function calling，
而是使用文本格式的 ReAct prompt，让 LLM 以特定格式输出 Thought 和 Action。
这使得推理过程完全透明可见。
"""

import json
import os
import re
from openai import OpenAI

# ============================================================
# 初始化 LLM 客户端
# ============================================================
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "your-api-key"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ============================================================
# ReAct System Prompt
# ============================================================
# 这是 ReAct Agent 最核心的部分 —— system prompt。
# 它告诉 LLM 按照 Thought → Action → Observation 的循环来思考和行动。
#
# 关键设计要点:
# 1. Thought 和 Action 必须成对出现
# 2. Final Answer 标志着循环结束
# 3. 给出明确的格式示例，LLM 会模仿这个格式
# 4. 工具列表嵌入在 prompt 中，清晰描述每个工具的用途

REACT_SYSTEM_PROMPT = """
你是一个使用 ReAct (Reasoning + Acting) 模式工作的智能助手。

## 工作方式
你的思考过程遵循以下循环:
1. Thought: 分析当前情况，思考下一步应该做什么
2. Action: 执行一个具体的操作（调用工具）
3. Observation: 观察操作的结果
4. 重复以上步骤，直到获得足够的信息来回答问题

## 可用工具
你可以使用以下工具来获取信息和执行操作:

- search(query: str): 在互联网上搜索信息。参数 query 是搜索关键词。
- calculate(expression: str): 执行数学计算。参数 expression 是数学表达式，如 "22 * 9 / 5 + 32"。
- get_time(): 获取当前日期和时间。

## 输出格式（严格遵守）
每一轮你必须按照以下格式输出:

Thought: [你的思考过程，分析当前状态，决定下一步做什么]
Action: [工具名称]([参数])

当你获得足够信息后，输出:

Thought: [最终思考，确认信息足够]
Final Answer: [你的最终回答]

## 格式示例

用户问题: "北京今天天气怎么样？"
你的输出:
Thought: 用户想知道北京今天的天气。这是实时信息，我需要搜索。让我搜索北京天气。
Action: search("北京 今天 天气")

(系统会返回 Observation，然后你继续)
Thought: 我获取到了北京今天的天气数据: 晴，22°C。信息已经足够回答用户的问题。
Final Answer: 北京今天天气晴朗，气温 22°C。

## 重要规则
1. 每轮必须同时包含 Thought 和 Action（或 Final Answer）
2. 不要跳过 Thought 直接输出 Action
3. Thought 中要解释清楚你为什么要采取这个行动
4. 当信息足够时，用 Final Answer 结束
5. 不要编造信息，依赖工具获取的观察结果
6. 如果搜索结果不够，可以尝试不同的搜索关键词
"""


# ============================================================
# 工具实现
# ============================================================

def search(query: str) -> str:
    """模拟搜索工具，返回与 query 相关的模拟信息"""
    mock_db = {
        "北京天气": "北京今天晴，气温 22°C，湿度 45%，风力 3 级。",
        "北京": "北京今天晴，气温 22°C，湿度 45%，风力 3 级。",
        "上海天气": "上海今天多云转阴，气温 28°C，湿度 70%，可能有小雨。",
        "上海": "上海今天多云转阴，气温 28°C，湿度 70%，可能有小雨。",
        "特斯拉 2024 交付": "特斯拉 2024 年全球交付量为 178.9 万辆，同比下降约 1%。",
        "特斯拉 2024": "特斯拉 2024 年全球交付量为 178.9 万辆，同比下降约 1%。",
        "比亚迪 2024 交付": "比亚迪 2024 年全球新能源车销量达到 427 万辆，同比增长 41.3%。",
        "比亚迪 2024": "比亚迪 2024 年全球新能源车销量达到 427 万辆，同比增长 41.3%。",
        "python": "Python 是一种解释型、面向对象的高级编程语言。",
        "python语言": "Python 是一种解释型、面向对象的高级编程语言。",
        "中国gdp 2024": "中国 2024 年 GDP 约为 134.9 万亿元人民币，同比增长 5.0%。",
    }
    query_lower = query.lower()
    for key, value in mock_db.items():
        if key in query_lower:
            return f"搜索结果: {value}"
    return f"搜索结果: 关于 '{query}' 未找到相关信息。"


def calculate(expression: str) -> str:
    """安全的数学表达式计算"""
    import ast
    import operator

    allowed_ops = {
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg,
    }

    def safe_eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            if type(node.op) not in allowed_ops:
                raise ValueError(f"不允许的运算符")
            return allowed_ops[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in allowed_ops:
                raise ValueError(f"不允许的运算符")
            return allowed_ops[type(node.op)](safe_eval(node.operand))
        elif isinstance(node, ast.Expression):
            return safe_eval(node.body)
        else:
            raise ValueError(f"不支持的表达式")

    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def get_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}，{now.strftime('%A')}"


# 工具名称到函数的映射
TOOLS = {
    "search": search,
    "calculate": calculate,
    "get_time": get_time,
}


# ============================================================
# 解析 LLM 的响应 —— 提取 Thought 和 Action
# ============================================================

def parse_react_output(text: str) -> dict:
    """
    解析 LLM 的 ReAct 格式输出。

    LLM 的输出格式为:
        Thought: [思考内容]
        Action: [工具名称]([参数])
    或:
        Thought: [思考内容]
        Final Answer: [最终回答]

    返回一个字典，包含解析结果。
    """
    result = {
        "thought": None,
        "action_type": None,   # "tool" 或 "final_answer"
        "tool_name": None,
        "tool_args": None,
        "final_answer": None,
        "raw": text,
    }

    # 提取 Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # 检查是 Final Answer 还是 Action
    if "Final Answer:" in text:
        result["action_type"] = "final_answer"
        fa_match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if fa_match:
            result["final_answer"] = fa_match.group(1).strip()
    elif "Action:" in text:
        result["action_type"] = "tool"
        # 解析 Action: tool_name(args)
        action_match = re.search(r"Action:\s*(\w+)\((.+?)\)", text)
        if action_match:
            result["tool_name"] = action_match.group(1)
            result["tool_args"] = action_match.group(2).strip().strip('"').strip("'")

    return result


# ============================================================
# ReAct Agent 主循环
# ============================================================

def run_react_agent(user_question: str, max_iterations: int = 10, verbose: bool = True) -> str:
    """
    运行 ReAct Agent。

    参数:
        user_question: 用户的问题
        max_iterations: 最大循环次数
        verbose: 是否打印详细的中间过程

    返回:
        Agent 的最终回答字符串
    """
    # 历史记录: 存储每一轮的 Thought、Action、Observation
    history = []

    # 当前上下文: 包含 system prompt、用户问题、以及所有历史和观察结果
    context = REACT_SYSTEM_PROMPT + "\n\n"

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f" 第 {iteration} 轮循环 ")
            print(f"{'='*70}")

        # ---- 步骤 1: 调用 LLM，获取 Thought + Action ----
        # 将当前上下文和历史一起发给 LLM
        prompt = context + f"\n用户问题: {user_question}\n\n"
        if history:
            prompt += "之前的步骤回顾:\n"
            for i, step in enumerate(history, 1):
                prompt += f"\n--- 步骤 {i} ---\n"
                prompt += f"Thought: {step['thought']}\n"
                prompt += f"Action: {step['action_raw']}\n"
                prompt += f"Observation: {step['observation']}\n"

        prompt += "\n请继续你的 Thought 和 Action (或 Final Answer):"

        # 调用 LLM
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "请根据以上信息继续思考。如果你已经有足够信息，请给出 Final Answer。"},
            ],
            temperature=0.1,
        )

        llm_output = response.choices[0].message.content

        if verbose:
            print(f"\n[LLM 输出]\n{llm_output}")

        # ---- 步骤 2: 解析 LLM 输出 ----
        parsed = parse_react_output(llm_output)

        if verbose:
            print(f"\n[解析结果]")
            print(f"  Thought: {parsed['thought'][:100]}..." if parsed['thought'] and len(parsed['thought']) > 100 else f"  Thought: {parsed['thought']}")
            print(f"  Type: {parsed['action_type']}")

        # ---- 步骤 3: 根据解析结果执行相应操作 ----
        if parsed["action_type"] == "final_answer":
            # LLM 认为任务完成，返回 Final Answer
            if verbose:
                print(f"\n[ReAct 完成] 共 {iteration} 轮")
            return parsed["final_answer"]

        elif parsed["action_type"] == "tool":
            # LLM 决定调用工具
            tool_name = parsed["tool_name"]
            tool_args = parsed["tool_args"]

            if verbose:
                print(f"\n[执行工具] {tool_name}({tool_args})")

            # 执行工具
            tool_func = TOOLS.get(tool_name)
            if tool_func is None:
                observation = f"错误: 未知工具 '{tool_name}'。可用工具: {list(TOOLS.keys())}"
            else:
                try:
                    observation = tool_func(tool_args)
                except Exception as e:
                    observation = f"工具执行错误: {e}"

            if verbose:
                print(f"[观察结果] {observation}")

            # 记录这一轮的历史
            history.append({
                "thought": parsed["thought"],
                "action_raw": f"{tool_name}({tool_args})",
                "observation": observation,
            })

        else:
            # 解析失败 —— LLM 的输出不符合格式
            if verbose:
                print(f"\n[警告] LLM 输出格式不符合 ReAct 规范，尝试修复...")

            # 在下一轮中提醒 LLM 遵循格式
            context += f"\n\n[系统提示] 你上一轮的输出格式不符合要求。请确保每一轮都包含 Thought: 和 Action: 或 Final Answer:。"

    # 达到最大循环次数，强制让 LLM 总结
    if verbose:
        print(f"\n[达到最大循环次数 {max_iterations}] 强制总结...")

    summary_prompt = f"""
    以下是到目前为止的所有信息:
    {json.dumps(history, ensure_ascii=False, indent=2)}

    请基于以上信息，给出你对原始问题的最终回答。

    原始问题: {user_question}
    请直接给出你的最终回答，不要再调用工具。
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.1,
    )
    return response.choices[0].message.content


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print(" ReAct Agent 演示 ")
    print("=" * 70)

    # ---- 测试 1: 多步信息检索 ----
    print("\n\n>>> 测试 1: 天气查询 + 温度换算")
    print("-" * 40)
    answer = run_react_agent("北京今天气温多少摄氏度？换算成华氏度是多少？")
    print(f"\n{'='*40}")
    print(f"最终答案:\n{answer}")

    # ---- 测试 2: 比较分析 ----
    print("\n\n>>> 测试 2: 数据比较")
    print("-" * 40)
    answer = run_react_agent("特斯拉和比亚迪2024年的交付量分别是多少？谁更多？")
    print(f"\n{'='*40}")
    print(f"最终答案:\n{answer}")
```

---

## 第七部分：ReAct 循环的常见问题和优化

### 7.1 循环不终止

**症状**：Agent 不断搜索同样的内容，或者在一个小循环里打转（搜索→不满意→换一个词搜索→不满意→换回来搜索→...）。

**原因分析**：
- LLM 没有明确的"信息足够"的判断标准
- 工具返回的信息对 LLM 来说不够清晰
- system prompt 中没有给出"何时停止"的指导

**解决方案**：
1. 在 system prompt 中加入明确的终止条件："当你拥有回答用户问题所需的所有信息时，立即给出 Final Answer，不要继续搜索或验证已经确认的信息。"
2. 设置最大循环次数（上限保护）
3. 在 Observation 中加入"信息充分度"的提示："这条信息看起来已经足够回答用户的问题了。"
4. 监控循环次数和内容重复度，如果检测到 Agent 在进行重复操作，强制介入

### 7.2 Thought 敷衍

**症状**：Agent 的 Thought 非常简单，形如 "我需要搜索"、"我搜到了"、"回答用户"——几乎没有实质性的分析。

**原因分析**：
- system prompt 对 Thought 的要求不够详细
- 或者 LLM 本身能力有限

**解决方案**：
1. 在 prompt 中给出 Thought 的质量标准："每个 Thought 应该回答三个问题：(1) 我现在知道了什么？(2) 我还不知道什么？(3) 为什么我下一步应该做 X 而不是 Y？"
2. 给出高质量的 Thought 示例作为 few-shot
3. 如果 LLM 输出的 Thought 太短或太敷衍，要求它重新输出

### 7.3 Action 参数错误

**症状**：Agent 想搜索"北京天气"，但填的参数是"我想知道北京天气怎么样"（一个完整的句子而不是搜索关键词），或者参数为空字符串。

**原因分析**：
- 工具 description 中对参数格式的说明不够清晰
- LLM 不理解"搜索关键词"和"自然语言问题"的区别

**解决方案**：
1. 在工具定义中明确写出参数格式要求和正例/反例
2. 在 system prompt 中加入通用的参数填写规则："搜索工具的参数应该是简洁的关键词，不是完整句子。好的例子: '北京天气'。不好的例子: '我想知道北京今天天气怎么样'。"
3. 在解析 Action 参数后，增加一个验证步骤——如果参数明显不对（过长、格式错误），可以提醒 LLM 修正

### 7.4 信息整合不佳

**症状**：Agent 成功搜索到了所有需要的信息，但 Final Answer 只是把搜索结果罗列了一遍，没有进行整合、分析或总结。

**原因分析**：LLM 把"信息检索"当成了"回答"，缺少"综合输出"的步骤。

**解决方案**：
1. 在 system prompt 中强调 Final Answer 的质量要求："Final Answer 不是搜索结果的简单堆砌。你需要对获取到的信息进行整合、比较、分析和总结，用你自己的语言组织成一个连贯、完整的回答。"
2. 在循环接近尾声时（比如已经搜索了多个关键词却没有给出 Final Answer），可以主动提醒 LLM："你已经积累了足够的信息。请综合分析这些信息，给出一个完整的 Final Answer。"

---

## 基础练习

### 练习 1：手动追踪 ReAct 循环

给定以下场景，手动写出一个 ReAct 循环的完整 Thought-Action-Observation 序列（不需要写代码，用文字描述每一步）：

用户问题："请帮我查一下 Python 3.12 相比 3.11 有哪些重要的新特性，并给出一个你认为最实用的特性示例。"

要求至少包含 3 轮 Thought-Action-Observation 循环，最后给出 Final Answer。

### 练习 2：对比三种 prompt 策略

用相同的工具集和相同的问题，分别测试以下三种 system prompt 策略，观察 LLM 的 ReAct 循环行为差异：

策略 A：只告诉 LLM"你可以使用工具"，不指定任何格式
策略 B：使用本章的 ReAct prompt 格式（Thought/Action/Observation）
策略 C：在策略 B 的基础上，加入 Self-Reflection 指令（每次 Action 后反思）

测试问题："上海今天天气适合户外运动吗？"

记录每种策略的：(1) 循环次数、(2) 工具调用正确率、(3) 最终回答质量（1-5分），写 200 字以上的分析。

### 练习 3：设计 Self-Ask 问题分解提示词

设计一个 system prompt，让 LLM 对复杂问题进行 Self-Ask 分解。测试问题："比较 Python、JavaScript 和 Go 三种语言在 Web 后端开发中的优劣势，并给出选择建议。"

要求：prompt 中包含明确的分解格式指导和综合回答要求。记录 LLM 的分解结果和综合回答质量。

---

## 进阶练习

### 练习 4：实现带验证的 ReAct Agent

在基础 ReAct Agent 的基础上，添加以下验证机制：

1. 每个 Observation 之后，LLM 需要输出一个 "Confidence: [高/中/低]" 来评估搜索结果的可靠性
2. 如果 Confidence 为"低"，LLM 应该自动用不同的搜索词重新搜索
3. 如果连续 3 次 Confidence 为"低"，LLM 应该告诉用户"我无法找到可靠的信息"并给出基于已有最佳信息的回答
4. 测试你的实现在"搜索一个不存在的信息"场景下的表现

### 练习 5：实现 ReAct 循环的可视化追踪

实现一个函数 `visualize_react_loop(history)`，接收 ReAct 循环的历史记录列表，生成一个文本形式的可视化追踪。要求：

1. 使用缩进和箭头（→）清晰展示每个步骤
2. 长文本（Thought 和 Observation）截断到 100 字符并在末尾添加 "..."
3. 高亮显示循环中的异常（如工具调用失败、重复搜索等）
4. 在末尾输出统计信息：总循环次数、总 token 消耗（估算）、有效步骤数、重复步骤数

---

## 常见错误

**错误 1：把 Thought 当成可有可无的装饰**
一些开发者在实现 ReAct 时，只在 prompt 中要求 LLM 输出 Action，Thought 被省略了。这样做得到的不是 ReAct，而是"盲目行动"——LLM 不知道为什么调用这个工具、调用完不知道分析结果、不知道什么时候该停。Thought 是 ReAct 的灵魂，不能省略。

**错误 2：Thought 和 Action 的顺序颠倒**
ReAct 的顺序必须是 Thought 在前、Action 在后。因为 Thought 应该解释"我为什么做这个 Action"。如果先输出 Action 再输出 Thought（类似于"我调用了搜索，因为我想查天气"），信息量就大大降低了——你无法在一个人"已经做了"之后再问他"你为什么这么做"。Thought 必须发生在 Action 之前（至少在逻辑上）。

**错误 3：在 Final Answer 中继续引用"搜索中"的信息**
有些 LLM 会在 Final Answer 中说"根据搜索结果，北京今天天气..."但实际上它搜索到的信息已经被整合了。应该直接用确定性的语气陈述结果，不需要再提"搜索"这个过程。

**错误 4：温度设置过高**
ReAct 循环要求 LLM 做出稳定的、可预测的决策。temperature 高于 0.5 会让 LLM 的行为变得不可预测——同样的问题，有时搜 2 次就停了，有时搜 10 次还不停。建议 ReAct Agent 使用 temperature <= 0.3。

**错误 5：Observation 太简略或被截断**
如果工具返回的 Observation 信息不完整（比如只截取了搜索结果的标题而没有摘要），LLM 可能无法做出正确的下一步判断。确保 Observation 包含足够的上下文信息。同时也要注意 Observation 不能太长——如果一次搜索返回了 5000 字，LLM 可能会被信息淹没。

**错误 6：忘记处理 LLM 格式输出错误**
LLM 不一定每次都严格按照你要求的格式输出。它可能忘记加 "Thought:" 前缀，可能把 Action 写成 "Action: 搜索北京天气"（缺少括号格式），可能在 Thought 中直接给出答案。你的解析器必须具备一定的容错能力，不能假设 LLM 的输出 100% 符合规范。

**错误 7：在 ReAct prompt 中不给格式示例**
很多开发者只是在 prompt 中用文字描述了格式要求（"请输出 Thought:...然后 Action:..."），但没有给出具体的示例。LLM 对"抽象的文字描述"的理解远不如对"具体的示例"的模仿。一定要在 prompt 中给出至少一个完整的 Thought-Action 示例。

**错误 8：Observation 以非结构化文本返回**
如果工具返回的是一个很长的非结构化文本，LLM 在下一轮 Thought 中可能难以提取关键信息。工具返回的 Observation 应该结构清晰、重点突出。可以考虑在返回文本前加一个小小的结构化标签，比如"[数据] 特斯拉 2024 交付量: 178.9万辆"。

---

## 本章小结

ReAct 是当前 LLM Agent 领域最基础的推理模式。本章从以下维度深入剖析了 ReAct：

1. **ReAct 的本质**：推理（Reasoning）和行动（Acting）的交替循环。推理为行动提供方向，行动为推理提供事实基础。两者协同，产生 1+1>2 的效果。

2. **循环机制**：Thought（分析→决策）→ Action（调用工具）→ Observation（获取反馈）→ 回到 Thought，直到信息足够，输出 Final Answer。

3. **与 CoT 的对比**：CoT 适合纯推理问题，ReAct 适合需要外部信息的任务。两者不是互斥的——好的 Thought 本身就应该包含 CoT 式的逐步推理。

4. **Self-Reflection**：让 LLM 在给出最终答案前自我检查。在 Agent 中的多个时机都可以引入反思机制，但要注意反思质量和过度反思的问题。

5. **Self-Ask**：LLM 自主分解复杂问题为子问题，逐一解决。与 ReAct 结合使用可以实现"先分解问题，再用工具获取每个子问题所需的数据"。

6. **实践要点**：好的 system prompt 设计、容错的输出解析、明确的终止条件、结构化的 Observation 返回，这些都是实现一个可靠的 ReAct Agent 的关键细节。

下一章我们将深入讨论 Agent 的记忆与状态管理——如何让 Agent 在长对话中记住关键信息，以及如何实现跨会话的持久记忆。
