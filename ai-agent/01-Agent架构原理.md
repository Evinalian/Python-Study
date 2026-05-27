# 第1章 Agent 架构原理

## 学习目标

完成本章后，你将能够：

1. 理解 Agent 与普通 LLM 调用的本质区别，从"一问一答"到"自主循环决策"
2. 掌握 Agent 的核心循环：感知、规划、执行、观察
3. 区分不同层次的 Agent 分类体系
4. 识别常见的 Agent 架构模式及其适用场景
5. 从零实现一个不依赖框架的完整 Agent
6. 了解主流 Agent 框架的定位和选型依据

## 前置知识

- 已完成 python-core 23 章，熟悉 Python 编程
- 已完成 pytorch-core 7 章，理解张量和神经网络基本概念
- 已完成 ai-application 9 章，有 LLM API 调用和 RAG 构建的实践经验
- 对 OpenAI SDK 的基本使用有了解

---

## 第一部分：什么是 Agent

### 1.1 从一个常见的误解开始

很多人在学完 RAG 之后，认为自己已经"在做 Agent 了"。他们写这样一个程序：用户输入一个问题，系统先去向量数据库检索相关文档，然后把检索结果拼到 prompt 里，最后调用 LLM 生成回答。这个过程叫"Agent 吗"？严格来说不是。这只是一个 chain——一个有固定步骤的管道。

管道的特征是：**路径在代码编写时就确定了**。无论用户问什么，系统都走同样的流程——先检索、再拼接、再生成。管道的控制流是程序员定义的，LLM 在这里只负责"生成文字"这一个环节。

Agent 的本质区别在于：**LLM 本身参与决策控制流**。在 Agent 系统中，LLM 不仅要生成回答，还要决定"下一步该做什么"。它可能决定先去查资料，看了资料后发现信息不够，于是决定再查一次；查完之后发现需要调用计算器做个数值计算；算完之后发现结果合理，于是把最终答案返回给用户。这个过程中，**每一步走什么路径，不是程序员预先写死的，而是 LLM 在运行时根据当前情况动态决定的**。

用一句话概括：**管道是固定的路线图，Agent 是实时的导航决策**。

### 1.2 深入理解"一问一答"与"自主循环"的区别

我们通过一个具体的场景来理解这两种模式。假设用户的问题是："北京今天的气温是多少摄氏度？换算成华氏度是多少？"

**普通 LLM 调用的处理方式：**

你直接调用 OpenAI API，把用户的问题作为 user message 传进去。LLM 返回一个回答。如果 LLM 训练数据中有北京今天的天气信息（这几乎不可能），它会给出一个数字；如果没有，它会编造一个数字，或者告诉你它无能为力。

**Chain（管道）的处理方式：**

你写了一段代码，在调用 LLM 之前，先去天气 API 查询北京今天的气温。拿到气温数据后，你把它拼接进 prompt，然后调用 LLM，让 LLM 回答摄氏度和华氏度两个值。这一步对很多问题够用了。

但这里有一个问题你发现了吗？**华氏度的换算，你让 LLM 做了，而 LLM 做数学运算是不可靠的。** 你可以在管道中再加一个步骤——调用 LLM 之前先用 Python 把摄氏度转成华氏度——但这样做的话，你的管道就越来越复杂。而且如果用户下次问的是"上海明天的气温"，或者"北京和深圳的温差"，你的管道逻辑又需要调整。

**Agent 的处理方式：**

Agent 拿到用户问题后，它的"大脑"（LLM）会自主思考：

1. "我需要知道北京今天的气温。我有一个天气查询工具，可以调用它。" —— 调用天气 API，拿到结果："北京今天 22°C"
2. "我拿到了摄氏温度 22°C，还需要换算成华氏度。我不能直接心算，我有个计算工具，用它来算。" —— 调用计算工具 `22 * 9 / 5 + 32 = 71.6`
3. "现在我有所有需要的信息了。我可以给用户最终答案：北京今天 22°C，相当于 71.6°F。"

注意区别：**这套流程不是程序员预先写好的。** 程序员只提供了工具（天气查询、计算器），而 Agent 自己决定在什么时候调用哪个工具、调用几次、什么顺序。甚至如果天气 API 第一次调用失败返回了错误，Agent 还会自己决定是否重试、是否换个城市名再查。

### 1.3 Agent 的形式化定义

在学术文献和工程实践中，Agent 可以形式化地定义为：

> 一个 Agent 是一个系统，它能够**感知**环境状态，通过内部**推理**做出**决策**，**执行**行动来改变环境状态，并从执行结果中**学习**和**适应**。

这个定义中的五个关键词构成了 Agent 的五个核心能力：

- **感知（Perceive）**：获取外部信息和当前状态的能力。对于 LLM Agent 来说，感知的来源包括用户输入、工具返回的结果、对话历史、外部数据源等。
- **推理（Reason）**：分析和理解当前状态、判断下一步应该做什么的能力。这是 LLM 的核心价值所在——它能够理解自然语言描述的复杂情境，并做出逻辑推理。
- **决策（Decide）**：从多个可能的行动中选择最优行动的能力。一个 Agent 可能有几十个工具可用，它需要判断在当前状态下调用哪个工具最合适。
- **执行（Act）**：实际执行选定行动的能力。这包括调用外部 API、运行代码、读写文件、发送消息等。
- **适应（Adapt）**：根据执行结果调整后续行为的能力。如果工具调用失败了，Agent 要能从错误中恢复；如果观察到新信息，Agent 要能更新自己的认知。

---

## 第二部分：Agent 的核心循环

### 2.1 感知阶段：我在哪里？我知道什么？

感知是 Agent 循环的起点。在每一次循环迭代中，Agent 首先需要明确当前的"状态"。这个状态包含哪些信息？

**用户输入**：用户最初的问题或指令。在整个会话中，这是相对固定的。但用户的意图可能在多轮交互中被逐步细化。

**对话历史**：从会话开始到现在的所有交互记录，包括用户说了什么、Agent 回复了什么、每次工具调用的输入和输出。对话历史承载了上下文，让 Agent 知道"我们进行到哪一步了"。

**最近观察**：最近一次工具调用的返回结果。这是最新鲜的信息，也是 Agent 下一步决策的最直接依据。

**系统状态**：一些全局性的信息，比如当前时间、可用工具列表、剩余 token 数等。这些信息帮助 Agent 做出更合理的决策——例如知道 token 快用完了就该尽快给结论，而不是继续绕圈子。

### 2.2 规划阶段：我要怎么做？

有了当前状态的全面感知后，Agent 需要制定行动计划。这个阶段是最能体现 LLM 价值的环节。

**对于简单任务**，规划可能只是一步："调用天气 API 获取北京天气"。

**对于中等复杂任务**，规划可能是一个小型的多步策略："先查天气，如果温度高于 30°C 就提醒用户防暑，如果低于 10°C 就提醒用户保暖。"

**对于高度复杂的任务**，规划可能涉及任务分解（task decomposition）。Agent 把一个复杂目标分解成多个子目标，然后逐个击破。例如用户说"帮我写一份关于人工智能发展趋势的研究报告"，Agent 可能分解为：

1. 搜索近两年 AI 领域的重大突破
2. 整理各个子领域（NLP、CV、RL 等）的进展
3. 分析产业发展和投资趋势
4. 撰写报告大纲
5. 按大纲逐节撰写内容
6. 生成参考文献列表

这就是 Plan-Execute 型 Agent 的核心思想——先规划，再执行。

### 2.3 执行阶段：动手做！

规划完成之后，Agent 需要把计划转化为实际行动。在 LLM Agent 的语境下，"执行"通常表现为调用一个工具（function call）。

执行阶段需要考虑的工程问题包括：

**参数构造**：LLM 需要根据当前状态，为工具调用构造正确的参数。例如天气查询工具需要城市名作为参数，LLM 必须从用户的自然语言输入中提取出这个参数值。

**异步调用**：有些工具调用可能耗时较长（比如搜索互联网、下载大文件）。Agent 系统需要考虑是同步等待还是异步处理。

**并行调用**：当多个工具调用之间没有依赖关系时，可以并行执行以提高效率。例如同时查询北京和上海的天气，而不是串行等一个查完再查另一个。

**超时与重试**：工具执行可能失败。Agent 需要有超时机制和重试策略，不能永远等下去。

### 2.4 观察阶段：结果如何？学到什么？

工具执行完毕后，Agent 读取执行结果，这就是观察阶段。观察阶段的核心问题是：

**结果解读**：工具返回的可能是一大段 JSON、一个错误码、一个空列表。Agent 需要正确解读这个结果的含义——调用成功了吗？返回的信息够用吗？

**状态更新**：新信息需要整合到 Agent 的当前状态中。如果天气 API 返回"北京 22°C"，Agent 需要记住这个信息，以便后续使用。

**计划调整**：根据观察结果，Agent 可能需要调整原有计划。如果查询结果为空，可能需要换一个搜索词重试；如果返回了意外的新信息，可能需要追加新的步骤。

**终止判断**：Agent 需要判断任务是否已经完成。如果完成了，给出最终答案；如果还没完成，回到规划阶段继续循环。

---

## 第三部分：Agent 的分类体系

### 3.1 按复杂度分类

#### 3.1.1 ReAct Agent（推理-行动循环）

ReAct 是 Reasoning + Acting 的缩写，是当前最基础也最广泛使用的 Agent 模式。它的核心思想很简单：LLM 交替进行"思考"和"行动"。

在一个 ReAct 循环中，每一步 LLM 都输出一个 Thought（思考当前状态，决定下一步）和一个 Action（具体的工具调用）。工具执行完毕后，Observation（观察结果）反馈给 LLM，LLM 进入下一轮 Thought。

ReAct Agent 适合中等复杂度的任务，比如信息检索、数据分析、多步计算。它的优势在于简洁直观，容易实现和调试。局限在于：当任务步骤很多时，LLM 容易"迷路"，忘记最初的目标；而且每一步都依赖前一步的结果，容易出现连锁错误。

**关于 ReAct 的深入讨论见第 3 章。**

#### 3.1.2 Plan-Execute Agent（计划-执行分离）

Plan-Execute Agent 将规划和执行拆分为两个阶段。

**规划阶段**：LLM 拿到用户任务后，不急于执行，而是先制定一个详细的执行计划。计划通常以步骤列表的形式呈现："第1步：做X；第2步：做Y；第3步：做Z。" 规划阶段的输出是一个结构化的计划文档。

**执行阶段**：执行器（可以是同一个 LLM 的另一个实例，也可以是专门的执行组件）按照计划逐步执行。每完成一步，检查执行结果，更新计划状态。

这种模式的优势在于：
- 规划阶段可以做得非常仔细，不着急动手，降低盲目行动的风险
- 执行阶段可以专注于"照计划做"，减少每一步的决策负担
- 计划本身可以作为 checkpoint，方便追踪进度

代价是：规划和执行各需要一轮（或多轮）LLM 调用，token 消耗更大。而且计划可能不完美——如果执行中发现计划有缺陷，需要重新规划。

#### 3.1.3 Multi-Agent（多智能体协作）

Multi-Agent 系统包含多个 Agent 实例，每个实例承担不同的角色，通过协作完成复杂任务。

典型的多 Agent 架构如下：
- **角色分工**：比如一个"研究 Agent"负责搜集信息，一个"写作 Agent"负责撰写文本，一个"审核 Agent"负责检查质量
- **层级结构**：一个"管理 Agent"负责任务分解和分派，多个"工作者 Agent"负责执行各自的任务
- **辩论模式**：多个 Agent 从不同角度分析同一个问题，互相质疑和补充，最终达成共识

Multi-Agent 的优势在于：
- 每个 Agent 的职责单一，prompt 可以高度定制
- 不同 Agent 可以有不同的工具集，提高工具管理的清晰度
- 多个视角可以减少单一 Agent 的盲点和偏见

代价也很明显：
- 多轮通信 = 大量的 API 调用和 token 消耗
- Agent 之间的协调是一大工程挑战
- 调试复杂度呈指数级增长

### 3.2 按能力层次分类

#### 3.2.1 单步工具调用

这是最基础的能力层次。LLM 收到用户消息后，判断是否需要调用工具，如果需要就调用一次，然后把结果返回给用户。没有多轮循环，没有复杂推理。

典型例子是 ChatGPT 早期的插件系统：用户说"帮我查一下今天的天气"，LLM 调用天气插件，拿到结果，直接回复。

#### 3.2.2 多步推理+工具

在这个层次，LLM 可以进行多轮推理和工具调用。它不只是"判断→调用→返回"，而是"判断→调用→观察→再判断→再调用→...→返回"。

典型例子是让 Agent 做在线调研："帮我查一下 OpenAI 最新的模型发布了哪些功能"。Agent 可能会先搜索"OpenAI latest model"，看了搜索结果觉得信息不够，再搜索"GPT-4o features"，再看结果，再搜索"OpenAI 2025 release"，最后综合所有搜索结果给出回答。

#### 3.2.3 自主任务分解

这是目前最高层次的能力。Agent 不仅知道要调用哪些工具，还能自主将一个大任务分解成多个子任务，并为每个子任务规划执行路径。整个过程不需要人工干预。

典型例子是 AutoGPT 或 BabyAGI 这类项目：你给一个目标"帮我做市场调研并生成一份 PPT"，Agent 自己分解出搜索竞争对手、分析市场趋势、收集财务数据、生成 PPT 大纲、逐页撰写内容等子任务，并逐个完成。

---

## 第四部分：Agent 的架构模式

### 4.1 无状态 Agent vs 有状态 Agent

**无状态 Agent**：每次用户输入，Agent 都从零开始处理，不记住之前发生了什么。它把所有的上下文（对话历史、工具调用记录等）都塞进当前请求的 messages 数组中。

无状态 Agent 的优点：
- 实现简单，不需要外部存储
- 每次请求独立，天然支持水平扩展
- 不会因为"记错东西"导致后续错误

无状态 Agent 的缺点：
- messages 数组越来越长，很快超出 token 限制
- 无法跨会话保持信息——每次新对话都要从头开始
- 对于长任务，上下文窗口迟早不够用

**有状态 Agent**：Agent 维护一个外部状态存储（可以是内存中的变量、数据库、向量存储等）。不是把所有历史都塞进 messages，而是提取最关键的摘要和上下文，按需注入。

有状态 Agent 涉及记忆管理，这部分内容将在第 4 章深入讨论。

### 4.2 单 Agent vs 多 Agent 协作

**单 Agent 模式**的适用场景：
- 任务边界清晰，不需要角色分工
- 工具数量可控（通常在 10-20 个以内）
- 步骤数有限（通常 10 步以内可以完成）
- 对延迟敏感（多 Agent 意味着多轮通信）

**多 Agent 模式**的适用场景：
- 任务涉及多个专业领域，需要不同"角色"的知识
- 工具数量很大（几十上百），一个 Agent 难以管理
- 需要"内部审查"机制——一个 Agent 写，另一个 Agent 审
- 任务步骤非常多，需要分层次规划和执行

**多 Agent 协作的关键挑战**：

1. **通信协议**：Agent 之间怎么交流？通过自然语言消息？还是结构化数据？
2. **任务分配**：谁负责决定哪个 Agent 做什么？是中心化的协调者还是分布式协商？
3. **冲突解决**：两个 Agent 的意见不一致怎么办？投票？还是让上级 Agent 裁决？
4. **状态共享**：所有 Agent 共享一个全局状态，还是各自维护私有状态？

---

## 第五部分：从零实现一个最简 Agent

### 5.1 设计思路

在动手写代码之前，我们先想清楚我们要构建一个什么样的 Agent。

**目标**：一个不依赖任何 Agent 框架、只用 OpenAI SDK 实现的，能自动使用工具的对话 Agent。

**核心组件**：
1. 一个 LLM 调用接口（OpenAI SDK）
2. 一组工具（Python 函数，带 JSON Schema 描述）
3. 一个循环控制器（管理"思考→调用→观察"的循环）
4. 消息历史管理器（存储对话和调用记录）

**循环逻辑**：
1. 用户输入加入消息列表
2. 调用 LLM，让 LLM 决定是否调用工具
3. 如果 LLM 返回文本（无工具调用）：这就是最终回答，循环结束
4. 如果 LLM 返回工具调用：执行该工具，将结果作为新消息加入列表，回到步骤 2
5. 如果循环次数超过上限：强制终止，让 LLM 基于已有信息给出回答

### 5.2 代码实现

```python
"""
最简 Agent 实现 —— 不依赖任何 Agent 框架，仅用 OpenAI SDK
展示 Agent 核心循环：感知 → 规划 → 执行 → 观察

运行前请确保：
    pip install openai
    并设置环境变量 OPENAI_API_KEY
"""

import json
import os
from openai import OpenAI

# ============================================================
# 第一步：初始化 LLM 客户端
# ============================================================
# 这里我们使用 OpenAI 兼容的 API。如果你用的是其他 LLM 提供商
# （如 DeepSeek、通义千问等），只需修改 base_url 即可。
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# 第二步：定义工具
# ============================================================
# 每个工具由两部分组成：
#   1. JSON Schema（告诉 LLM 这个工具是干什么的、需要什么参数）
#   2. 执行函数（实际干活的 Python 代码）
#
# JSON Schema 的 description 字段特别重要 —— LLM 就是靠它来判断
# "这个工具能不能解决我现在的问题"。description 写得不好，
# LLM 就不会调用你的工具，或者乱调。

def search_web(query: str) -> dict:
    """
    模拟网络搜索工具。
    在实际生产环境中，这里会调用真实的搜索 API（如 Google Search、Bing 等）。
    这里我们用一个简单的模拟实现来展示工具的结构。
    """
    # ---- 模拟搜索结果 ----
    mock_database = {
        "北京天气": "北京今天晴，气温 22°C，湿度 45%，风力 3 级。",
        "北京": "北京今天晴，气温 22°C，湿度 45%，风力 3 级。",
        "上海天气": "上海今天多云转阴，气温 28°C，湿度 70%，可能有小雨。",
        "上海": "上海今天多云转阴，气温 28°C，湿度 70%，可能有小雨。",
        "python": "Python 是一种解释型、面向对象的高级编程语言，由 Guido van Rossum 于 1991 年首次发布。",
        "openai": "OpenAI 是一家人工智能研究公司，开发了 GPT 系列大语言模型、DALL-E 图像生成模型等产品。",
    }
    # 模糊匹配：只要 query 中包含关键字就返回对应结果
    for key, value in mock_database.items():
        if key in query.lower():
            return {"success": True, "query": query, "result": value}
    return {"success": True, "query": query, "result": f"关于 '{query}' 的搜索结果：未找到相关信息（模拟）。"}


def calculate(expression: str) -> dict:
    """
    安全的数学表达式计算工具。
    使用 Python 的 eval 时要格外小心 —— 生产环境中必须加沙箱。
    这里我们用 ast.literal_eval 做安全限制，但真正的安全沙箱需要更多工作。
    """
    import ast
    import operator

    # ---- 安全的数学表达式求值 ----
    # 只允许基本的数学运算和白名单中的函数
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def safe_eval(node):
        """递归安全求值 AST 节点"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"不允许的运算符: {op_type.__name__}")
            return allowed_operators[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = safe_eval(node.operand)
            op_type = type(node.op)
            if op_type not in allowed_operators:
                raise ValueError(f"不允许的运算符: {op_type.__name__}")
            return allowed_operators[op_type](operand)
        elif isinstance(node, ast.Expression):
            return safe_eval(node.body)
        else:
            raise ValueError(f"不支持的表达式类型: {type(node).__name__}")

    try:
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree)
        return {"success": True, "expression": expression, "result": result}
    except Exception as e:
        return {"success": False, "expression": expression, "error": str(e)}


def get_current_time() -> dict:
    """获取当前时间"""
    from datetime import datetime
    now = datetime.now()
    return {
        "success": True,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": now.strftime("%A"),
    }


# ============================================================
# 第三步：构建工具的 JSON Schema 列表
# ============================================================
# 这是给 LLM 看的"工具说明书"。每个工具的 Schema 包含：
#   - name: 工具的唯一标识名（LLM 用这个来指定要调用哪个工具）
#   - description: 工具的用途说明（LLM 靠这个判断该不该用这个工具）
#   - parameters: 参数的 JSON Schema（LLM 按这个格式填充参数值）
#
# description 的写法要点：
#   1. 说清楚这个工具干什么用
#   2. 说清楚什么情况下该用这个工具
#   3. 说清楚什么情况下不该用这个工具（避免误调用）
#   4. 如果能给出使用场景示例，LLM 的理解会更准确

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "在互联网上搜索信息。当你需要获取实时信息、事实性知识、或者你不确定的信息时，使用此工具。注意：简单的常识性问题不需要搜索。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，用简洁的语言描述你想查询的内容。例如 '北京今天天气' 而不是 '我想知道北京今天天气怎么样'。",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算。当你需要进行算术运算、数学表达式求值时使用此工具。支持的运算：加减乘除、幂运算。例如：'3.14 * 25 + 100'、'2 ** 10'。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "需要计算的数学表达式字符串。例如 '22 * 9 / 5 + 32'（摄氏度转华氏度）。",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。当你需要知道现在是什么时间、今天星期几时使用。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# ---- 工具名称到执行函数的映射 ----
TOOL_FUNCTIONS = {
    "search_web": search_web,
    "calculate": calculate,
    "get_current_time": get_current_time,
}


# ============================================================
# 第四步：Agent 核心循环
# ============================================================
# 这是 Agent 的心脏。循环的每一步都是：
#   1. 把当前的所有消息（对话历史+工具调用记录）发给 LLM
#   2. 检查 LLM 的回复：是文字还是工具调用？
#   3. 如果是文字 → 返回给用户，循环结束
#   4. 如果是工具调用 → 执行工具，将结果加入消息列表，回到步骤 1
#
# 同时设置了最大循环次数（MAX_ITERATIONS），防止 Agent 陷入无限循环。
# 当循环次数达到上限时，强制让 LLM 基于已有信息给出最终回答。

MAX_ITERATIONS = 10  # 最大循环次数，防止无限循环


def run_agent(user_message: str, system_prompt: str = None) -> str:
    """
    运行 Agent 的核心循环。

    参数:
        user_message: 用户的输入消息
        system_prompt: 系统提示词（可选），用于设定 Agent 的行为模式

    返回:
        Agent 的最终回答（字符串）
    """
    # ---- 初始化消息列表 ----
    # messages 是整个对话的完整历史记录。它包含：
    #   - system message: 设定 Agent 的角色和行为
    #   - user messages: 用户的输入
    #   - assistant messages: LLM 的回复（包括文字和工具调用请求）
    #   - tool messages: 工具执行的结果
    if system_prompt is None:
        system_prompt = (
            "你是一个智能助手，可以使用工具来帮助用户解决问题。\n"
            "请遵循以下原则：\n"
            "1. 仔细分析用户的问题，判断是否需要使用工具\n"
            "2. 如果需要实时信息或你不确定的信息，使用 search_web 工具\n"
            "3. 如果需要数学计算，使用 calculate 工具，不要心算\n"
            "4. 如果所有必要信息都已获取，直接给出最终回答\n"
            "5. 回答问题时引用你获取到的实际数据，不要编造信息"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    iteration = 0  # 当前循环次数

    # ---- 主循环 ----
    while iteration < MAX_ITERATIONS:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"第 {iteration} 轮循环")
        print(f"{'='*60}")

        # 步骤 1: 调用 LLM
        # 将整个 messages 发给 LLM，让 LLM 基于完整上下文做决策
        # tool_choice="auto" 表示让 LLM 自己决定要不要调工具
        response = client.chat.completions.create(
            model="gpt-4o",  # 建议使用支持 function calling 的模型
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",  # LLM 自主决定是否调用工具
            temperature=0.1,     # 低温度让决策更稳定
        )

        # 获取 LLM 的回复
        assistant_message = response.choices[0].message

        # 步骤 2: 判断 LLM 是否想调用工具
        # tool_calls 字段非空 = LLM 决定调用工具
        # tool_calls 字段为空 = LLM 给出了最终文字回答
        if assistant_message.tool_calls:
            # ---- 工具调用分支 ----
            # LLM 决定调用工具。我们需要：
            #   1. 把 LLM 的工具调用请求加入消息历史
            #   2. 逐个执行请求的工具
            #   3. 将每个工具的返回结果加入消息历史
            #   4. 回到循环开头，让 LLM 看到工具结果后继续思考

            # 把 LLM 的回复（包含 tool_calls）加入消息列表
            messages.append(assistant_message.model_dump())

            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"  >> 调用工具: {tool_name}({tool_args})")

                # 执行工具
                tool_function = TOOL_FUNCTIONS.get(tool_name)
                if tool_function is None:
                    tool_result = {"success": False, "error": f"未知工具: {tool_name}"}
                else:
                    try:
                        tool_result = tool_function(**tool_args)
                    except Exception as e:
                        tool_result = {"success": False, "error": str(e)}

                print(f"  << 工具返回: {json.dumps(tool_result, ensure_ascii=False)}")

                # 将工具执行结果加入消息历史
                # role="tool" 的消息必须关联到对应的 tool_call_id
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        else:
            # ---- 文字回答分支 ----
            # LLM 没有调用工具，直接给出了文字回答。
            # 这通常意味着 LLM 认为任务已完成，或者不需要使用工具。
            final_answer = assistant_message.content
            print(f"  >> Agent 最终回答: {final_answer}")
            return final_answer

    # ---- 达到最大循环次数 ----
    # 如果循环了 MAX_ITERATIONS 次还没停，说明任务太复杂
    # 最后一轮：强制让 LLM 总结
    print(f"\n达到最大循环次数 ({MAX_ITERATIONS})，强制要求总结...")
    messages.append({
        "role": "user",
        "content": "请基于以上的工具调用结果，给出你的最终回答。不要继续调用工具。",
    })
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1,
    )
    return response.choices[0].message.content


# ============================================================
# 第五步：测试 Agent
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("最简 Agent 演示")
    print("=" * 60)

    # ---- 测试 1: 需要搜索和计算的复合任务 ----
    print("\n\n>>> 测试 1: 天气查询 + 温度换算")
    print("-" * 40)
    result = run_agent("北京今天气温多少摄氏度？换算成华氏度是多少？")
    print(f"\n最终结果:\n{result}")

    # ---- 测试 2: 纯计算任务（不需要搜索） ----
    print("\n\n>>> 测试 2: 纯数学计算")
    print("-" * 40)
    result = run_agent("计算 (123 + 456) * 789 / 100")
    print(f"\n最终结果:\n{result}")

    # ---- 测试 3: 常识问题（不需要任何工具） ----
    print("\n\n>>> 测试 3: 常识问题")
    print("-" * 40)
    result = run_agent("Python 是什么？")
    print(f"\n最终结果:\n{result}")
```

### 5.3 代码详解

让我们逐段深入理解这个实现。

**LLM 客户端初始化**：我们使用 OpenAI SDK 创建一个客户端。`base_url` 参数使得这段代码可以适配任何 OpenAI 兼容的 API（如 DeepSeek、通义千问、本地部署的 vLLM 等）。如果你用的是其他提供商，只需要改这个 URL 和 API key 即可。

**工具的定义**：每个工具由两个东西组成——`TOOLS_SCHEMA`（给 LLM 看的说明书）和 `TOOL_FUNCTIONS`（实际执行的函数）。这种解耦的设计很重要：LLM 只需要知道工具的接口规范（名称、描述、参数类型），而不需要知道工具的内部实现。这样一来，你可以随时替换工具的实现（比如把模拟搜索换成真实的 Google Search API），而不需要改动给 LLM 的 Schema。

**工具 Schema 的 description 字段**是整个系统中最重要的"隐形代码"。LLM 判断"该不该用这个工具"的唯一依据就是 description。如果 description 写得模糊不清，LLM 可能会：
- 该用的时候不用（因为它不认为这个工具适合当前任务）
- 不该用的时候乱用（因为它误解了工具的用途）
- 用错了参数（因为不知道参数该填什么格式）

**Agent 核心循环**是整个系统的发动机。关键设计决策包括：

1. **tool_choice="auto"**：让 LLM 自己决定是否调用工具。你也可以设为 "required"（必须调工具）或 "none"（禁止调工具），或者指定某个工具名强制调用。

2. **消息历史的完整性**：每次调用 LLM 时，我们都发送完整的 messages 历史。这意味着 LLM 能看到从对话开始到现在的所有内容——用户说了什么、自己之前调用过哪些工具、每个工具返回了什么。这就是 Agent 的"记忆"——尽管这是一种朴素的无状态记忆。

3. **temperature=0.1**：低温度让 LLM 的行为更可预测、更稳定。在 Agent 场景中，我们需要 LLM 按照逻辑做出可靠的决策，而不是"创意发挥"。

4. **MAX_ITERATIONS**：安全带。理论上 LLM 应该自己判断何时完成任务，但实践中它可能会陷入循环（查了又查、算了又算）。最大循环次数是一个硬性的安全阀。

---

## 第六部分：Agent 框架选型

### 6.1 框架概述

近年来出现了大量 Agent 开发框架。它们提供了不同层次的抽象，从底层的"帮你管理 function calling 循环"到高层的"帮你编排多个 Agent 的协作"。选择合适的框架，取决于你的项目需求。

### 6.2 AutoGPT / BabyAGI：自主任务分解的先驱

**AutoGPT** 和 **BabyAGI** 是 2023 年出现的两个开创性项目，它们第一次向大众展示了"让 LLM 自主完成复杂任务"的可能性。

**核心理念**：给 Agent 一个目标，它自己分解成子任务，自己执行，自己检查。整个过程不需要人的干预。

**AutoGPT 的工作方式**：
- 你给定一个总体目标和一个名字
- Agent 自己生成任务列表，排列优先级
- 按照优先级逐个执行任务
- 每完成一个任务，根据结果更新任务列表
- 直到所有任务完成或达到停止条件

**适用场景**：研究型的、探索性的、长时运行的任务。比如"做一份关于 X 行业的市场调研报告"。

**局限性**：
- 不可控：Agent 可能会做大量无用的循环和搜索
- 不可预测：你很难预知它会花多长时间、消耗多少 token
- 质量不稳定：自主分解的质量取决于 LLM 的能力

**当前地位**：这两个项目更偏向概念验证和研究探索，一般不推荐直接用于生产环境。但它们的理念（自主任务分解、优先级排序、自我修正）已经成为后续所有框架的基石。

### 6.3 CrewAI：多 Agent 协作的开箱即用方案

**CrewAI** 的设计哲学是"像管理团队一样管理 Agent"。它把多个 Agent 组成一个 Crew（团队），每个 Agent 有明确的角色（Role）、目标（Goal）和背景故事（Backstory）。Agent 之间通过 Task 进行协作——一个 Task 指定了期望的输出和分配给哪个 Agent。

**核心概念**：
- **Agent**：一个具有特定角色和目标的 LLM 实例
- **Task**：一个需要完成的工作单元，有明确的描述和期望输出
- **Crew**：一组 Agent 和一个任务序列，CrewAI 负责编排它们的执行

**适用场景**：
- 需要角色分工的复杂项目（如"一个研究员 Agent 搜集资料，一个作家 Agent 撰写报告"）
- 希望用"自然语言描述角色和任务"来配置 Agent 行为的场景
- 不需要细粒度控制循环流程的场景

**局限性**：
- 灵活性有限：CrewAI 的编排逻辑是固定的，不适合需要高度定制化循环的场景
- 调试困难：多个 Agent 之间的消息传递是黑盒的，出了错不容易定位
- 成本较高：多个 Agent 轮流调用 LLM，token 消耗大

### 6.4 LangGraph：基于图的 Agent 编排引擎

**LangGraph** 是 LangChain 生态中的一个库，它将 Agent 的工作流建模为一个有向图（Directed Graph）。

**核心理念**：Agent 的执行流程是一个图，节点（Node）是处理步骤，边（Edge）是转换规则。LLM 的决策体现在"从当前节点应该走向哪个节点"的选择上。

**LangGraph 的关键特性**：

1. **显式的状态管理**：每个节点的输入输出通过一个共享的 State 对象传递。State 可以包含对话历史、中间变量、工具调用结果等。这种显式的状态管理比"把所有东西塞进 messages"更可控。

2. **条件边（Conditional Edges）**：不仅可以从节点 A 固定走到节点 B，还可以根据当前状态动态决定走向 B 还是 C。这天然适合实现 Agent 的"决策"逻辑——根据当前状态选择下一步行动。

3. **循环支持**：图可以包含循环，天然支持 Agent 的"思考-行动-观察"循环模式。

4. **检查点（Checkpointing）**：可以在任意节点保存状态快照，支持暂停恢复和回溯。

**适用场景**：
- 需要精确控制执行流程的 Agent 系统
- 复杂的、多分支的、带条件逻辑的工作流
- 需要状态持久化和回溯能力的长时任务

**学习曲线**：LangGraph 的抽象层次较高，你需要理解图、节点、边、状态、条件边等概念。但一旦上手，它的灵活性和可维护性都非常强。

### 6.5 框架选型速查表

| 需求 | 推荐方案 |
|------|----------|
| 快速验证一个 Agent 想法 | 裸写（OpenAI SDK），不用框架 |
| 简单的单 Agent + 少量工具 | 裸写，或 LangChain 的 AgentExecutor |
| 多 Agent 协作，角色分工明确 | CrewAI |
| 复杂的工作流，需要精确控制流程 | LangGraph |
| 研究性质的自主 Agent | 参考 AutoGPT/BabyAGI 的理念，用 LangGraph 实现 |
| 生产级 Agent 服务 | LangGraph + 自定义组件 |

**重要提示**：框架是工具，不是必需品。如果你的 Agent 逻辑很简单——一个循环、几个工具——裸写几十行代码就够了。引入框架会增加依赖、增加学习成本、增加调试难度。只有当你的项目复杂到"裸写难以管理"的程度时，框架才从负担变成助力。

---

## 基础练习

### 练习 1：扩展工具集

为第 5 部分的最简 Agent 添加一个新的工具：`translate`（翻译工具）。要求：
- 工具接受两个参数：`text`（待翻译文本）和 `target_language`（目标语言）
- Schema 描述清晰，让 LLM 知道什么情况下用翻译工具
- 翻译功能可以用一个简单的字典映射来模拟（中英互译即可）

### 练习 2：观察循环行为

修改第 5 部分代码中的 `MAX_ITERATIONS` 为 1，分别测试以下三种问题，观察和记录 Agent 的行为差异：
- "北京天气怎么样？"（需要搜索）
- "计算 3^8"（需要计算）
- "你好"（不需要任何工具）

解释为什么有些情况下 `MAX_ITERATIONS=1` 会导致 Agent 无法完成任务。

### 练习 3：实现任务分解提示词

设计一个 system prompt，让 Agent 在解决复杂问题之前，先输出一个执行计划（不需要修改代码逻辑，只改 prompt）。测试你设计的 prompt 在以下问题上的表现："帮我研究一下 Python 和 JavaScript 在 Web 开发中的优劣势，然后给出一个推荐。"

---

## 进阶练习

### 练习 4：实现超时和错误重试

为 Agent 添加以下机制：
- 每个工具调用设置 5 秒超时
- 如果工具调用失败（返回 success=False 或超时），Agent 自动重试一次
- 如果重试后仍然失败，Agent 应该优雅地告知用户无法完成该步骤，而不是直接崩溃

### 练习 5：对比不同 Agent 模式的效率

分别用以下三种方式实现同一个任务："查询上海和深圳的天气，对比两地的温差，并判断哪边更适合周末出游"：
- 方式 A：裸写管道（硬编码调用顺序）
- 方式 B：ReAct Agent（本章实现的模式）
- 方式 C：Plan-Execute 模式（先规划再执行）

分析三种方式的 token 消耗、执行时间、灵活性和可维护性，写一段 300 字以上的分析。

---

## 常见错误

**错误 1：混淆管道和 Agent**
管道是"写完代码就知道执行路径"，Agent 是"运行时动态决定执行路径"。很多开发者把"在 prompt 里加了一些指令让 LLM 决定调不调工具"当作 Agent，但如果工具调用的顺序和条件都是代码写死的，那就还是管道。Agent 的关键特征是：**LLM 参与决策**。

**错误 2：在 Agent 循环中使用高 temperature**
Agent 的决策需要稳定性和可重复性。temperature 设为 0.7 或更高，会让 LLM 的决策变得随机——有时决定调工具 A，有时决定调工具 B，有时直接回答。这会让 Agent 的行为不可预测。建议 Agent 循环使用 temperature <= 0.3。

**错误 3：没有设置最大循环次数**
LLM 可能会陷入"反复搜索同一个东西"或"反复计算同一个结果"的循环。如果不设上限，你的 API 费用就会无限增长。MAX_ITERATIONS 是最基本的安全措施。

**错误 4：Tool Schema 的 description 写得像函数文档**
给 LLM 看的 description 和给人看的函数文档写法不同。对人你可以写"执行网络搜索"，对 LLM 你需要写"在互联网上搜索实时信息。当你需要获取最新数据、你不确定的事实、或者训练数据截止日期之后的信息时使用此工具。注意：简单的常识性问题不需要搜索。"——提供使用场景和不使用的场景参考。

**错误 5：把所有逻辑都塞进 Agent 循环**
Agent 不是万能的。对于确定性的、规则明确的任务（比如数据格式转换、固定流程的审批），用传统的代码逻辑更可靠、更快、更便宜。Agent 适合那些"需要灵活判断"的环节。

**错误 6：一个 Agent 挂几十个工具**
工具越多，LLM 选错的概率越大。如果你有 80 个工具，考虑分组——不同的子 Agent 管理不同的工具组。或者用层级结构——先让一个 Agent 判断用户意图属于哪个领域，再调用那个领域的专职 Agent。

**错误 7：忘记在消息历史中保留完整的工具调用记录**
每次调用 LLM 都必须附带完整的对话历史，包括之前的工具调用和返回结果。如果你"简化"了历史（比如只保留用户消息和最终回答），Agent 就失去了"我是怎么走到这一步的"的上下文，后续决策将变得盲目。

**错误 8：工具执行函数不做错误处理**
如果工具执行函数因为一个异常而崩溃，整个 Agent 循环就断了。始终在工具执行函数中捕获异常，并返回结构化的错误信息（如 `{"success": False, "error": "具体错误信息"}`），让 Agent 能够理解发生了什么并决定如何应对。

**错误 9：迷信框架，忽视理解原理**
学会了 LangChain 的 AgentExecutor 不等于理解了 Agent。框架封装了底层细节，你点了"运行"就知道结果。但当 Agent 行为不符合预期时，如果你不理解底层的循环逻辑、消息传递机制和工具调用流程，你将完全无法调试。在学习阶段，先裸写，再框架。

**错误 10：忽略 token 消耗的成本意识**
Agent 的每次循环都会累积消息历史。如果一个 Agent 循环了 10 轮，每轮的消息历史都比上一轮更长，token 消耗是累积增长的。在设计和测试 Agent 时，始终关注 token 消耗。一个实用的技巧是：在每轮循环开始时打印消息历史的预估 token 数。

---

## 本章小结

本章从"Agent 是什么"这个基本问题出发，逐步深入到 Agent 的循环机制、分类体系、架构模式和框架选型。核心要点回顾：

1. **Agent 的本质**：LLM 参与决策控制流，而不是仅在固定管道中生成文字。
2. **核心循环**：感知（获取当前状态）→ 规划（决定做什么）→ 执行（调用工具）→ 观察（解读结果），循环直到任务完成。
3. **分类体系**：从简单的 ReAct Agent 到 Plan-Execute Agent 到 Multi-Agent，复杂度递增，适用场景不同。
4. **架构选择**：无状态 vs 有状态、单 Agent vs 多 Agent，各有取舍。
5. **框架选型**：简单场景裸写即可，复杂场景 LangGraph 提供强大编排能力，CrewAI 适合角色分工明确的多 Agent 场景。

下一章我们将深入探讨 Agent 的工具系统设计——如何设计好的工具 Schema、如何管理大量工具、如何处理工具执行中的各种异常。
