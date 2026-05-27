# 第七章 LangGraph 工作流引擎

## 学习目标

完成本章后，你将能够：

1. 理解为什么 Agent 逻辑需要工作流引擎——线性调用链无法表达分支、循环和人机交互
2. 掌握 LangGraph 的核心概念：StateGraph、Node、Edge、State
3. 使用条件边实现 Agent 的分支决策逻辑
4. 构建循环工作流：Agent 反复调用工具直到满足终止条件
5. 实现 Human-in-the-Loop：在工作流关键节点暂停等待人类审批
6. 使用子图嵌套将复杂子流程封装为可复用的组件
7. 对流式状态输出建立理解，知晓如何实时观察 Agent 执行过程
8. 构建一个完整的 Agent 工作流：ReAct 循环 + 条件分支 + 人工审批

## 前置知识

- ReAct 推理模式（第03章）：理解 Thought-Action-Observation 循环
- 工具系统设计（第02章）：工具注册、Tool Schema、工具执行
- Plan-Execute 模式（第05章）：理解任务分解和状态管理
- 图论基础概念：节点、边、有向图、状态机
- Python 类型注解：`TypedDict`、`Annotated`、`Literal`
- LLM API 调用：OpenAI SDK 的 Function Calling 机制

---

## 1. 为什么需要工作流引擎

### 1.1 线性调用链的局限

在前几章中，我们构建的 Agent 大多遵循一个相对简单的执行模式：

```
用户输入 → Agent 思考 → 调用工具 → 返回结果 → 结束
```

或者稍微复杂一点：

```
用户输入 → Agent 思考 → 调用工具 → Agent 再思考 → 返回结果 → 结束
```

这种线性模式的问题在于：**它假设 Agent 的执行路径是可预测的、不会分叉的**。但真实的 Agent 行为远比这复杂：

- **分支决策**：Agent 调用工具后，根据返回结果走不同的处理路径。"如果数据库查询返回空 → 使用缓存数据；如果返回正常 → 直接分析"
- **循环执行**：Agent 可能需要多次调用工具，每次根据上次结果决定是否继续。"搜索 → 分析结果 → 不满足 → 再搜索 → ..."
- **人机交互**：某些关键步骤需要人类审批后才能继续。"Agent 写完代码 → 暂停 → 人类审查 → 批准/驳回 → 继续"
- **错误恢复**：某步失败时，根据失败类型走不同的恢复路径。"网络超时 → 重试；权限拒绝 → 升级给人类"

用代码来描述这些复杂逻辑，很快就会变成难以维护的"意大利面条"：

```python
# 这种"if-else 嵌套 + while 循环"的模式有什么问题？
def execute_agent(task: str):
    result = call_llm(task)
    
    if result.needs_tool:
        tool_result = call_tool(result.tool_name, result.tool_args)
        
        if tool_result.success:
            analysis = call_llm(f"分析: {tool_result.data}")
            
            if analysis.needs_more_data:
                more_data = call_tool("fetch_more", ...)
                # ... 又一层嵌套 ...
                pass
            elif analysis.is_complete:
                if analysis.requires_approval:
                    approved = ask_human(analysis.summary)
                    if not approved:
                        # 回到分析步骤...
                        pass
                return analysis.final_result
            else:
                # 又一种情况...
                pass
        else:
            if tool_result.error_type == "timeout":
                # 重试...
                pass
            elif tool_result.error_type == "permission":
                # 升级...
                pass
            else:
                # 放弃...
                pass
    else:
        return result.content

# 问题：
# 1. 逻辑和 LLM 调用混在一起，难以测试
# 2. 加一个新分支要改很多地方
# 3. 无法可视化执行流程
# 4. 无法插入通用逻辑（如日志、超时、重试）
```

### 1.2 工作流引擎的解决方案

工作流引擎的核心思想是**将 Agent 的执行逻辑建模为有向图**：

- **节点（Node）**：执行具体操作的单元。每个节点是一个纯函数——输入状态，输出状态更新
- **边（Edge）**：定义节点之间的转移规则。分为普通边（无条件转移）和条件边（根据状态决定下一节点）
- **状态（State）**：在节点之间流转的数据结构。所有节点共享同一份状态，但每个节点只读写自己关心的部分

这样一来，复杂的 Agent 逻辑就可以用一张图来表达：

```
        ┌──────────┐
        │  START   │
        └────┬─────┘
             │
             ▼
        ┌──────────┐
        │ 思考     │ ← Agent 调用 LLM 分析当前状态
        └────┬─────┘
             │ 条件边: 需要调用工具？
        ┌────┴────┐
        │         │
        ▼         ▼
   ┌────────┐ ┌──────────┐
   │调工具   │ │ 生成答案  │ ← 不需要工具，直接回答
   └───┬────┘ └────┬─────┘
       │            │
       │            ▼
       │       ┌──────────┐
       │       │ 人类审批？│ ← 需要人工审批吗？
       │       └────┬─────┘
       │        ┌───┴───┐
       │        ▼       ▼
       │   ┌────────┐ ┌──┐
       │   │等待审批 │ │END│
       │   └───┬────┘ └──┘
       │       │
       │       ▼
       │   审批通过？
       │   ┌──┴──┐
       │   ▼     ▼
       │  END  ┌────────┐
       │       │回到思考 │ ← 审批不通过，重新思考
       │       └────────┘
       │
       └──────────┘ (工具结果返回思考)
```

这就是 LangGraph 的核心价值：**把 Agent 的控制流从代码中"提升"到图中，让流程变得可视化、可组合、可测试**。

### 1.3 LangGraph 与 LangChain 的关系

LangGraph 是 LangChain 生态系统的一部分，但它是一个独立的库。简单理解：

- **LangChain**：提供了与 LLM 交互的组件（ChatModel、Prompt、Tools、Retrieval 等）
- **LangGraph**：在 LangChain 组件之上，提供了编排这些组件的图执行引擎

你可以把 LangChain 的组件看成"乐高积木块"，把 LangGraph 看成"搭积木的图纸和框架"。LangGraph 本身不关心节点的具体实现——你可以用 LangChain 的 Runnable，也可以用纯 Python 函数，甚至可以混合使用。

```python
"""
LangGraph 核心概念速览 —— 在深入细节之前，先建立全局认知。

安装:
    pip install langgraph langchain langchain-openai

核心概念对应关系:
    StateGraph  →  有向图容器，你往里添加节点和边
    Node        →  图中的顶点，每个节点是一个 Python 函数
    Edge        →  图中的边，定义了"执行完 A 后去 B"
    State       →  在节点之间流转的数据（TypedDict / Pydantic Model）
    Conditional Edge → 根据 State 的值决定去哪个节点
"""
from typing import TypedDict, Literal, Annotated
import operator

# LangGraph 使用 TypedDict 或 Pydantic 模型定义状态
# 状态中的每个字段定义了"这个字段在节点间如何合并"
class AgentState(TypedDict):
    """
    Agent 的状态定义。
    
    这是整个工作流中所有节点共享和修改的数据结构。
    每个节点可以:
    - 读取 State 中的任意字段
    - 返回 State 的部分更新（LangGraph 会自动合并）
    """
    # messages 字段: 对话历史列表
    # Annotated[list, operator.add] 表示"追加"而非"替换"——
    # 当节点返回 {"messages": [new_msg]} 时，新消息追加到历史，不覆盖
    messages: Annotated[list, operator.add]
    
    # 普通字段: 没有 Annotated → 节点返回的新值会替换旧值
    next_step: str           # 下一步做什么 ("continue" | "end" | "human_approval")
    tool_calls: list[dict]   # 当前需要执行的工具调用
    iteration_count: int     # 已执行的循环次数
    final_answer: str        # 最终回复


# 演示：一个最小化的 LangGraph 工作流
if __name__ == "__main__":
    print("=== LangGraph 核心概念速览 ===")
    print("""
    StateGraph: 定义工作流的"骨架"
    ├── 添加节点: graph.add_node("node_name", node_function)
    ├── 添加边:   graph.add_edge("from_node", "to_node")
    ├── 添条件边: graph.add_conditional_edges("from_node", router_function, route_map)
    ├── 设入口:   graph.set_entry_point("first_node")
    └── 编译:     app = graph.compile()
    
    然后:
    result = app.invoke(initial_state)  ← 同步执行
    result = await app.ainvoke(initial_state)  ← 异步执行
    
    流式输出:
    async for chunk in app.astream(initial_state):
        print(chunk)  ← 实时观察每一步的状态变化
    """)
```

---

## 2. LangGraph 核心概念深入

### 2.1 StateGraph：状态图——工作流的骨架

StateGraph 是 LangGraph 的入口类。你创建一个 StateGraph 实例，然后往里面添加节点和边，最后编译得到一个可执行的应用。

创建一个 StateGraph 需要指定**状态类型**——这告诉 LangGraph "在这个工作流中流转的数据长什么样"。

```python
"""
StateGraph 的创建和使用。

注意: LangGraph 的 API 在 0.2.x 版本有重要变化。
本章代码基于 langgraph >= 0.2.0。
如果使用 0.1.x，部分 API 不同（主要是消息类型）。
"""
from typing import TypedDict, Literal, Annotated
import operator
from langgraph.graph import StateGraph, END


# ============================================================
# 步骤1: 定义 State
# ============================================================
class SimpleState(TypedDict):
    """
    一个简单的工作流状态。
    
    为什么用 TypedDict 而不是普通 dict？
    - 类型安全：IDE 可以自动补全，写错字段名会报错
    - LangGraph 可以检查节点返回的键是否在 State 中定义
    - 可以为每个字段定义 reducer（合并策略）
    
    Annotated[list, operator.add] 的含义:
    - Annotated 是 Python 的类型注解增强
    - 第一个参数 list 是类型
    - 第二个参数 operator.add 是 reducer 函数
    - reducer 决定"当两个节点都返回同一字段时，如何合并"
    - operator.add 对 list 来说是拼接：[1,2] + [3,4] = [1,2,3,4]
    - 如果没有 Annotated → 默认是"替换"（后返回的覆盖先返回的）
    """
    counter: int            # 简单计数器（替换策略）
    messages: Annotated[list[str], operator.add]  # 消息列表（追加策略）
    current_step: str       # 当前步骤描述


# ============================================================
# 步骤2: 创建 StateGraph
# ============================================================
graph = StateGraph(SimpleState)
# graph 现在是一个"空图"，还没有任何节点和边


# ============================================================
# 步骤3: 定义节点函数
# ============================================================
# 每个节点函数接收 State，返回 State 的部分更新

def node_step_one(state: SimpleState) -> dict:
    """
    节点函数签名: (state) -> dict
    
    注意: 返回的是 dict，不是完整的 State。
    返回的 dict 中的键会被更新到 State 中。
    未返回的键保持原值不变。
    """
    print(f"[step_one] 当前 counter = {state['counter']}")
    return {
        "counter": state["counter"] + 1,
        "messages": ["Step One completed"],
        "current_step": "step_one_done",
    }


def node_step_two(state: SimpleState) -> dict:
    print(f"[step_two] 当前 counter = {state['counter']}")
    return {
        "counter": state["counter"] * 2,
        "messages": ["Step Two completed"],
        "current_step": "step_two_done",
    }


def node_step_three(state: SimpleState) -> dict:
    print(f"[step_three] 当前 counter = {state['counter']}")
    return {
        "counter": state["counter"] - 5,
        "messages": ["Step Three completed"],
        "current_step": "finished",
    }


# ============================================================
# 步骤4: 将节点添加到图中
# ============================================================
graph.add_node("step_one", node_step_one)
graph.add_node("step_two", node_step_two)
graph.add_node("step_three", node_step_three)

# 重要概念：graph.add_node() 只是"注册"节点，不定义执行顺序
# 执行顺序由边（Edge）来定义


# ============================================================
# 步骤5: 定义边——节点间的转移关系
# ============================================================

# 设置入口点：工作流从哪个节点开始
graph.set_entry_point("step_one")

# 添加普通边：执行完 A 后无条件转移去 B
graph.add_edge("step_one", "step_two")
graph.add_edge("step_two", "step_three")

# 添加终止边：执行完后结束工作流
graph.add_edge("step_three", END)
# END 是 LangGraph 的特殊节点，表示工作流结束


# ============================================================
# 步骤6: 编译——把图"固化"为可执行的应用
# ============================================================
app = graph.compile()


# ============================================================
# 步骤7: 执行
# ============================================================
if __name__ == "__main__":
    # 初始化状态
    initial_state: SimpleState = {
        "counter": 0,
        "messages": [],
        "current_step": "starting",
    }

    print("=" * 50)
    print("执行工作流")
    print("=" * 50)

    # invoke() 同步执行整个工作流
    final_state = app.invoke(initial_state)

    print(f"\n最终状态:")
    print(f"  counter: {final_state['counter']}")      # 预期: (0+1)*2-5 = -3
    print(f"  messages: {final_state['messages']}")
    # 预期: ['Step One completed', 'Step Two completed', 'Step Three completed']
    print(f"  current_step: {final_state['current_step']}")
```

### 2.2 节点：执行的原子单元

节点是 LangGraph 中最核心的单元。深入理解节点的行为对于设计正确的工作流至关重要。

```python
"""
LangGraph 节点的深入理解。

关键行为：
1. 节点函数接收完整的 State，返回 State 的部分更新（dict）
2. LangGraph 在每步之间自动应用 reducer 合并更新
3. 节点可以是同步或异步函数
4. 节点内部可以调用 LLM、执行工具、访问数据库——什么都可以
"""
from typing import TypedDict, Annotated
import operator
import asyncio
from langgraph.graph import StateGraph, END


class NodeDemoState(TypedDict):
    # 使用 operator.add 的消息列表
    messages: Annotated[list[str], operator.add]
    # 普通字段
    value: int


# ============================================================
# 节点类型1: 同步节点（最常见）
# ============================================================
def sync_node(state: NodeDemoState) -> dict:
    """同步节点 —— 直接返回更新字典"""
    return {
        "messages": [f"Sync node: value was {state['value']}"],
        "value": state["value"] + 10,
    }


# ============================================================
# 节点类型2: 异步节点
# ============================================================
async def async_node(state: NodeDemoState) -> dict:
    """异步节点 —— 适合调用 API、数据库等 I/O 操作"""
    await asyncio.sleep(0.1)  # 模拟 I/O
    return {
        "messages": [f"Async node: value was {state['value']}"],
        "value": state["value"] * 2,
    }


# ============================================================
# 节点类型3: 带异常处理的节点
# ============================================================
def robust_node(state: NodeDemoState) -> dict:
    """
    生产环境中的节点应该做异常处理。
    不要让一个节点的异常导致整个工作流崩溃。
    """
    try:
        # 业务逻辑
        if state["value"] < 0:
            raise ValueError("value 不能为负")
        result = state["value"] * 3
        return {
            "messages": [f"Robust node: success, result = {result}"],
            "value": result,
        }
    except Exception as e:
        # 错误处理：返回错误信息而非崩溃
        return {
            "messages": [f"Robust node: error - {e}"],
            "value": -1,  # 设置一个安全值
        }


# ============================================================
# 节点类型4: 不修改 State 的节点（只读节点）
# ============================================================
def read_only_node(state: NodeDemoState) -> dict:
    """只读节点 —— 读取 State 但不修改它"""
    # 打印状态用于调试/日志
    print(f"[DEBUG] 当前状态: value={state['value']}, "
          f"messages count={len(state['messages'])}")
    # 返回空字典 = 不修改任何东西
    return {}


# ============================================================
# 节点类型5: LLM 调用节点（最重要）
# ============================================================
def llm_node(state: NodeDemoState) -> dict:
    """
    调用 LLM 的节点 —— 这是 Agent 中最常见的节点类型。
    
    节点内部可以调用任何外部服务，LangGraph 不限制。
    """
    import os
    from openai import OpenAI
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # 用之前累积的 messages 作为上下文
    context = "\n".join(state["messages"][-5:])  # 最近 5 条消息
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个状态分析助手。"},
            {"role": "user", "content": f"基于以下上下文，请问接下来应该做什么？\n{context}"},
        ],
        temperature=0.0,
    )
    
    reply = response.choices[0].message.content
    return {
        "messages": [f"LLM: {reply}"],
    }


if __name__ == "__main__":
    print("=== 节点类型演示 ===")
    print("1. 同步节点 — 最常见的纯计算节点")
    print("2. 异步节点 — 用于 I/O 密集型操作（API 调用、数据库查询）")
    print("3. 鲁棒节点 — 带异常处理，防止单节点崩溃传播")
    print("4. 只读节点 — 用于日志、监控、调试")
    print("5. LLM 节点 — Agent 中最核心的节点类型")
```

### 2.3 边：控制流的定义

边定义了节点之间的转移关系。LangGraph 中有三种边：

**普通边（Normal Edge）**：执行完节点 A 后，无条件转移到节点 B。

**条件边（Conditional Edge）**：执行完节点 A 后，根据 State 的当前值决定去哪个节点。这是实现分支逻辑的关键。

**终止边（END）**：转移到 END 表示工作流结束。

```python
"""
LangGraph 边的深入理解 —— 重点是条件边。
"""
from typing import TypedDict, Literal, Annotated
import operator
from langgraph.graph import StateGraph, END


class EdgeDemoState(TypedDict):
    score: int
    messages: Annotated[list[str], operator.add]


# ============================================================
# 普通边演示
# ============================================================
def node_a(state: EdgeDemoState) -> dict:
    return {"messages": ["[A] 执行完毕"]}


def node_b(state: EdgeDemoState) -> dict:
    return {"messages": ["[B] 执行完毕"]}


# ============================================================
# 条件边演示 —— 核心概念
# ============================================================
def evaluator_node(state: EdgeDemoState) -> dict:
    """
    评估节点：根据 score 给出评级。
    但它不路由——路由由条件边的路由函数负责。
    """
    if state["score"] >= 80:
        grade = "excellent"
    elif state["score"] >= 60:
        grade = "pass"
    else:
        grade = "fail"
    return {
        "messages": [f"[Evaluator] score={state['score']}, grade={grade}"],
    }


def route_by_score(state: EdgeDemoState) -> Literal["excellent_handler", "pass_handler", "fail_handler"]:
    """
    路由函数 —— 条件边的核心。
    
    输入: 当前 State
    输出: 下一个节点的名称（必须是图中的节点名或 END）
    
    关键规则：
    - 返回的字符串必须是 add_node() 中注册的节点名
    - 返回 END 表示终止
    - 返回类型用 Literal 注解以提供类型安全
    """
    score = state["score"]
    if score >= 80:
        return "excellent_handler"
    elif score >= 60:
        return "pass_handler"
    else:
        return "fail_handler"


def excellent_handler(state: EdgeDemoState) -> dict:
    return {"messages": ["🏆 优秀！成绩 >= 80"]}


def pass_handler(state: EdgeDemoState) -> dict:
    return {"messages": ["✅ 及格！成绩 >= 60 但 < 80"]}


def fail_handler(state: EdgeDemoState) -> dict:
    return {"messages": ["❌ 不及格！成绩 < 60"]}


# ============================================================
# 复杂的条件路由 —— 多条件分支
# ============================================================
def complex_router(state: EdgeDemoState) -> str:
    """
    复杂的路由逻辑 —— 可以有多达任意多种分支。
    
    返回的字符串决定下一个节点。
    """
    score = state["score"]
    msg_count = len(state["messages"])
    
    # 条件1: 分数极高，跳过处理
    if score >= 95:
        return "direct_end"
    # 条件2: 分数高且消息多 → 需要总结
    elif score >= 80 and msg_count > 5:
        return "summarize_node"
    # 条件3: 分数中等
    elif score >= 50:
        return "process_node"
    # 条件4: 分数低
    else:
        return END  # 直接结束


# ============================================================
# 构建图（只展示结构，不完整）
# ============================================================
if __name__ == "__main__":
    print("=== 条件边演示 ===\n")

    graph = StateGraph(EdgeDemoState)

    graph.add_node("evaluator", evaluator_node)
    graph.add_node("excellent_handler", excellent_handler)
    graph.add_node("pass_handler", pass_handler)
    graph.add_node("fail_handler", fail_handler)

    graph.set_entry_point("evaluator")

    # ===== 条件边: add_conditional_edges =====
    # 参数: (源节点, 路由函数, 路由映射)
    graph.add_conditional_edges(
        "evaluator",                    # 从哪个节点出发
        route_by_score,                 # 路由函数
        {                               # 路由映射: 路由函数返回 → 目标节点
            "excellent_handler": "excellent_handler",
            "pass_handler": "pass_handler",
            "fail_handler": "fail_handler",
        }
    )
    # 路由映射的作用：
    # - LangGraph 用这个映射来验证路由函数的返回值是否合法
    # - 如果路由函数返回的值不在映射的 key 中 → 运行时错误
    # - 这是一种"白名单"安全机制

    # 所有 handler 执行完都结束
    graph.add_edge("excellent_handler", END)
    graph.add_edge("pass_handler", END)
    graph.add_edge("fail_handler", END)

    app = graph.compile()

    # 测试不同分数的路由
    for test_score in [95, 72, 45]:
        print(f"\n--- 测试: score = {test_score} ---")
        result = app.invoke({"score": test_score, "messages": []})
        print(f"  Path: {result['messages']}")
```

### 2.4 State：流转的数据

State 是 LangGraph 中最容易出错的地方。深入理解它的行为对于避免 bug 至关重要。

```python
"""
LangGraph State 的深入理解 —— Reducer 机制和常见陷阱。

核心概念：
1. State 在各节点间是"只读"的（节点不能直接修改传入的 state 对象）
2. 节点通过返回 dict 来"更新" State
3. 每个字段可以有自己的 reducer（合并策略）
4. Annotated[type, reducer] 定义了该字段的合并方式
"""
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END


# ============================================================
# Reducer 机制详解
# ============================================================
class ReducerDemoState(TypedDict):
    # === 追加策略 (operator.add) ===
    # 适用: 消息历史、日志、事件列表
    # 行为: 新值追加到旧值末尾
    log: Annotated[list[str], operator.add]
    
    # === 替换策略 (默认，无 Annotated) ===
    # 适用: 状态标记、计数器、配置
    # 行为: 新值完全替换旧值
    status: str          # 默认 reducer = 替换
    
    # === 自定义 Reducer ===
    # 适用: 需要特殊合并逻辑的字段
    # 这里: max_reducer 保留最大值
    # max_value: Annotated[int, lambda old, new: max(old, new)]
    # 如果自己定义 reducer，可以自由控制合并逻辑


def node_add_logs(state: ReducerDemoState) -> dict:
    """演示 operate.add 的效果"""
    return {
        "log": ["Node A executed"],  # 追加，不替换
        "status": "step_a_complete",  # 替换
    }


def node_add_more_logs(state: ReducerDemoState) -> dict:
    return {
        "log": ["Node B executed"],  # 追加到 ["Node A executed"] 后面
        "status": "step_b_complete",  # 替换 "step_a_complete"
    }


if __name__ == "__main__":
    graph = StateGraph(ReducerDemoState)
    graph.add_node("a", node_add_logs)
    graph.add_node("b", node_add_more_logs)
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    app = graph.compile()

    result = app.invoke({"log": [], "status": "initial"})
    print(f"log: {result['log']}")     # ['Node A executed', 'Node B executed']
    print(f"status: {result['status']}")  # 'step_b_complete'
    print(f"\n说明: log 字段使用 operator.add → 追加")
    print(f"      status 字段无 Annotated → 替换")
```

---

## 3. 构建完整的 ReAct Agent

现在我们将前面的概念整合起来，构建一个完整的 ReAct Agent 工作流。这是理解 LangGraph 的最佳方式——将它应用于你已经熟悉的模式。

```python
"""
LangGraph 构建 ReAct Agent —— 本章的核心实战。

ReAct 循环在 LangGraph 中的图表示:

        ┌──────────┐
        │  START   │
        └────┬─────┘
             │
             ▼
        ┌──────────┐
        │  agent   │ ← 调用 LLM，决定下一步做什么
        │ (思考)    │
        └────┬─────┘
             │ 条件边: should_continue
        ┌────┴─────────┐
        │              │
        ▼              ▼
   ┌─────────┐   ┌──────────┐
   │  tools  │   │   END    │ ← Agent 决定不再需要工具
   │(执行工具) │   └──────────┘
   └────┬────┘
        │
        └──→ 回到 agent 节点（循环）
"""
from typing import TypedDict, Annotated, Literal
import operator
import json
from langgraph.graph import StateGraph, END

# LangChain 集成
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage,
    BaseMessage
)
from langchain_core.tools import tool


# ============================================================
# 步骤1: 定义 State
# ============================================================
class AgentState(TypedDict):
    """
    ReAct Agent 的状态。
    
    messages: 使用 operator.add 意味着每次节点返回的 messages
              会被追加到现有列表末尾。这对于对话历史管理很重要。
    
    next_step: 标志位，控制下一步去哪
    iteration_count: 防止无限循环的计数器
    """
    messages: Annotated[list[BaseMessage], operator.add]
    next_step: str
    iteration_count: int


# ============================================================
# 步骤2: 定义工具
# ============================================================
@tool
def search_knowledge_base(query: str) -> str:
    """
    搜索内部知识库。用于查找公司产品、流程、策略等相关信息。
    
    参数:
        query: 搜索查询字符串
    返回:
        搜索结果摘要
    """
    # 模拟知识库
    knowledge_base = {
        "退换货": "退货政策：购买后30天内可无条件退货。"
                  "需保留原始包装和购买凭证。退货流程：在线申请→打印退货标签→寄回商品→退款处理(5-7个工作日)",
        "发货": "标准配送：下单后1-2个工作日内发货，3-7个工作日送达。"
                "加急配送：下单当天发货，1-2个工作日送达（需额外付费）",
        "支付": "支持微信支付、支付宝、银行卡、Apple Pay。"
                "分期付款：满500元可分3期（0手续费），满1000元可分6/12期",
    }
    for key, value in knowledge_base.items():
        if key in query:
            return value
    return f"未找到关于 '{query}' 的相关信息。建议尝试不同的关键词。"


@tool
def calculate(expression: str) -> str:
    """
    执行数学计算。支持基本的算术运算。
    
    参数:
        expression: 数学表达式字符串，如 "2+3*4"
    返回:
        计算结果
    """
    try:
        # 安全计算（只允许数字和基本运算符）
        allowed = set("0123456789+-*/(). ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含不允许的字符"
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    from datetime import datetime
    return f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# 工具列表
TOOLS = [search_knowledge_base, calculate, get_current_time]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ============================================================
# 步骤3: 定义节点函数
# ============================================================
def agent_node(state: AgentState) -> dict:
    """
    Agent 节点 —— 工作流的大脑。
    
    职责:
    1. 将当前对话历史 + 用户查询发给 LLM
    2. LLM 决定: 回复用户 OR 调用工具
    3. 如果 LLM 想调用工具 → 生成 ToolMessage
    4. 更新 next_step 标志
    
    这个节点不实际执行工具——它只负责"决策"。
    工具执行由 tools_node 完成。
    """
    # 初始化 LLM（绑定工具）
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.0,
        api_key="...",  # 实际中使用环境变量
    )
    # wrap 工具到 LLM
    llm_with_tools = llm.bind_tools(TOOLS)
    
    # 准备 messages
    # 如果 messages 中没有 system message，在前面加上
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        system_msg = SystemMessage(content="""你是一个有用的 AI 客服助手。
你可以:
1. 使用 search_knowledge_base 查询产品信息、政策、流程
2. 使用 calculate 执行数学计算
3. 使用 get_current_time 获取当前时间

当你不确定时，使用工具查询而不是猜测。
你的回答应简洁、准确、有帮助。""")
        messages = [system_msg] + messages
    
    # 调用 LLM
    response = llm_with_tools.invoke(messages)
    
    # 检查是否有工具调用
    has_tool_calls = (
        hasattr(response, "tool_calls") and
        response.tool_calls and
        len(response.tool_calls) > 0
    )
    
    return {
        "messages": [response],
        "next_step": "tools" if has_tool_calls else "end",
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def tools_node(state: AgentState) -> dict:
    """
    工具执行节点 —— 执行 agent_node 中 LLM 请求的工具调用。
    
    职责:
    1. 从最后一条 AI 消息中提取 tool_calls
    2. 执行对应的工具函数
    3. 将结果封装为 ToolMessage 返回
    
    ToolMessage 必须包含 tool_call_id ——
    这是 LLM 用来关联"请求"和"结果"的关键字段。
    没有正确的 tool_call_id，LLM 无法理解这个结果对应哪个调用。
    """
    # 获取最后一条 AI 消息
    last_message = state["messages"][-1]
    
    # 提取 tool_calls
    tool_calls = last_message.tool_calls
    
    # 执行每个工具调用
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        print(f"  🔧 执行工具: {tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
        
        # 查找并执行工具
        tool_fn = TOOLS_BY_NAME.get(tool_name)
        if tool_fn:
            try:
                result = tool_fn.invoke(tool_args)
                print(f"  ✅ 结果: {result[:100]}...")
            except Exception as e:
                result = f"工具执行错误: {e}"
                print(f"  ❌ 错误: {e}")
        else:
            result = f"未知工具: {tool_name}"
            print(f"  ❌ {result}")
        
        # 创建 ToolMessage
        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )
    
    return {
        "messages": tool_messages,
        "next_step": "agent",  # 工具结果返回给 Agent 继续思考
    }


# ============================================================
# 步骤4: 定义路由函数
# ============================================================
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    路由函数 —— 决定 agent_node 之后的去向。
    
    读取 next_step 标志位来决定。
    也可以用其他条件（如 iteration_count > 某个阈值 → 强制结束）
    """
    # 安全机制：防止无限循环
    if state.get("iteration_count", 0) > 20:
        print("⚠ 达到最大迭代次数，强制结束")
        return "end"
    
    next_step = state.get("next_step", "end")
    if next_step == "tools":
        return "tools"
    else:
        return "end"


# ============================================================
# 步骤5: 构建和编译图
# ============================================================
def build_react_agent():
    """构建 ReAct Agent 工作流图"""
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    
    # 设置入口点
    graph.set_entry_point("agent")
    
    # 添加条件边：agent 执行后根据 should_continue 决定
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # 需要工具 → 去 tools 节点
            "end": END,        # 不需要工具 → 结束
        }
    )
    
    # tools 执行后总是回到 agent（让 LLM 分析工具结果）
    graph.add_edge("tools", "agent")
    
    # 编译
    return graph.compile()


# ============================================================
# 步骤6: 运行 Agent
# ============================================================
def run_agent(user_query: str):
    """运行 Agent 处理用户查询"""
    app = build_react_agent()
    
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "next_step": "agent",
        "iteration_count": 0,
    }
    
    print(f"\n{'=' * 60}")
    print(f"用户: {user_query}")
    print(f"{'=' * 60}")
    
    # 执行
    result = app.invoke(initial_state)
    
    # 打印对话历史（可视化 Agent 的思考过程）
    print(f"\n=== 执行过程 ===")
    for i, msg in enumerate(result["messages"]):
        if isinstance(msg, HumanMessage):
            print(f"[{i}] 👤 用户: {msg.content}")
        elif isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[{i}] 🤖 Agent: 调用 {tc['name']}"
                          f"({json.dumps(tc['args'], ensure_ascii=False)})")
            else:
                print(f"[{i}] 🤖 Agent: {msg.content}")
        elif isinstance(msg, ToolMessage):
            content_preview = msg.content[:80].replace("\n", " ")
            print(f"[{i}] 🔧 工具结果: {content_preview}...")
        elif isinstance(msg, SystemMessage):
            pass  # 不打印系统消息
    
    # 最终回答
    final_answer = result["messages"][-1].content
    print(f"\n=== 最终回答 ===")
    print(final_answer)
    print(f"总迭代次数: {result['iteration_count']}")


if __name__ == "__main__":
    # 测试 Agent
    run_agent("我想退货，请问流程是什么？")
    print("\n" + "-" * 60)
    run_agent("我买了3件商品，每件299元，总共多少钱？")
```

---

## 4. 条件分支：多路径决策

### 4.1 意图路由 Agent

一个常见的 Agent 模式是"意图路由"：根据用户问题的类型，路由到不同的处理通道。

```python
"""
条件分支实战 —— 意图路由 Agent。

场景: 一个客服系统，用户的问题可能是:
- 退货相关 → 路由到退货处理流程
- 发货查询 → 路由到物流查询流程
- 产品咨询 → 路由到产品咨询流程
- 投诉建议 → 路由到投诉处理流程
- 其他 → 通用回复
"""
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage


class IntentRouterState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    intent: str           # 识别出的意图
    context: dict         # 上下文信息（如订单号）


# ============================================================
# 节点: 意图识别
# ============================================================
def classify_intent(state: IntentRouterState) -> dict:
    """
    识别用户意图。
    
    这是整个工作流的第一道关卡。它决定用户问题应该走哪条处理路径。
    """
    user_message = state["messages"][-1].content
    
    # 调用 LLM 做意图分类
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    
    prompt = f"""分析以下用户消息的意图，返回JSON。

用户消息: "{user_message}"

意图分类（选一个）:
- return_refund: 退换货相关（退货、换货、退款）
- shipping: 物流/发货相关（物流查询、发货时间、配送）
- product_inquiry: 产品咨询（产品功能、价格、规格）
- complaint: 投诉建议
- general: 其他一般性问题

返回格式: {{"intent": "类别", "reason": "简短理由", "key_info": "提取的关键信息"}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    import json
    try:
        result = json.loads(response.content)
        intent = result.get("intent", "general")
        reason = result.get("reason", "")
    except json.JSONDecodeError:
        intent = "general"
        reason = "分类解析失败"

    return {
        "messages": [AIMessage(content=f"[意图识别] {intent}: {reason}")],
        "intent": intent,
    }


# ============================================================
# 路由函数
# ============================================================
def route_by_intent(state: IntentRouterState) -> str:
    """根据意图路由到不同的处理节点"""
    intent = state.get("intent", "general")
    
    route_map = {
        "return_refund": "handle_return",
        "shipping": "handle_shipping",
        "product_inquiry": "handle_product",
        "complaint": "handle_complaint",
        "general": "handle_general",
    }
    
    return route_map.get(intent, "handle_general")


# ============================================================
# 处理节点（各意图的专门处理逻辑）
# ============================================================
def handle_return(state: IntentRouterState) -> dict:
    """退货意图处理"""
    return {
        "messages": [AIMessage(content=(
            "我们支持购买后30天内无条件退货。\n"
            "退货步骤:\n"
            "1. 登录账户 → 我的订单 → 选择要退货的商品\n"
            "2. 填写退货原因并提交申请\n"
            "3. 下载退货标签（可自行打印或到快递点出示）\n"
            "4. 将商品连同包装寄回\n"
            "5. 我们收到商品后5-7个工作日退款到您的支付账户\n\n"
            "需要我帮您发起退货申请吗？"
        ))],
    }


def handle_shipping(state: IntentRouterState) -> dict:
    """物流查询意图处理"""
    return {
        "messages": [AIMessage(content=(
            "关于物流查询:\n"
            "- 标准配送: 1-2个工作日内发货，3-7个工作日送达\n"
            "- 加急配送: 当天发货，1-2个工作日送达\n"
            "- 您可以在'我的订单'中查看实时物流状态\n\n"
            "如果超过预计时间还未收到，请提供订单号，我帮您查询。"
        ))],
    }


def handle_product(state: IntentRouterState) -> dict:
    """产品咨询处理"""
    return {
        "messages": [AIMessage(content=(
            "感谢您对我们的产品感兴趣！\n"
            "请告诉我您想了解的具体产品型号或品类，"
            "我可以为您提供详细的产品规格、价格和用户评价。\n\n"
            "您也可以直接浏览我们的官网查看所有产品。"
        ))],
    }


def handle_complaint(state: IntentRouterState) -> dict:
    """投诉建议处理"""
    return {
        "messages": [AIMessage(content=(
            "非常抱歉让您有不好的体验。这绝对不是我们想要的服务标准。\n\n"
            "我已经记录了您的反馈，并会立即转发给相关部门处理。\n"
            "通常情况下，我们会在24小时内给您回复。\n\n"
            "如果问题紧急，您也可以直接拨打我们的客服热线: 400-xxx-xxxx。"
        ))],
    }


def handle_general(state: IntentRouterState) -> dict:
    """通用问题处理"""
    return {
        "messages": [AIMessage(content=(
            "您好！我是智能客服助手，可以帮您解决以下问题:\n"
            "- 退换货流程和政策\n"
            "- 物流查询和配送问题\n"
            "- 产品咨询和购买指导\n"
            "- 投诉建议\n\n"
            "请告诉我您遇到了什么问题？"
        ))],
    }


# ============================================================
# 构建图
# ============================================================
if __name__ == "__main__":
    graph = StateGraph(IntentRouterState)

    # 添加所有节点
    graph.add_node("classify", classify_intent)
    graph.add_node("handle_return", handle_return)
    graph.add_node("handle_shipping", handle_shipping)
    graph.add_node("handle_product", handle_product)
    graph.add_node("handle_complaint", handle_complaint)
    graph.add_node("handle_general", handle_general)

    graph.set_entry_point("classify")

    # 条件边: 分类后根据意图路由
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "handle_return": "handle_return",
            "handle_shipping": "handle_shipping",
            "handle_product": "handle_product",
            "handle_complaint": "handle_complaint",
            "handle_general": "handle_general",
        }
    )

    # 所有处理节点执行完都结束
    for handler in ["handle_return", "handle_shipping", "handle_product",
                    "handle_complaint", "handle_general"]:
        graph.add_edge(handler, END)

    app = graph.compile()

    # 测试
    test_queries = [
        "我要退货，刚收到的东西不满意",
        "我的包裹到哪了",
        "你们这款手机多少钱",
        "你们客服态度太差了",
    ]

    for query in test_queries:
        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "intent": "",
            "context": {},
        })
        answer = result["messages"][-1].content
        print(f"\n用户: {query}")
        print(f"意图: {result['intent']}")
        print(f"回复: {answer[:100]}...")
```

---

## 5. 循环流程

### 5.1 Agent 循环与终止条件

第3节中的 ReAct Agent 已经展示了基本的循环模式：Agent 调用工具 → 获取结果 → 再思考 → 可能再调用工具。这里深入讨论循环的控制和终止。

```python
"""
循环流程的深入理解 —— 终止条件与循环控制。

循环是 Agent 工作流的核心机制，但必须小心控制：
- 没有终止条件 → 无限循环 → Token 耗尽
- 终止太早 → Agent 没有充分思考 → 输出质量差
- 终止太晚 → 不必要的 Token 开销
"""
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage


class LoopState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    iteration: int
    quality_score: float  # 输出质量评分
    status: str


# ============================================================
# 循环终止的条件类型
# ============================================================

# 条件1: 最大迭代次数（硬限制——最基础的保护）
MAX_ITERATIONS = 5

# 条件2: LLM 自己决定停止（当 LLM 返回的消息没有 tool_calls 时）
# 条件3: 质量达标（输出质量评分 > 阈值时停止）
# 条件4: Token 预算耗尽（计算当前使用的总 token 数）


def should_continue_loop(state: LoopState) -> Literal["continue", "end"]:
    """
    综合判断循环是否应该继续。
    
    优先级: 安全限制 > Token 预算 > 质量达标 > LLM 意愿
    """
    # 1. 硬限: 最大迭代次数
    if state["iteration"] >= MAX_ITERATIONS:
        print(f"  [Router] 达到最大迭代次数 ({MAX_ITERATIONS})，强制终止")
        return "end"
    
    # 2. 质量达标（可选条件）
    if state.get("quality_score", 0) >= 0.95:
        print(f"  [Router] 质量达标 ({state['quality_score']:.2f})，终止")
        return "end"
    
    # 3. LLM 的意愿（从最后一条 AI 消息判断）
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        has_tool_calls = (
            hasattr(last_msg, "tool_calls") and
            last_msg.tool_calls and
            len(last_msg.tool_calls) > 0
        )
        if not has_tool_calls:
            print("  [Router] LLM 决定终止（无工具调用）")
            return "end"
        else:
            print("  [Router] LLM 仍需工具，继续循环")
            return "continue"
    
    return "end"


# ============================================================
# 循环中的状态累积与清理
# ============================================================
def demonstrate_state_accumulation():
    """
    演示: 循环中 State 的累积行为。
    
    messages 使用 operator.add → 每次循环都追加，不会丢失历史。
    这是 Agent 的"记忆"——它能看到之前所有的交互。
    
    iteration 使用替换策略 → 每个节点返回新值覆盖旧值。
    """
    print("=== 循环中的状态累积行为 ===\n")
    print("messages (Annotated[list, operator.add]):")
    print("  每次返回的消息被追加到列表末尾")
    print("  → 对话历史自动累积，Agent 有完整记忆\n")
    print("iteration (int, 无 Annotated):")
    print("  每次返回的新值替换旧值")
    print("  → 适合计数器、状态标志等不累积的字段\n")
    print("关键提醒:")
    print("  - 循环中 messages 会不断增长，注意 token 消耗")
    print("  - 长对话可能需要"消息裁剪"策略（只保留最近 N 条）")
    print("  - 可以使用 summary 字段做对话摘要，减少 token 消耗")


if __name__ == "__main__":
    demonstrate_state_accumulation()
```

---

## 6. Human-in-the-Loop（人机交互）

### 6.1 为什么需要人机交互

Agent 不是全能的。在以下场景中，我们需要在 Agent 工作流中插入人类决策点：

- **关键操作审批**：发送真实邮件、执行数据库写入、调用付费 API 等操作需要人类确认
- **质量把关**：Agent 生成的代码、报告、翻译等需要人类审查后才能交付
- **模糊决策澄清**：Agent 遇到歧义时，与其猜错，不如问人类
- **合规审查**：某些行业（金融、医疗、法律）要求特定操作必须有人类参与

LangGraph 提供了 `interrupt` 机制来支持 Human-in-the-Loop。在节点中调用 `interrupt()` 会暂停工作流，直到人类提供输入。

```python
"""
Human-in-the-Loop —— 在 Agent 工作流中插入人类决策点。

LangGraph 实现 Human-in-the-Loop 的方式:
1. 在节点中，遇到需要人类决策时，将状态保存并暂停
2. 外部系统（如 Web UI、命令行）展示当前状态给人类
3. 人类做出决策后，将决策作为输入恢复工作流执行

对于 LangGraph >= 0.2.x:
- 使用 graph.compile(interrupt_before=["node_name"]) 在指定节点前暂停
- 使用 graph.compile(interrupt_after=["node_name"]) 在指定节点后暂停
- app.invoke(state) → 暂停 → 返回当前状态
- app.invoke(Command(resume=human_input)) → 恢复执行
"""
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


class ApprovalState(TypedDict):
    """含审批流程的工作流状态"""
    messages: Annotated[list[BaseMessage], operator.add]
    pending_action: str           # 等待审批的操作描述
    approval_status: str          # "pending" / "approved" / "rejected"
    final_result: str


# ============================================================
# 节点: Agent 执行任务
# ============================================================
def execute_task(state: ApprovalState) -> dict:
    """Agent 执行任务，但某些操作需要审批"""
    # 模拟 Agent 完成了某个需要审批的操作
    action_description = "发送营销邮件给 10000 名用户（内容摘要: 618大促通知）"
    
    return {
        "messages": [AIMessage(content=f"任务执行中。发现需要审批的操作: {action_description}")],
        "pending_action": action_description,
        "approval_status": "pending",
    }


# ============================================================
# 节点: 等待人类审批 (通过 interrupt)
# ============================================================
def request_approval(state: ApprovalState) -> dict:
    """
    请求人类审批节点。
    
    调用 interrupt() 暂停工作流。
    暂停后，外部系统可以读取 state 中的 pending_action，
    展示给人类，收集人类的审批决定。
    """
    # interrupt() 暂停工作流并等待恢复
    # 参数是人类需要看到的信息
    human_decision = interrupt({
        "question": "请审批以下操作:",
        "action": state["pending_action"],
        "options": ["approve", "reject", "modify"],
        "risk_level": "medium",  # 可帮助人类快速判断
    })
    
    # human_decision 是人类恢复执行时传入的值
    if human_decision == "approve":
        return {
            "messages": [AIMessage(content="✅ 操作已获批准，继续执行...")],
            "approval_status": "approved",
        }
    elif human_decision == "reject":
        return {
            "messages": [AIMessage(content="❌ 操作被拒绝，跳过此步骤。")],
            "approval_status": "rejected",
        }
    else:  # modify
        return {
            "messages": [AIMessage(content="📝 操作需修改，请提供修改意见。")],
            "approval_status": "pending",  # 保持 pending 等待修改
        }


# ============================================================
# 节点: 根据审批结果执行后续操作
# ============================================================
def handle_approval_result(state: ApprovalState) -> dict:
    """根据审批结果执行不同的后续操作"""
    if state["approval_status"] == "approved":
        # 模拟执行操作
        result = "邮件发送成功: 已发送 10000 封，打开率预计 25%"
        return {
            "messages": [AIMessage(content=result)],
            "final_result": result,
        }
    else:
        result = "操作已取消。Agent 将寻找替代方案。"
        return {
            "messages": [AIMessage(content=result)],
            "final_result": result,
        }


# ============================================================
# 构建带审批的工作流
# ============================================================
if __name__ == "__main__":
    print("=== Human-in-the-Loop 工作流演示 ===\n")
    
    graph = StateGraph(ApprovalState)
    
    graph.add_node("execute", execute_task)
    graph.add_node("request_approval", request_approval)
    graph.add_node("handle_result", handle_approval_result)
    
    graph.set_entry_point("execute")
    graph.add_edge("execute", "request_approval")
    graph.add_edge("request_approval", "handle_result")
    graph.add_edge("handle_result", END)
    
    # 编译时指定在 request_approval 节点前中断
    app = graph.compile()
    
    print("工作流启动...")
    print("第一段执行后会在 request_approval 节点调用 interrupt() 暂停")
    print("人类审批后，传入审批结果恢复执行\n")
    
    # 注: 此处为代码说明，实际运行需要交互式环境
    print("交互流程:")
    print("  1. app.invoke(initial_state) → 执行到 request_approval → 暂停")
    print("  2. 展示待审批操作给人类")
    print("  3. 人类选择 'approve' / 'reject' / 'modify'")
    print("  4. app.invoke(Command(resume='approve')) → 恢复执行")
    print("  5. Agent 根据审批结果完成后续操作")
```

---

## 7. 子图嵌套

### 7.1 将复杂子流程封装为子图

当一个工作流变得很大（20+ 个节点）时，直接管理所有节点和边会变得困难。子图允许你将一个子流程封装成一个"复合节点"——对外看起来像一个节点，内部是一个完整的状态图。

```python
"""
子图嵌套 —— 将复杂子流程封装为可复用的组件。

适用场景:
- 一个"搜索-分析-总结"子流程在多个工作流中用到
- 工作流中某个阶段非常复杂（20+ 节点），需要单独管理
- 希望团队分工：不同人负责不同的子图

子图的关键:
- 子图有自己的 State（可以是父 State 的子集或转换）
- 父图调用子图就像调用一个节点
"""
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END


# ============================================================
# 父图 State
# ============================================================
class MainState(TypedDict):
    user_query: str
    research_result: str
    analysis_result: str
    final_report: str
    # 注意：不需要在父 State 中包含子图的内部 State


# ============================================================
# 子图: "研究分析"子流程
# ============================================================
class ResearchSubState(TypedDict):
    """研究子图的内部 State（父图不需要知道这些字段）"""
    query: str
    search_results: str
    analysis: str
    summary: str


def search_node(state: ResearchSubState) -> dict:
    """子图节点: 搜索"""
    # 模拟搜索
    return {
        "search_results": f"关于 '{state['query']}' 的搜索结果: ...（模拟数据）"
    }


def analyze_node(state: ResearchSubState) -> dict:
    """子图节点: 分析"""
    return {
        "analysis": f"分析: {state['search_results'][:50]}... → 发现3个关键洞察"
    }


def summarize_node(state: ResearchSubState) -> dict:
    """子图节点: 总结"""
    return {
        "summary": f"研究总结: {state['analysis'][:50]}... → 建议下一步行动"
    }


def build_research_subgraph():
    """构建研究子图"""
    subgraph = StateGraph(ResearchSubState)
    
    subgraph.add_node("search", search_node)
    subgraph.add_node("analyze", analyze_node)
    subgraph.add_node("summarize", summarize_node)
    
    subgraph.set_entry_point("search")
    subgraph.add_edge("search", "analyze")
    subgraph.add_edge("analyze", "summarize")
    subgraph.add_edge("summarize", END)
    
    return subgraph.compile()


# ============================================================
# 父图节点
# ============================================================
def prepare_research(state: MainState) -> dict:
    """
    准备研究 —— 将父图 State 映射为子图输入。
    
    这个节点的作用是"适配器"——将父图的数据格式
    转换为子图期望的格式。
    """
    # 返回的字典中，子图需要的字段用子图格式
    # （在这里实际代码中会调用子图）
    return {}


def run_research(state: MainState) -> dict:
    """
    运行研究子图。
    
    这里的简化实现直接模拟子图执行。
    在生产代码中，你会调用:
        research_subgraph.invoke({"query": state["user_query"], ...})
    """
    query = state["user_query"]
    # 模拟子图执行
    research_output = f"研究完成: 针对'{query}'进行了深入搜索和分析"
    return {
        "research_result": research_output,
    }


def generate_report(state: MainState) -> dict:
    """基于研究结果生成最终报告"""
    return {
        "final_report": f"最终报告\n{'='*40}\n{state['research_result']}\n\n结论: ..."
    }


# ============================================================
# 构建父图
# ============================================================
if __name__ == "__main__":
    print("=== 子图嵌套演示 ===\n")
    
    graph = StateGraph(MainState)
    
    graph.add_node("prepare", prepare_research)
    graph.add_node("research", run_research)  # 这是一个子图
    graph.add_node("report", generate_report)
    
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "research")
    graph.add_edge("research", "report")
    graph.add_edge("report", END)
    
    app = graph.compile()
    
    result = app.invoke({
        "user_query": "分析2024年AI Agent市场趋势",
        "research_result": "",
        "analysis_result": "",
        "final_report": "",
    })
    
    print(result["final_report"])
    print("\n子图嵌套的优势:")
    print("  1. 研究子图可以独立开发、测试、复用")
    print("  2. 父图只关心"输入query → 输出research_result"")
    print("  3. 修改子图内部不影响父图")
    print("  4. 同一个子图可以在不同父图中使用")
```

---

## 8. 完整实战：多功能的 Agent 工作流

现在整合本章所有概念，构建一个具备以下能力的完整 Agent 工作流：

- ReAct 循环（工具调用）
- 意图路由（条件分支）
- 循环安全控制
- Human-in-the-Loop 审批

```python
"""
完整实战 —— 企业级 Agent 工作流。

整合: ReAct循环 + 意图路由 + 循环控制 + 人工审批

架构图:

                         ┌──────────┐
                         │  START   │
                         └────┬─────┘
                              │
                              ▼
                         ┌──────────┐
                         │ 意图分类  │
                         └────┬─────┘
                              │ 条件边: route_by_intent
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         ┌─────────┐    ┌─────────┐    ┌─────────┐
         │ 退货流程 │    │ 物流流程 │    │ 通用流程 │
         └─────────┘    └─────────┘    └────┬────┘
                                            │
                                            ▼
                                       ┌─────────┐
                                       │ Agent   │ ← ReAct 循环入口
                                       │ 思考    │
                                       └────┬────┘
                                            │ should_continue
                                       ┌────┴────┐
                                       ▼         ▼
                                  ┌────────┐ ┌──────────┐
                                  │调工具   │ │ 审批检查  │
                                  └───┬────┘ └────┬─────┘
                                      │           │ needs_approval?
                                      │      ┌────┴────┐
                                      │      ▼         ▼
                                      │  ┌───────┐ ┌──────┐
                                      │  │等审批  │ │ END  │
                                      │  └───┬───┘ └──────┘
                                      │      │
                                      └──────┘ (回到 Agent)
"""
from typing import TypedDict, Annotated, Literal
import operator
import json
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
)
from langchain_core.tools import tool


# ============================================================
# State 定义
# ============================================================
class WorkflowState(TypedDict):
    """
    企业级 Agent 工作流的状态。
    
    字段规划原则:
    - 对话历史: messages (Annotated, 追加)
    - 控制标志: intent, status, approval_status (替换)
    - 计数器: iteration (替换)
    - 业务数据: pending_action, final_result (替换)
    """
    messages: Annotated[list[BaseMessage], operator.add]
    intent: str                     # 识别的意图
    status: str                     # 工作流状态: "classifying", "executing", "waiting_approval", "done"
    iteration: int                  # ReAct 循环计数
    approval_status: str            # 审批状态: "none", "pending", "approved", "rejected"
    pending_action: str             # 等待审批的操作描述
    final_result: str               # 最终结果


# ============================================================
# 工具定义
# ============================================================
@tool
def query_database(query: str) -> str:
    """查询客户数据库。支持按订单号、客户ID、日期范围查询。"""
    mock_data = {
        "ORD-12345": "订单ORD-12345: 商品iPhone 15 Pro, 下单日期2024-05-20, 状态:已签收",
        "ORD-67890": "订单ORD-67890: 商品MacBook Air, 下单日期2024-05-22, 状态:运输中",
    }
    for key, value in mock_data.items():
        if key in query:
            return value
    return f"未找到订单 '{query}'"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    发送邮件（模拟）。注意: 此操作需要审批。
    """
    return f"[模拟] 邮件已发送 → To: {to}, Subject: {subject}"


@tool
def calculate_refund(order_amount: float, days_since_purchase: int) -> str:
    """计算退款金额。根据购买天数计算折旧后退款。"""
    if days_since_purchase <= 7:
        refund_rate = 1.0
    elif days_since_purchase <= 15:
        refund_rate = 0.9
    elif days_since_purchase <= 30:
        refund_rate = 0.8
    else:
        return "超出30天退货期限，无法退款"
    
    refund_amount = order_amount * refund_rate
    return (f"退款计算: 订单金额 {order_amount}元, "
            f"购买 {days_since_purchase} 天, "
            f"折旧率 {(1-refund_rate)*100:.0f}%, "
            f"退款金额 {refund_amount:.2f}元")


TOOLS = [query_database, send_email, calculate_refund]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ============================================================
# 节点实现
# ============================================================
def classify_intent(state: WorkflowState) -> dict:
    """节点1: 意图分类"""
    user_msg = state["messages"][-1].content
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    response = llm.invoke([
        SystemMessage(content="分类用户意图: return/shipping/product/complaint/general, 返回JSON"),
        HumanMessage(content=user_msg),
    ])
    
    try:
        result = json.loads(response.content)
        intent = result.get("intent", "general")
    except json.JSONDecodeError:
        intent = "general"
    
    return {
        "intent": intent,
        "status": "executing",
        "messages": [AIMessage(content=f"[系统] 识别意图: {intent}")],
    }


def agent_node(state: WorkflowState) -> dict:
    """节点2: Agent 思考（ReAct 循环的核心节点）"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    llm_with_tools = llm.bind_tools(TOOLS)
    
    system = SystemMessage(content="""你是专业的客服 Agent。
你可以使用工具查询订单、计算退款、发送邮件。
发送邮件前必须经过人类审批。
遇到不确定的情况，使用工具查询而不是猜测。""")
    
    messages = [system] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    
    has_tools = (
        hasattr(response, "tool_calls") and
        response.tool_calls and
        len(response.tool_calls) > 0
    )
    
    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
    }


def tools_node(state: WorkflowState) -> dict:
    """节点3: 工具执行"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    
    tool_messages = []
    for tc in tool_calls:
        tool_fn = TOOLS_BY_NAME.get(tc["name"])
        if tool_fn:
            try:
                result = tool_fn.invoke(tc["args"])
            except Exception as e:
                result = f"错误: {e}"
        else:
            result = f"未知工具: {tc['name']}"
        
        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tc["id"],
        ))
    
    return {"messages": tool_messages}


def approval_check(state: WorkflowState) -> dict:
    """节点4: 审批检查 —— 在需要审批的操作前暂停"""
    # 检查是否需要审批
    last_msg = state["messages"][-1]
    needs_approval = False
    action_desc = ""
    
    if isinstance(last_msg, AIMessage):
        content = last_msg.content.lower() if last_msg.content else ""
        if "send_email" in content or "发送邮件" in content:
            needs_approval = True
            action_desc = "发送邮件"
    
    if needs_approval:
        return {
            "status": "waiting_approval",
            "approval_status": "pending",
            "pending_action": action_desc,
            "messages": [AIMessage(content=f"⏸ [系统] 操作'{action_desc}'需要审批，已暂停")],
        }
    else:
        return {"approval_status": "none"}


def wait_for_approval(state: WorkflowState) -> dict:
    """节点5: 等待人类审批"""
    decision = interrupt({
        "message": f"请审批操作: {state['pending_action']}",
        "context": f"Agent 即将执行: {state['pending_action']}",
        "options": ["approve", "reject"],
    })
    
    if decision == "approve":
        return {
            "approval_status": "approved",
            "status": "executing",
            "messages": [AIMessage(content="✅ 操作已批准")],
        }
    else:
        return {
            "approval_status": "rejected",
            "status": "executing",
            "messages": [AIMessage(content="❌ 操作被拒绝，请修改方案")],
        }


# ============================================================
# 路由函数
# ============================================================
def route_by_intent(state: WorkflowState) -> str:
    intent = state.get("intent", "general")
    # 简单意图直接路由到 agent（所有意图都走 ReAct 循环处理）
    return "agent"


def route_after_agent(state: WorkflowState) -> Literal["tools", "approval_check", "end"]:
    """Agent 思考后的路由"""
    if state.get("iteration", 0) > 15:
        return "end"
    
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage):
        has_tools = (
            hasattr(last_msg, "tool_calls") and
            last_msg.tool_calls
        )
        if has_tools:
            return "tools"
    
    # 没有工具调用 → 检查是否需要审批 → 否则结束
    return "approval_check"


def route_after_approval(state: WorkflowState) -> Literal["wait", "end"]:
    """审批检查后的路由"""
    if state.get("approval_status") == "pending":
        return "wait"
    else:
        return "end"


# ============================================================
# 构建完整工作流
# ============================================================
def build_workflow():
    graph = StateGraph(WorkflowState)
    
    # 添加所有节点
    graph.add_node("classify", classify_intent)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("approval_check", approval_check)
    graph.add_node("wait_approval", wait_for_approval)
    
    # 设置入口
    graph.set_entry_point("classify")
    
    # 意图分类后 → Agent
    graph.add_edge("classify", "agent")
    
    # Agent 思考后 → 工具 / 审批检查 / 结束
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "approval_check": "approval_check",
            "end": END,
        }
    )
    
    # 工具执行后 → 回到 Agent 继续思考
    graph.add_edge("tools", "agent")
    
    # 审批检查后 → 等待审批 / 结束
    graph.add_conditional_edges(
        "approval_check",
        route_after_approval,
        {
            "wait": "wait_approval",
            "end": END,
        }
    )
    
    # 审批后 → 回到 Agent 继续执行
    graph.add_edge("wait_approval", "agent")
    
    return graph.compile()


# ============================================================
# 运行演示
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("企业级 Agent 工作流")
    print("=" * 60)
    print("""
    功能:
    ✅ ReAct 循环（工具调用 + 反思）
    ✅ 意图路由（根据用户问题类型智能分流）
    ✅ 循环安全控制（最大 15 轮迭代）
    ✅ Human-in-the-Loop（关键操作需人类审批）
    ✅ 状态追踪（完整的执行日志）
    
    架构: 5 个节点 + 3 条条件边
    """)
    
    app = build_workflow()
    print("✅ 工作流编译成功")
```

---

## 9. 流式状态输出（概念与用法）

LangGraph 支持流式输出，让外部系统可以实时观察 Agent 的执行过程。这对于构建可观察的 Agent 系统至关重要。

```python
"""
流式状态输出 —— 实时观察 Agent 工作流的每一步。

LangGraph 的流式输出允许外部系统（Web UI、CLI、监控系统）:
- 看到 Agent 当前执行到了哪个节点
- 看到每一步后的 State 变化
- 在 Human-in-the-Loop 场景中及时展示审批请求

流式模式:
- app.stream(state) → 同步流，每次 yield 一个节点更新
- app.astream(state) → 异步流，每次 yield 一个节点更新
"""
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, END
import asyncio


class StreamState(TypedDict):
    counter: int
    logs: Annotated[list[str], operator.add]


async def demo_streaming():
    """演示流式输出（概念代码，没有完整图定义）"""
    print("=== 流式输出概念演示 ===\n")
    
    print("同步流式输出 (stream):")
    print("""
    for event in app.stream(initial_state):
        # event = {node_name: state_update}
        node_name = list(event.keys())[0]
        state_update = event[node_name]
        print(f"[{node_name}] {state_update}")
    
    输出示例:
    [agent] {'messages': [AIMessage(...)], 'next_step': 'tools'}
    [tools] {'messages': [ToolMessage(...)], 'next_step': 'agent'}
    [agent] {'messages': [AIMessage('最终回答...')], 'next_step': 'end'}
    """)
    
    print("\n异步流式输出 (astream):")
    print("""
    async for event in app.astream(initial_state):
        node_name = list(event.keys())[0]
        print(f"Async [{node_name}]")
    
    应用场景:
    - WebSocket 推送实时状态到前端
    - SSE (Server-Sent Events) 流式返回给用户
    - 实时日志输出和监控
    """)
    
    print("\n流式输出 + Human-in-the-Loop:")
    print("""
    async for event in app.astream(initial_state):
        if "wait_approval" in event:
            # 发送审批请求到前端
            approval_data = event["wait_approval"]["pending_action"]
            send_to_frontend({"type": "approval_required", "data": approval_data})
            
            # 等待人类输入后使用 Command(resume=...) 恢复
        else:
            # 更新前端显示当前进度
            send_to_frontend({"type": "progress", "data": event})
    """)


if __name__ == "__main__":
    asyncio.run(demo_streaming())
```

---

## 基础练习

### 练习 1: 构建一个基本的状态图
**场景**: 使用 LangGraph 构建一个简单的多步骤工作流（如 3 步数据处理流水线）。
**要求**:
- 定义 State（TypedDict，包含至少 3 个字段）
- 定义至少 3 个节点函数
- 使用普通边连接节点
- 使用条件边实现至少一个分支
- 验证最终 State 的正确性
**文件**: `exercise/ai-agent/ch07_LangGraph工作流引擎/ex1_basic_state_graph.py`

### 练习 2: 实现 ReAct Agent 工作流
**场景**: 用 LangGraph 重新实现第03章学过的 ReAct Agent。
**要求**:
- 定义 AgentState（messages, next_step, iteration_count）
- 实现 agent_node（LLM 决策 + Function Calling）
- 实现 tools_node（工具执行）
- 实现 should_continue 路由函数（包含循环次数限制）
- 至少注册 3 个工具
- 验证 Agent 能正确地进行多轮工具调用
**文件**: `exercise/ai-agent/ch07_LangGraph工作流引擎/ex2_react_agent.py`

### 练习 3: 实现 Human-in-the-Loop 审批流程
**场景**: 在工作流中插入人类审批节点，对关键操作进行把关。
**要求**:
- 设计一个包含"自动化执行"和"人工审批"的工作流
- 使用 interrupt() 暂停工作流
- 支持 approve / reject 两种审批结果
- 根据审批结果走不同的后续路径
- 实现 Command(resume=...) 恢复机制
**文件**: `exercise/ai-agent/ch07_LangGraph工作流引擎/ex3_human_in_the_loop.py`

## 进阶练习

### 练习 4: 实现一个多意图路由 Agent
**场景**: 构建一个能根据用户意图自动路由到不同处理通道的 Agent。
**要求**:
- 实现意图分类节点（至少 4 种意图）
- 每种意图有独立的处理子流程
- 某些处理子流程内部包含 ReAct 循环（调用工具）
- 使用子图封装复杂的子流程
- 实现完整的循环安全控制（最大迭代次数 + 超时 + Token 预算）
**文件**: `exercise/ai-agent/ch07_LangGraph工作流引擎/ex4_intent_router.py`

### 练习 5: 构建一个完整的客服 Agent 工作流
**场景**: 整合本章所有概念，构建企业级客服 Agent 工作流。
**要求**:
- 意图路由（退换货/物流/产品咨询/投诉/通用）
- ReAct 工具调用循环（至少 4 个工具）
- Human-in-the-Loop 审批关键操作
- 子图封装（如"退款计算+审批"子流程）
- 循环安全机制（迭代上限 + Token 预算监控）
- 流式状态输出（使用 astream）
- 完整的执行日志
**文件**: `exercise/ai-agent/ch07_LangGraph工作流引擎/ex5_customer_service_agent.py`

---

## 常见错误

### 错误 1: 忘记在条件边的路由映射中包含 END

```python
# 错误: 路由函数可能返回 END，但路由映射中没有
graph.add_conditional_edges(
    "agent",
    router_fn,
    {"tools": "tools", "end": "end_node"}  # ← END 不在映射中！
)

# 正确:
graph.add_conditional_edges(
    "agent",
    router_fn,
    {"tools": "tools", END: END}  # ← 必须显式包含 END
)
```

### 错误 2: State 字段的 Reducer 选错

```python
# 错误: messages 字段没加 Annotated
class BadState(TypedDict):
    messages: list[str]  # ← 默认替换策略！每次节点返回都会覆盖

# 正确: 使用 operator.add 进行追加
class GoodState(TypedDict):
    messages: Annotated[list[str], operator.add]
```

### 错误 3: 节点返回的键不在 State 中

```python
def bad_node(state: MyState) -> dict:
    return {"typo_field": "value"}  # ← State 中没有 typo_field！

# LangGraph 在 0.2.x 中会静默忽略，在 0.3.x 中会报错
# 正确: 只返回 State 中定义的字段
```

### 错误 4: 循环工作流缺少终止条件

```python
# 错误: Agent 节点永远不设置结束条件
def router(state):
    return "tools"  # ← 永远返回 tools → 无限循环！

# 正确: 包含结束条件
def router(state):
    if state["iteration"] > 10:
        return END
    if state["status"] == "done":
        return END
    return "tools"
```

### 错误 5: 在 interrupt() 后使用错误的恢复方式

```python
# 错误: 直接再次 invoke
# result = app.invoke({...})  # 这不会恢复暂停的工作流！

# 正确: 使用 Command(resume=...)
# result = app.invoke(Command(resume={"decision": "approve"}))
```

### 错误 6: ToolMessage 缺少 tool_call_id

```python
# 错误: ToolMessage 没有 tool_call_id
ToolMessage(content="result")  # LLM 无法关联这个结果

# 正确:
ToolMessage(content="result", tool_call_id=tool_call["id"])
```

### 错误 7: 过度嵌套子图导致调试困难

```python
# 反模式: 三层嵌套子图
# MainGraph → SubGraph A → SubSubGraph B → SubSubSubGraph C
# 每个子图有自己的 State 和逻辑 → 调试时需要在多个层次间跳转

# 建议: 最多两层嵌套，优先扁平化设计
```

### 错误 8: 在异步环境中使用同步节点

```python
# 错误: 在作为 async 调用时，所有节点都应该是 async def
# 同步节点在异步上下文中会阻塞事件循环

# 解决: 对于 I/O 密集型节点，使用 async def
# 对于纯计算节点，可以保持同步（LangGraph 会在线程池中执行）
```

---

## 本章小结

本章深入学习了 LangGraph 工作流引擎，掌握了如何用状态图来建模和执行复杂的 Agent 逻辑：

| 知识点 | 核心要点 |
|--------|----------|
| 工作流引擎动机 | 线性调用链无法表达分支/循环/Human-in-the-Loop，需要图模型 |
| StateGraph | LangGraph 的入口类，定义工作流骨架；编译后得到可执行应用 |
| Node | 执行具体逻辑的 Python 函数，输入 State 返回部分更新 |
| State | 节点间流转的数据，TypedDict 定义，Annotated 控制合并策略 |
| Edge | 普通边（无条件转移）+ 条件边（根据 State 路由）+ END（终止） |
| 条件边 | add_conditional_edges + 路由函数 + 路由映射 = 分支决策 |
| 循环 | Agent → tools → Agent → ... 直到满足终止条件 |
| 循环安全 | max_iterations + iteration_count + Token 预算监控 |
| Human-in-the-Loop | interrupt() 暂停，Command(resume=...) 恢复 |
| 子图 | 将复杂子流程封装为独立状态图，父图当作节点调用 |
| 流式输出 | stream/astream 实时观察每一步的状态变化 |
| ReAct Agent | agent_node(LLM决策) ↔ tools_node(工具执行) 循环 |
| 意图路由 | 分类节点 + 条件边 → 不同处理通道 |

下一章将学习 Agent 评估与安全——当 Agent 变得越来越强大，我们如何确保它做的事是正确的、安全的、可信任的。
