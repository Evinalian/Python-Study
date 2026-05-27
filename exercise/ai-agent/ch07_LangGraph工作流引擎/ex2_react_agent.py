"""
章节：第07章 LangGraph 工作流引擎
题目：用 LangGraph 实现 ReAct Agent
类型：基础练习

题目描述：
使用 LangGraph 重新实现第03章学过的 ReAct Agent。将 ReAct 循环（Thought → Action → Observation → Thought → ...)
建模为 LangGraph 的状态图，理解图模型如何优雅地表达循环逻辑。

要求：
1. 定义 AgentState：
   - messages: Annotated[list[BaseMessage], operator.add]（LangChain 消息历史）
   - iteration_count: int（循环计数器）
   - status: str（工作流状态）
2. 实现 agent_node：
   - 从 messages 中提取最新用户消息
   - 调用 LLM 的 bind_tools() 接口
   - 如果 LLM 返回 tool_calls → 标记 status="tools"
   - 如果 LLM 返回普通文本 → 标记 status="end"
3. 实现 tools_node：
   - 从 messages[-1] 中提取 tool_calls
   - 执行对应的工具函数
   - 返回 ToolMessage 列表
   - 注意 tool_call_id 必须正确匹配
4. 实现路由函数 should_continue：
   - 根据 status 决定去 tools 还是 END
   - 包含循环次数上限保护
5. 至少注册 3 个 LangChain Tool：
   建议: search_info, calculate, get_current_time
6. 测试多轮工具调用场景（用户问题需要调用 2+ 次工具）

提示：
- 使用 LangChain 的 BaseMessage/HumanMessage/AIMessage/ToolMessage
- Tool 使用 @tool 装饰器定义
- LLM 使用 ChatOpenAI(model="gpt-4o").bind_tools(TOOLS)
- ToolMessage 的 tool_call_id 必须从 tool_call["id"] 获取
- messages 使用 operator.add 确保对话历史完整保留
"""

from typing import TypedDict, Annotated, Literal
import operator
import json

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：导入依赖
# - from langgraph.graph import StateGraph, END
# - from langchain_openai import ChatOpenAI
# - from langchain_core.messages import (HumanMessage, AIMessage, ToolMessage, BaseMessage)
# - from langchain_core.tools import tool
#
# 第2步：定义工具（至少 3 个）
# - search_knowledge_base(query): 模拟搜索知识库
# - calculate(expression): 安全计算数学表达式
# - get_current_datetime(): 获取当前时间
# - 使用 @tool 装饰器
# - 每个工具的 docstring 要完整（LLM 会根据 docstring 决定调用哪个工具）
#
# 第3步：定义 AgentState
# - messages: Annotated[list[BaseMessage], operator.add]
# - iteration_count: int
# - status: str
#
# 第4步：实现 agent_node
# - 创建 ChatOpenAI 实例
# - 使用 bind_tools(TOOLS) 绑定工具
# - 构建 system prompt（Identity + 工具使用指导）
# - 调用 llm_with_tools.invoke(messages)
# - 检查 response.tool_calls 是否存在
# - 返回: messages=[response], status="tools"/"end", iteration_count+=1
#
# 第5步：实现 tools_node
# - 获取 messages[-1].tool_calls
# - 遍历 tool_calls:
#   * 根据 tool_call["name"] 查找工具
#   * 调用 tool.invoke(tool_call["args"])
#   * 创建 ToolMessage(content=str(result), tool_call_id=tool_call["id"])
# - 返回: messages=tool_messages列表
#
# 第6步：实现 should_continue 路由函数
# - 检查 status == "tools" → return "tools"
# - 检查 iteration_count > 15 → return END
# - 否则 → return END
# - 注意: 返回值必须与 add_conditional_edges 的映射一致
#
# 第7步：构建和编译图
# - graph.add_node("agent", agent_node)
# - graph.add_node("tools", tools_node)
# - graph.set_entry_point("agent")
# - graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
# - graph.add_edge("tools", "agent")
# - app = graph.compile()
#
# 第8步：测试
# - 准备需要多轮工具调用的测试问题
# - 用 app.invoke(initial_state) 执行
# - 打印最终回答和执行路径

# 前置准备：
# - 需要安装的包：langgraph, langchain, langchain-openai, python-dotenv
# - 需要设置的环境变量：OPENAI_API_KEY
#
# 提示：
# - 不要硬编码 API Key
# - bind_tools 让 LLM 输出的消息中包含 tool_calls 字段
# - ToolMessage 的 tool_call_id 必须与 tool_calls 中的 id 对应
# - 测试时可以用 temperature=0.0 确保结果稳定
