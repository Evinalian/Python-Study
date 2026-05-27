"""
章节：第07章 LangGraph 工作流引擎
题目：实现多意图路由 Agent
类型：进阶练习

题目描述：
构建一个能根据用户意图自动路由到不同处理通道的 Agent。使用条件边实现
意图分类 → 多通道路由，某些通道内部包含工具调用循环。

要求：
1. 定义路由状态 RouterState：
   - messages: Annotated[list[BaseMessage], operator.add]
   - intent: str（分类结果）
   - channel_results: dict（各通道的执行结果）
   - iteration_count: int
2. 实现意图分类节点：
   - 调用 LLM 将用户问题分类为: return_refund / shipping / product_inquiry / complaint / general
   - 使用 response_format={"type": "json_object"} 确保 JSON 输出
3. 为每种意图实现处理节点：
   - return_refund: 退货流程（可能需要查询订单 → 计算退款 → 生成说明）
   - shipping: 物流查询（可能需要查询订单 → 获取物流信息）
   - complaint: 投诉处理（升级机制）
   - general: 通用对话（简单的 LLM 回复）
4. 对需要多步工具调用的意向，内部实现迷你 ReAct 循环：
   - 可以使用子图（build_return_subgraph()）封装
   - 子图内包含 agent_node + tools_node 循环
5. 实现路由函数：
   - route_by_intent(state) → 返回对应的处理节点名
6. 实现循环安全：
   - 最大 5 轮 ReAct 迭代
   - 超时处理（对复杂意图设置时间上限）
7. 测试至少 3 种不同意图的完整流程

提示：
- 意图分类用 LLM + JSON Mode 确保结构化输出
- 子图编译后可以像普通节点一样 add_node 到父图
- 子图的 State 可以是父 State 的子集（只包含子图需要的字段）
- 复杂意图的 ReAct 循环要确保能终止
- 处理节点模拟返回即可（不需要真实的工具调用）
"""

from typing import TypedDict, Annotated, Literal
import operator
import json

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：定义 RouterState
# - messages: Annotated[list[BaseMessage], operator.add]
# - intent: str ("", "return_refund", "shipping", "product_inquiry", "complaint", "general")
# - intent_reason: str（分类理由）
# - channel_results: dict（各通道结果）
# - iteration: int
#
# 第2步：实现 classify_intent 节点
# - 获取 messages[-1].content
# - 构建分类 prompt（包含意图列表和说明）
# - 调用 LLM, temperature=0.0, response_format={"type": "json_object"}
# - 解析 JSON: {"intent": "...", "reason": "..."}
# - 返回: intent + intent_reason + messages
#
# 第3步：实现各意图的处理节点
# 对每个意图实现一个处理函数：
# - handle_return(state) → 返回模拟的退货指导
# - handle_shipping(state) → 返回模拟的物流查询结果
# - handle_product(state) → 返回模拟的产品信息
# - handle_complaint(state) → 记录投诉并升级
# - handle_general(state) → 调用 LLM 做通用回复
# 每个处理函数返回的结果存入 channel_results[state["intent"]]
#
# 第4步：实现 route_by_intent 路由函数
# - 根据 state["intent"] 返回对应的节点名
# - 使用映射字典: {"return_refund": "handle_return", ...}
# - 返回 Literal 类型
#
# 第5步：（可选进阶）实现子图
# - 对于 return_refund 意图，构建一个子图
#   * 子图内部: query_order → calculate_refund → generate_instructions
# - 子图有自己的 State，从父 State 中提取所需字段
# - build_return_subgraph() 返回编译后的子图
#
# 第6步：构建父图
# - add_node 添加所有意图处理节点
# - add_node 添加分类节点
# - set_entry_point("classify")
# - add_conditional_edges("classify", route_by_intent, {...映射...})
# - 所有处理节点 → END
# - compile()
#
# 第7步：测试
# - 准备 5 个测试用例，覆盖所有意图
# - 分别 invoke 并验证路由正确性
# - 打印每个用例的: 用户输入 → 识别的意图 → 处理结果摘要

# 前置准备：
# - 需要安装的包：langgraph, langchain, langchain-openai, python-dotenv
# - 需要设置的环境变量：OPENAI_API_KEY
