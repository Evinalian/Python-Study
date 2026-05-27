"""
章节：第07章 LangGraph 工作流引擎
题目：构建一个完整的客服 Agent 工作流
类型：进阶练习

题目描述：
整合本章所有概念，构建一个企业级客服 Agent 工作流。这个工作流应该具备：
意图路由、ReAct 工具调用、Human-in-the-Loop 审批、循环安全、流式输出。

要求：
1. 定义完整的企业级状态 CustomerServiceState：
   - messages: 对话历史
   - intent: 识别的意图
   - status: 工作流状态 ("classifying" / "executing" / "waiting_approval" / "done")
   - iteration: ReAct 循环计数
   - approval_required: bool（当前操作是否需要审批）
   - approval_status: str
   - pending_action: str（等待审批的操作描述）
   - executed_tools: list[str]（已执行的工具列表，用于审计）
   - final_result: str
2. 实现至少 6 个节点：
   a. classify_intent: 意图分类（5+ 意图）
   b. agent: LLM 决策节点（ReAct 核心）
   c. tools: 工具执行节点
   d. approval_check: 审批检查节点
   e. wait_approval: 等待人类审批（interrupt）
   f. final_respond: 最终回复生成
3. 实现至少 4 个工具：
   - query_order(order_id): 查询订单
   - query_shipping(tracking_number): 查询物流
   - calculate_refund(order_amount, days): 计算退款
   - escalate_to_human(reason): 升级给人工客服
4. 实现至少 3 条条件边：
   - route_by_intent: 分类后路由
   - route_after_agent: Agent决策后的路由（tools/approval_check/END）
   - route_after_approval: 审批结果后的路由
5. 实现循环安全机制：
   - 最大 10 轮 ReAct 迭代
   - 每轮输出剩余配额给日志
   - 达到上限时生成"升级给人工"的兜底回复
6. 实现流式输出：
   - 使用 astream() 进行异步流式输出
   - 每次节点更新打印节点名和关键状态变化
7. 编写完整的端到端测试：
   - 测试 3 个不同意图的完整流程
   - 验证循环终止逻辑
   - 验证审批流程（模拟人类决策）
   - 打印完整的执行审计日志

提示：
- 使用 LangChain 的 BaseMessage 体系管理对话历史
- 审批检查逻辑: 检测 Agent 是否准备执行"发送"类操作
- 流式输出中，每个 event 的 key 是节点名，value 是该节点的输出
- 工具注册使用 TOOLS_BY_NAME 字典方便查找
- escalate_to_human 工具触发后应立即终止当前流程
"""

from typing import TypedDict, Annotated, Literal
import operator
import json
import asyncio

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：定义数据结构
# - CustomerServiceState (TypedDict)
# - 使用 Annotated 正确配置 reducer
# - messages 用 operator.add
# - 其他字段用默认替换策略
#
# 第2步：定义工具
# - 使用 @tool 装饰器定义 4+ 个工具
# - 每个工具要有完整的 docstring（description 和 args 说明）
# - TOOLS_BY_NAME 字典做名称到工具的映射
#
# 第3步：实现 intent_classifier 节点
# - 使用 LLM + JSON Mode 分类
# - 意图: return_refund, shipping_query, product_inquiry, complaint, general
# - 返回结构化分类结果
#
# 第4步：实现 agent_node
# - 使用 ChatOpenAI + bind_tools
# - 构建包含客服身份和规则的 system prompt
# - 处理 tool_calls 检测
# - 设置标志位: needs_tools, needs_approval
#
# 第5步：实现 tools_node
# - 提取 tool_calls 并执行
# - 创建 ToolMessage（正确设置 tool_call_id）
# - 对于 escalate_to_human 工具 → 设置特殊标志
#
# 第6步：实现 approval_check_node
# - 检查是否需要进行审批的操作（如发送邮件、退款执行）
# - 如果需要 → 设置 approval_status="pending"
# - 如果不需要 → 设置 approval_status="none"
#
# 第7步：实现 wait_approval_node
# - 调用 interrupt(approval_context)
# - 接收人类决策
# - 更新 approval_status
#
# 第8步：实现 final_respond_node
# - 基于完整的对话历史生成最终回复
# - 或提取最后一条 AI 消息作为回复
#
# 第9步：实现路由函数
# - route_by_intent → 意图到处理节点映射
# - route_after_agent → tools / approval_check / respond / END
# - route_after_approval → agent / respond / END
# - route_after_tools → agent
#
# 第10步：构建和编译图
# - 添加所有节点
# - 设置入口点
# - 添加所有条件边
# - 确保所有路径最终可达 END
# - compile()
#
# 第11步：实现测试
# - sync_test(): 同步调用测试基本功能
# - async_stream_test(): 异步流式输出测试
# - approval_test(): 测试 Human-in-the-Loop 流程
# - 打印执行摘要: 路径、迭代次数、工具调用次数

# 前置准备：
# - 需要安装的包：langgraph, langchain, langchain-openai, python-dotenv
# - 需要设置的环境变量：OPENAI_API_KEY
#
# 提示：
# - 这是一个复杂的工作流，建议先画出流程图再编码
# - 分模块测试：先测节点函数，再测路由，最后测完整流程
# - 循环安全是生产环境的必需品，不要省略
# - Human-in-the-Loop 的 interrupt 恢复需要正确使用 Command
