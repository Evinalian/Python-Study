"""
章节：第07章 LangGraph 工作流引擎
题目：构建一个基本的状态图
类型：基础练习

题目描述：
使用 LangGraph 构建一个简单的 3-4 步工作流，掌握 StateGraph、Node、Edge 的基本用法。
这是 LangGraph 的"Hello World"——理解这几个核心概念后，复杂工作流只是它们的组合。

要求：
1. 使用 TypedDict 定义 State（至少包含 counter、messages、status 三个字段）
2. 至少定义 3 个节点函数：
   - initialize: 初始化状态
   - process: 处理数据（如修改 counter 值）
   - finalize: 最终处理和清理
3. 使用普通边连接 initialize → process → finalize
4. 实现一个条件边：根据 counter 的值决定下一步
   - counter >= 10 → 直接结束
   - counter < 10 → 再经过一次 process 再结束
5. 添加至少一条 END 边
6. compile() 编译图并用 invoke() 运行
7. 打印初始和最终状态对比

提示：
- 安装: pip install langgraph langchain langchain-openai
- messages 字段使用 Annotated[list[str], operator.add] 实现消息追加
- 节点函数返回 dict，只包含需要更新的字段
- add_conditional_edges 需要三个参数：源节点、路由函数、路由映射
- 路由映射中必须包含 END
"""

from typing import TypedDict, Literal, Annotated
import operator

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：定义 State
# - 使用 TypedDict 定义 WorkflowState
# - counter: int（普通替换策略）
# - messages: Annotated[list[str], operator.add]（追加策略）
# - status: str（普通替换策略）
#
# 第2步：实现节点函数
# - initialize(state) → 设置 counter=0, status="initialized", messages=["开始"]
# - process(state) → counter += 5, messages 追加处理日志
# - finalize(state) → status="done", messages 追加完成日志
# - extra_process(state) → counter += 1, messages 追加额外处理日志
#
# 第3步：实现路由函数
# - route_after_process(state) → 返回 "extra" 或 "finalize"
#   * counter >= 10 → "finalize"
#   * counter < 10 → "extra"
#
# 第4步：构建图
# - graph = StateGraph(WorkflowState)
# - add_node() 注册所有节点
# - set_entry_point("initialize")
# - add_edge() 连接 initialize → process
# - add_conditional_edges() 连接 process → extra / finalize
# - add_edge() 连接 extra → process（形成循环）
# - add_edge() 连接 finalize → END
# - compile()
#
# 第5步：测试
# - 用 invoke() 执行
# - 打印 initial_state 和 final_state
# - 验证 counter 的值是否符合预期
# - 验证 messages 是否按顺序累积
#
# 提示：
# - 条件边的路由映射格式: {"route_value": "target_node_name"}
# - 如果路由函数返回的值不在映射的 key 中 → 运行时错误
# - 注意循环的终止条件，确保最终能到达 END
# - 使用 Literal 类型注解路由函数的返回类型
