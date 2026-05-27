"""
章节：第07章 LangGraph 工作流引擎
题目：实现 Human-in-the-Loop 审批流程
类型：基础练习

题目描述：
在工作流中插入人类审批节点。当 Agent 执行到需要审批的操作时，工作流暂停，
等待人类做出决定后再继续执行。这是企业级 Agent 系统的关键安全机制。

要求：
1. 设计一个"数据分析 → 生成报告 → 审批 → 发送"的工作流
2. 定义 ApprovalState：
   - data: str（待分析的数据）
   - analysis_result: str
   - report: str
   - approval_status: str ("none" / "pending" / "approved" / "rejected")
   - final_result: str
3. 实现以下节点：
   - analyze_data: 分析原始数据（模拟）
   - generate_report: 基于分析生成报告（模拟）
   - request_approval: 使用 interrupt() 暂停工作流，等待人类审批
   - send_report: 审批通过后发送报告（模拟）
   - handle_rejection: 审批拒绝后处理
4. 使用 interrupt() 暂停工作流：
   - 在 request_approval 节点中调用 interrupt(approval_payload)
   - 外部系统收到暂停信号，展示审批内容给人类
   - 人类选择 approve 或 reject
   - 使用 Command(resume=decision) 恢复工作流
5. 使用条件边根据审批结果路由：
   - approved → send_report → END
   - rejected → handle_rejection → END
6. 编写完整的测试流程（模拟人类审批）

提示：
- from langgraph.types import interrupt, Command
- interrupt() 会暂停执行并返回调用方，需要的信息作为参数传入
- 恢复时使用 app.invoke(Command(resume={"decision": "approve"}))
- 审批节点中调用 interrupt() 后，后续代码会在恢复时继续执行
- interrupt() 可以传递任何可序列化的数据给外部系统
"""

from typing import TypedDict, Annotated, Literal
import operator

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：定义 ApprovalState
# - data: str
# - analysis_result: str
# - report: str
# - approval_status: str ("none" / "pending" / "approved" / "rejected")
# - final_result: str
# - logs: Annotated[list[str], operator.add]
#
# 第2步：实现 analyze_data 节点
# - 模拟分析原始数据
# - 返回: analysis_result + logs
#
# 第3步：实现 generate_report 节点
# - 基于 analysis_result 生成报告
# - 返回: report + logs
#
# 第4步：实现 request_approval 节点
# - 调用 interrupt() 暂停工作流
# - interrupt 的参数是一个 dict，包含：
#   * "message": 审批说明
#   * "report": state["report"] (展示给人类审阅的内容)
#   * "options": ["approve", "reject"]
# - interrupt 的返回值是人类的决定
# - 根据决定设置 approval_status
# - 返回: approval_status + logs
#
# 第5步：实现 send_report 节点
# - 模拟发送报告（仅在 approved 时执行）
# - 返回: final_result + logs
#
# 第6步：实现 handle_rejection 节点
# - 处理拒绝情况（记录拒绝原因等）
# - 返回: final_result + logs
#
# 第7步：实现路由函数
# - route_after_approval(state) → "send" 或 "reject" 或 END
#   根据 approval_status 路由
#
# 第8步：构建图
# - analyze_data → generate_report → request_approval → 条件边
# - 条件边: approved → send_report → END
# - 条件边: rejected → handle_rejection → END
# - compile()
#
# 第9步：测试
# - test_data = "...模拟数据..."
# - 第一次: app.invoke({"data": test_data, ...}) → 暂停在 request_approval
# - 第二次: app.invoke(Command(resume="approve")) → 继续执行 send_report
# - 打印最终结果

# 提示：
# - interrupt() 是同步函数，在 async 节点中也这样调用
# - 暂停后的状态保存在 LangGraph 的内部 checkpoint 中
# - 恢复时只需要传入 Command，不需要重新传入初始状态
# - 如果工作流中只有一个 interrupt 点，恢复后的执行不需要额外的节点指定
