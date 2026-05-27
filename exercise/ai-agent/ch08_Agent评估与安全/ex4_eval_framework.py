"""
章节：第08章 Agent评估与安全
题目：实现完整的 Agent 评估框架
类型：进阶练习

题目描述：
整合本章所有概念，构建一个完整的 Agent 评估框架。这个框架应该能够：
- 管理测试基准（多种类型的测试用例）
- 自动化评估 Agent 在测试集上的表现
- 生成多维度的评估报告
- 检测安全问题

要求：
1. 实现完整的评估数据结构：
   - TestCase: 测试用例（id, category, user_input, expected_behavior, etc.）
   - EvalResult: 单维度评估结果（dimension, score, reason, evidence）
   - AgentTrajectory: Agent 执行轨迹
   - EvalReport: 完整评估报告
2. 实现测试基准构建器：
   - 至少 8 个测试用例，覆盖 5 种类型（Happy Path, Edge Case, Error, Adversarial, Multi-turn）
   - 每个用例包含描述、期望行为、成功标准
3. 实现综合评估器：
   - TaskCompletionEvaluator: LLM-as-Judge 评估任务完成度
   - ToolAccuracyEvaluator: 评估工具调用准确率
   - 至少 2 个评估维度
4. 实现 AgentEvalFramework 主类：
   - run_evaluation(agent_fn, benchmark): 运行完整评估
   - 对每个用例: 安全检查 → 执行 Agent → 多维度评估 → 记录结果
   - generate_report(): 生成 EvalReport
5. 报告包含：
   - 各维度平均分
   - 综合评分（通过率）
   - 改进建议
   - 安全事件统计
6. 测试：
   - 用一个模拟的 Agent（简单的规则实现）跑评估流程
   - 验证框架能正确评分

提示：
- 框架设计要考虑可扩展性：容易添加新的评估维度
- 模拟 Agent 可以用一个简单的函数替代（返回预设的回复）
- 评估报告要包含 actionable 的信息——告知开发者具体如何改进
- 安全检查和评估检查可以在框架中独立配置
"""

import json
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：定义所有数据结构
# - TestCaseCategory 枚举: HAPPY_PATH, EDGE_CASE, ERROR, ADVERSARIAL, MULTI_TURN
# - EvalDimension 枚举: TASK_COMPLETION, TOOL_ACCURACY, EFFICIENCY, SAFETY
# - TestCase dataclass: 包含 id, category, description, user_input,
#   expected_behavior, expected_key_points, expected_tools,
#   success_criteria, max_expected_steps
# - EvalResult dataclass: dimension, score, max_score, reason, evidence
# - EvalReport dataclass: agent_name, task_count, results, dimension_scores,
#   overall_score, summary, recommendations
#   * print_summary() 方法打印格式化的报告
#
# 第2步：实现测试基准构建器
# - build_benchmark() → list[TestCase]
# - 至少 8 个用例:
#   * 2 Happy Path（正常订单查询、简单计算）
#   * 2 Edge Case（不完整的输入、边界数据）
#   * 1 Error（不存在的资源）
#   * 2 Adversarial（Prompt Injection、权限提升尝试）
#   * 1 Multi-turn（需要上下文记忆）
#
# 第3步：实现评估器
# - TaskCompletionEvaluator:
#   * 使用 LLM-as-Judge
#   * 评估 prompt 包含用户任务 + Agent 回复 + 评估标准
#   * 返回 EvalResult
# - ToolAccuracyEvaluator:
#   * 检查 Agent 是否调用了期望的工具
#   * 检查是否调用了禁止的工具
#   * 检查是否有冗余的工具调用
#
# 第4步：实现 AgentEvalFramework
# - __init__(name, agent_fn, benchmark, evaluators)
# - run_evaluation(verbose=True):
#   * 遍历 benchmark 中的每个 TestCase
#   * 可选: 安全检查（Prompt Injection 检测）
#   * 调用 agent_fn(user_input) 获取回复
#   * 对每个 evaluator 执行评估
#   * 记录结果: {case_id, category, passed, eval_results, security_issues}
# - generate_report():
#   * 计算各维度平均分
#   * 计算通过率 (passed / total)
#   * 生成改进建议:
#     - 通过率 < 80% → 整体优化建议
#     - 某维度 < 0.7 → 针对该维度的优化建议
#     - 安全事件数 > 0 → 安全加固建议
#   * 返回 EvalReport
#
# 第5步：测试
# - 创建一个简单的模拟 Agent 函数
#   * 对已知查询返回预设回复
#   * 对未知查询返回通用回复
# - 构建 benchmark
# - 运行框架
# - 验证报告内容是否合理

# 提示：
# - 模拟 Agent 可以是一个字典映射: {用户问题 → 预定义回复}
# - 安全事件统计来自检查过程中发现的注入、泄露等问题
# - 通过标准: 所有必选评估维度 >= 阈值 + 无安全事件
# - 报告应该是对开发者有用的（不只是数字，还有 actionable insights）
