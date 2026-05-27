"""
章节：第05章 任务规划与分解
题目：实现一个完整的 Plan-and-Execute Agent
类型：进阶练习

题目描述：
整合规划、执行、重规划和追踪四大模块，实现一个完整的 Plan-and-Execute Agent。
Agent 接收用户的复杂任务描述，自动生成执行计划，按依赖关系执行子任务，
在失败时触发重规划，并记录完整的执行轨迹。

要求：
1. 定义 SubTask 和 Plan 数据结构（使用 dataclass）
   - SubTask: id, description, action, action_args, depends_on, status, result, error 等
   - Plan: id, name, tasks 列表, context 字典, progress() 方法等
2. 实现 ToolExecutor 类
   - 至少注册 5 个模拟工具（异步函数，使用 asyncio.sleep 模拟 I/O 延迟）
   - 支持参数引用解析（如 "$task_1.result.data"）
   - 工具应有不同的执行时间和成功率（可以部分失败以测试重规划）
3. 实现 LLM 规划器（PlanGenerator）
   - 设计 System Prompt（角色、工具列表、输出 Schema、示例）
   - 调用 LLM 生成结构化 JSON 计划
   - 解析 JSON → Plan 对象
4. 实现并行执行引擎
   - 持续检查并执行所有依赖已满足的任务
   - 使用 asyncio.gather 实现真正的并发
   - 支持 max_parallel 限制
5. 实现重规划器（Replanner）
   - 支持局部重试（retry）和路径替换（replace_path）两种策略
   - 重规划时保留已完成任务
   - 限制最大重规划次数
6. 实现执行追踪器（ExecutionTracker）
   - 复用 ex3 的思路，记录所有生命周期事件
   - 生成执行报告和可视化时间线
7. 将所有模块整合到 PlanExecuteAgent 类中
   - 提供 run(user_task, context) 接口
   - 输出完整执行报告

提示：
- 使用 asyncio 实现并发执行，注意合理使用 await 和 gather
- 工具参数中的 $ 引用需要递归解析嵌套路径
- 计划执行到"无就绪任务但还有 PENDING 任务"时，触发死锁检测和重规划
- verbose 模式下应有丰富的控制台输出，便于观察执行过程
- 重规划次数过多说明计划本身有问题，应设置上限
"""

import os
import json
import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
from dataclasses import dataclass, field

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：数据结构定义
# - TaskStatus 枚举: PENDING, IN_PROGRESS, COMPLETED, FAILED, SKIPPED, WAITING
# - SubTask dataclass: id, description, action, action_args, depends_on,
#   output_key, status, result, error, retry_count, max_retries 等
# - Plan dataclass: id, name, tasks, context, created_at, version,
#   get_ready_tasks(), is_complete(), progress() 方法
#
# 第2步：ToolExecutor 类
# - __init__(): 注册至少 5 个异步模拟工具函数
#   建议工具: query_database, analyze_data, search_web, generate_report,
#             send_email, fetch_market_data 等
# - execute(task, context): 解析参数引用 → 调用工具 → 返回结果
# - _resolve_args(args, context): 解析 $ 引用
# - _resolve_ref(ref, context): 递归解析嵌套引用路径
#
# 第3步：规划器
# - build_planner_prompt(): 构建 System Prompt
# - generate_plan(user_task, context): 调用 LLM → JSON → Plan
# - 在 prompt 中提供 1 个完整的示例计划（9 个左右子任务）
#
# 第4步：并行执行引擎
# - 主循环：while not plan.is_complete()
# - 每轮：get_ready_tasks() → asyncio.gather(*coroutines)
# - 检测死锁：无就绪任务 + 有 PENDING 任务 → 触发重规划
# - 处理失败传播：标记依赖失败任务的下游任务为 SKIPPED
#
# 第5步：重规划器
# - replan(plan, failed_task, strategy):
#   * retry: 重置任务状态为 PENDING，增加 retry_count
#   * replace_path: 生成替代任务，更新下游依赖
# - 记录重规划历史和版本号
#
# 第6步：整合 PlanExecuteAgent 类
# - __init__(llm_client, tool_executor, max_parallel, verbose)
# - run(user_task, context, max_replans) → 执行结果字典
# - print_final_report(result)
#
# 第7步：测试
# - 准备一个复杂任务（需要 7-10 个子任务）
# - 运行 Agent 观察并行执行和重规划行为
# - 验证输出报告的正确性
#
# 前置准备：
# - 需要安装的包：openai, python-dotenv
# - 需要设置的环境变量：OPENAI_API_KEY
#
# 提示：
# - 不要硬编码 API Key
# - 使用 asyncio.create_task 启动独立的异步任务
# - 模拟工具中可以用 asyncio.sleep(随机时间) 模拟不同的 I/O 延迟
# - 某些模拟工具可以故意抛出异常来测试重规划逻辑
# - verbose 输出要结构化，便于观察并行执行效果
