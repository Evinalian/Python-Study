"""
章节：第06章 多智能体协作
题目：实现一个简单的主从多 Agent 系统
类型：基础练习

题目描述：
构建一个三 Agent 系统（协调者 + 数据处理器 + 报告生成器），完成"分析文本数据并生成报告"的任务。
这是多 Agent 系统最基础的实现，理解协调者如何分配任务、Worker 如何协作。

要求：
1. 实现 Agent 基类，包含 name, role, system_prompt 和 think(task, context) 方法
2. 实现 Orchestrator 类（继承 Agent），包含：
   - plan(user_task, worker_profiles): 调用 LLM 分析任务并生成任务分配计划
   - execute(assignments, workers, results): 按计划依次执行任务
   - synthesize(assignments, results): 汇总各 Worker 的输出
3. 实现至少两个专业的 Worker Agent（如数据分析师、报告撰写师）
4. 实现 OrchestratorWorkerSystem 类，整合协调者和 Worker
5. 用模拟 LLM 调用（或真实的 OpenAI API）完成一次完整的协作流程

提示：
- Worker Agent 的 system_prompt 要比协调者更"专业"和"狭窄"
- 协调者的 plan 方法返回一个任务分配列表，包含 worker_name, task_description, expected_output
- execute 方法需要构建上下文传递给 Worker（包含前面 Worker 的输出）
- 使用 asyncio 或简单的同步调用均可
- 如果用真实 API，注意设置 temperature 控制输出稳定性
"""

import os
import json
import asyncio
from typing import Any, Optional
from dataclasses import dataclass, field

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：实现 TaskAssignment 数据结构
# - worker_name: 分配给哪个 Worker
# - task_description: 具体任务描述
# - expected_output: 期望的输出格式/内容
# - priority: 优先级 (1-10)
# - depends_on: 依赖的前置任务索引列表
#
# 第2步：实现 Agent 基类
# - __init__(name, role, system_prompt, tools, temperature)
# - think(task, context): 调用 LLM 处理任务
#   * 构建 messages: [system_prompt, context, task]
#   * 调用 client.chat.completions.create
#   * 返回 response.choices[0].message.content
# - get_profile(): 返回 Agent 的能力画像 {"name", "role", "expertise", "tools"}
#
# 第3步：实现 Orchestrator 类（继承 Agent）
# - register_worker(worker): 注册 Worker
#   * 维护 self.workers 列表和 self.worker_profiles 列表
# - plan(user_task): 生成任务分配计划
#   * 构建 prompt 描述可用 Worker 及其能力
#   * 调用 LLM 生成分配计划（JSON 格式）
#   * 返回 list[TaskAssignment]
# - execute(assignments, max_parallel): 执行分配计划
#   * 维护 results 字典 (assignment_index → result)
#   * 每轮找出依赖已满足的就绪任务
#   * 并行执行就绪任务（asyncio.gather）
#   * 为每个 Worker 构建包含依赖任务输出的 context
# - synthesize(assignments, results): 汇总结果
#   * 将所有 Worker 的输出传给协调者
#   * 协调者整合为连贯的最终回复
#
# 第4步：定义两个专业 Worker 的配置和实例
# - 数据分析师: 专注数据处理、统计分析、趋势识别
# - 报告撰写师: 专注将分析结果转化为结构化的专业报告
# - 每个 Agent 的 system_prompt 要详细定义其能力、边界、输出风格
#
# 第5步：实现 OrchestratorWorkerSystem 类
# - __init__(llm_client): 初始化
# - set_orchestrator(o) / add_worker(w)
# - run(user_task): 完整执行流程
#   1. 协调者分析任务 → 2. Workers 执行 → 3. 协调者汇总
#
# 第6步：测试
# - 准备一个复杂任务（如"分析一组销售数据，找出趋势，生成管理层报告"）
# - 运行系统观察分工和执行过程

# 前置准备：
# - 需要安装的包：openai, python-dotenv
# - 需要设置的环境变量：OPENAI_API_KEY

# 提示：
# - Agent 基类中 think 方法不要硬编码 API Key
# - 协调者的 plan prompt 要包含完整的 Worker 画像和分配原则
# - 每个 Worker 的 think 调用要传入 context（来自依赖 Worker 的输出）
# - 使用 temperature=0.0 确保规划稳定
# - 模拟数据时可以用简化的 mock 工具
