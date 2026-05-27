"""
章节：第05章 任务规划与分解
题目：实现智能动态重规划器
类型：进阶练习

题目描述：
实现一个智能重规划器，能在子任务执行失败时自动选择最合适的重规划策略，
在保留已完成工作成果的前提下调整后续计划，避免"一个失败导致全盘重来"。

要求：
1. 实现三种重规划策略：
   a. retry（重试）：将失败任务重置为 PENDING，增加重试计数
   b. replace_path（路径替换）：保持已完成任务，用替代方案替换失败任务及其下游
   c. full_replan（全局重规划）：保留已完成任务，放弃所有 PENDING 任务，重新生成后续计划
2. 实现自动策略选择：
   - 失败任务的重试次数 < max_retries → retry
   - 重试次数已耗尽，且失败影响范围小（<3 个下游） → replace_path
   - 重试次数已耗尽，且失败影响范围大（>=3 个下游） → full_replan
3. 实现计划快照机制：
   - 记录每次重规划前后的状态
   - 支持回滚（如果重规划后效果更差）
4. 记录完整的重规划历史：
   - 每次重规划记录：版本号、触发任务、失败原因、采用策略、影响的任务
5. 实现错误分类：
   - 将失败原因分类为：网络错误、数据格式错误、权限错误、资源耗尽等
   - 不同错误类型对应不同的默认策略
6. 实现执行结果评估：
   - 重规划后统计改善效果

提示：
- 重规划的本质是"局部修改"，不触碰已完成的任务
- replace_path 策略可以用 LLM 生成替代子任务（需要设计重规划的 prompt）
- 计划快照用 deepcopy 保存任务列表的完整副本
- 错误分类是选择策略的重要依据：网络错误适合重试，数据格式错误适合换路径
"""

import copy
import json
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：定义相关数据结构
# - ErrorCategory 枚举：NETWORK_ERROR, DATA_FORMAT_ERROR, PERMISSION_ERROR,
#   RESOURCE_EXHAUSTED, TOOL_UNAVAILABLE, UNKNOWN
# - ReplanStrategy 枚举：RETRY, REPLACE_PATH, FULL_REPLAN
# - ReplanRecord dataclass：version, trigger_task, error, strategy, affected_tasks,
#   timestamp, reason
#
# 第2步：实现 classify_error(error_message: str) → ErrorCategory
# - 根据错误消息的关键词判断错误类别
# - "timeout"/"connection" → NETWORK_ERROR
# - "format"/"parse"/"schema" → DATA_FORMAT_ERROR
# - "permission"/"denied"/"unauthorized" → PERMISSION_ERROR
# - "memory"/"quota"/"rate limit" → RESOURCE_EXHAUSTED
# - "not found"/"unavailable" → TOOL_UNAVAILABLE
# - 其他 → UNKNOWN
#
# 第3步：实现 choose_strategy(failed_task, plan, error_category) → ReplanStrategy
# - retry_count < max_retries → RETRY
# - 影响下游任务数 < 3 → REPLACE_PATH
# - 影响下游任务数 >= 3 → FULL_REPLAN
# - PERMISSION_ERROR → FULL_REPLAN（权限问题重试没用）
#
# 第4步：实现 Replanner 类
# - __init__(llm_client, max_replans)
# - replan(plan, failed_task) → 修改后的 Plan
#   * 自动选择策略
#   * 调用对应的策略方法
#   * 更新计划版本号和历史
# - _retry(failed_task):
#   * 重置为 PENDING，清除错误，retry_count++
# - _replace_path(plan, failed_task):
#   * 定位所有依赖失败任务的下游任务
#   * 可选：调用 LLM 生成替代方案
#   * 将替代任务插入计划，更新下游依赖
#   * 标记原失败任务为 SKIPPED
# - _full_replan(plan, failed_task):
#   * 找到最后完成的任务
#   * 标记所有 PENDING 任务为 SKIPPED
#   * 可选：调用 LLM 根据已完成结果重新生成后续
# - _snapshot(plan) → 计划快照
# - _count_affected_tasks(plan, failed_task_id) → 受影响的任务数
#
# 第5步：编写测试用例
# - 测试场景1：网络超时 → 应选择 RETRY
# - 测试场景2：数据格式错误，已重试3次 → 应选择 REPLACE_PATH
# - 测试场景3：权限错误 → 应选择 FULL_REPLAN
# - 测试场景4：大规模失败（影响5+下游） → 应选择 FULL_REPLAN
# - 验证每次重规划后的计划版本号递增
# - 验证已完成任务在重规划后状态不变
#
# 提示：
# - 重规划时要特别注意依赖更新：下游任务的 depends_on 需要从失败任务 ID 改为替代任务 ID
# - 重规划历史是重要的审计数据，应包含足够的信息供后续分析
# - 如果计划中有多个失败任务，优先处理依赖链最上游的那个
# - 考虑设置重规划冷却期（短时间内不重复触发同一任务的重规划）
