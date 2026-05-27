"""
章节：第05章 任务规划与分解
题目：实现带状态追踪的任务执行追踪器
类型：基础练习

题目描述：
实现一个完整的任务执行追踪系统，记录每个子任务从创建到完成的全生命周期状态变化。
追踪器需要维护状态流转、记录时间戳，并能够生成执行报告和时间线。

要求：
1. 定义任务状态枚举：PENDING / IN_PROGRESS / COMPLETED / FAILED / SKIPPED
2. 定义任务事件结构：包含时间戳、任务ID、事件类型、详情字典
3. 实现 ExecutionTracker 类：
   - record(event_type, task_id, **details): 记录事件
   - task_created / task_started / task_completed / task_failed: 便捷方法
   - generate_report(): 生成执行统计报告
   - print_timeline(): 打印文本版执行时间线
4. 状态流转规则：
   - PENDING → IN_PROGRESS（开始执行）
   - IN_PROGRESS → COMPLETED（成功）/ FAILED（失败）
   - FAILED → PENDING（重试）
   - PENDING → SKIPPED（因依赖失败而跳过）
5. 报告内容：总耗时、任务完成率、每个任务的等待时间和执行时间、事件数量

提示：
- 使用 datetime.now().isoformat() 记录时间戳，便于排序和解析
- generate_report 中按任务聚合事件，计算每个任务的时间指标
- print_timeline 可以将事件按任务分组后按时间顺序打印
- 支持 on_event 回调机制，便于后续扩展（如实时推送）
- 注意处理任务重试的情况（多次 IN_PROGRESS/Failed 事件）
"""

import time
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
from dataclasses import dataclass, field

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义 TaskStatus 枚举，包含 5 个状态值
#
# 2. 定义 TaskEvent 数据类
#    - timestamp: str（ISO 格式时间戳）
#    - task_id: str
#    - event_type: str（"created" / "started" / "completed" / "failed" / "skipped"）
#    - details: dict（额外信息如 result_preview, error_message 等）
#
# 3. 实现 ExecutionTracker 类
#    - __init__(self, plan_id): 初始化事件列表、开始/结束时间、回调列表
#    - start_plan() / end_plan(): 记录计划级别的开始和结束
#    - record(event_type, task_id, **details): 核心方法，创建 TaskEvent 并通知回调
#    - on_event(callback): 注册事件回调函数
#    - task_created(task) / task_started(task) / task_completed(task, result):
#      task_failed(task, error) / task_retried(task): 便捷方法
#    - generate_report(): 生成报告字典
#      * 按 task_id 聚合事件
#      * 计算每个任务的 wait_time（created→started）和 execution_time（started→completed/failed）
#      * 统计完成率
#      * 返回结构化的报告字典
#    - print_timeline(): 打印文本时间线
#      * 按任务分组
#      * 显示状态图标（✅/❌/⏳/⏭）
#      * 显示耗时
#
# 4. 编写测试代码
#    - 创建 ExecutionTracker 实例
#    - 模拟 4-5 个任务的完整执行（含成功、失败、重试）
#    - 打印执行报告和时间线
#
# 提示：
# - datetime.fromisoformat 可以解析 ISO 时间戳字符串
# - 计算等待时间 = 开始时间 - 创建时间（需要至少两个事件）
# - 重试任务会有多次 started/failed 事件，注意区分
# - 回调函数应该捕获异常，避免一个回调崩溃影响其他回调
