"""
章节：第08章 Agent评估与安全
题目：实现工具权限控制器 (ToolGuard)
类型：基础练习

题目描述：
实现一个工具守卫（ToolGuard），在 Agent 执行工具调用前进行权限检查。
这是 Agent 安全的最后一道防线——即使 LLM 决策出错或被注入，工具守卫
仍能阻止危险操作。

要求：
1. 定义 ToolPermission 数据类：
   - tool_name: 工具名称
   - risk_level: 风险等级 (READ_ONLY / WRITE / DESTRUCTIVE / SENSITIVE)
   - requires_approval: 是否需要人工审批
   - max_calls_per_session: 每会话最大调用次数
   - allowed_args_patterns: 参数白名单（正则验证）
   - blocked_args_patterns: 参数黑名单（正则拒绝）
   - allowed_contexts: 允许的执行上下文列表
2. 实现 ToolGuard 类：
   - register_tool(permission): 注册工具权限
   - check(tool_name, tool_args, context):
     * 检查1: 工具是否注册
     * 检查2: 调用次数是否超限
     * 检查3: 参数是否符合白名单
     * 检查4: 参数是否匹配黑名单
     * 检查5: 上下文是否允许
     * 检查6: 是否需要审批
     * 返回 {allowed, reason, requires_approval}
   - get_audit_log(): 返回调用统计和阻止记录
3. 实现至少 5 种不同的检查逻辑
4. 测试合法调用和被拒绝的调用（每种被拒绝的类型至少 1 个用例）
5. 实现审计日志功能：
   - 记录每次检查的结果
   - 统计每个工具的调用次数
   - 统计被阻止的调用及原因

提示：
- 风险等级决定默认行为: DESTRUCTIVE 默认需要审批
- 参数白名单示例: order_id 只允许 "ORD-\\d{5}" 格式
- 参数黑名单示例: 禁止 SQL 关键字 (DROP, DELETE, UNION 等)
- 上下文可以基于意图分类（如 "return_inquiry"）
- 审计日志对安全事件追溯至关重要
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import re
import json
from datetime import datetime

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：定义 ToolRisk 枚举
# - READ_ONLY: 只读，无副作用
# - WRITE: 写入，有副作用但可逆
# - DESTRUCTIVE: 破坏性（删除、发送、扣款）
# - SENSITIVE: 访问敏感数据
#
# 第2步：实现 ToolPermission 数据类
# - 包含上述所有字段
# - allowed_args_patterns: dict[str, str] — 参数名到正则模式的映射
# - blocked_args_patterns: dict[str, str] — 参数名到禁止模式的映射
# - requires_approval: 从风险等级自动推导默认值
#   (DESTRUCTIVE → True, SENSITIVE → True, 其他 → False)
#
# 第3步：实现 ToolGuard 类
# - __init__()
#   * _permissions: dict[str, ToolPermission]
#   * _call_counts: dict[str, int]
#   * _audit_log: list[dict]
# - register_tool(permission: ToolPermission)
# - check(tool_name, tool_args, context) → dict
#   * 执行全部 5-6 个检查
#   * 按顺序检查，第一个失败就返回
#   * 将检查结果记录到 _audit_log
#   * 通过检查后递增 _call_counts
# - _check_call_count(tool_name) → bool
# - _check_args_whitelist(tool_args, allowed_patterns) → list[str] (失败原因)
# - _check_args_blacklist(tool_args, blocked_patterns) → list[str]
# - _check_context(tool_name, context, allowed_contexts) → bool
# - get_audit_log() → dict
#   * 返回调用统计和最近 50 条日志
# - reset_session(): 重置当前会话的调用计数
#
# 第4步：编写测试
# - 创建 ToolGuard 实例
# - 注册 4-5 个不同风险等级的工具
# - 测试场景:
#   * 正常调用 → allowed
#   * 超限调用 → blocked
#   * 参数不匹配白名单 → blocked
#   * 参数匹配黑名单 → blocked
#   * 上下文不允许 → blocked
#   * 需要审批 → allowed + requires_approval=True
# - 打印审计日志

# 提示：
# - 参数检查使用 re.fullmatch（完全匹配）而非 re.search（部分匹配）
# - 对于需要审批的操作，allowed 返回 True 但 requires_approval=True
# - 调用方需要处理 requires_approval 的情况（暂停等待审批）
# - 审计日志应该包含时间戳，便于追溯
