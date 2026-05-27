"""
章节：第08章 Agent评估与安全
题目：实现 Prompt Injection 检测器
类型：基础练习

题目描述：
实现一个多层 Prompt Injection 检测和防护系统。Prompt Injection 是 Agent 面临
的最严重的安全威胁——攻击者试图通过构造特殊输入来覆盖 Agent 的系统指令。

要求：
1. 第一层：规则匹配
   - 至少 8 种常见的注入模式（正则表达式）
   - 包括：直接指令覆盖、系统角色伪造、越狱尝试、角色扮演等
   - 返回匹配的模式列表和风险等级
2. 第二层：LLM 安全审查
   - 用一个独立的 LLM（不同于 Agent 的 LLM）审查输入安全性
   - LLM 审查 prompt 重点关注：指令覆盖、权限提升、恶意操作、信息窃取
   - 返回 is_safe / risk_level / concerns
3. 实现输入清洗函数：
   - 移除/替换明显的注入模式
   - 保留原始和清洗后的版本以供对比
4. 整合为 PromptInjectionGuard 类：
   - check(input): 执行全部检查，返回 {is_safe, risk_level, violations, sanitized_input}
   - 支持配置各层的开关和阈值
5. 测试至少 8 种样本：
   - 4 种安全的正常输入
   - 4 种恶意的注入尝试（覆盖不同注入类型）

提示：
- 规则匹配用 re 模块，编译正则表达式提高性能
- 常见的注入模式包括：
  * "Ignore all previous instructions"
  * "[SYSTEM] You are now..."
  * "Pretend you are DAN..."
  * "Forget your training..."
  * 非法 Unicode 字符混淆
- LLM 安全审查的 prompt 设计要独立于 Agent 的 system prompt
- 输入清洗不是简单地删除：要保留可读性
- 被清洗的输入应该标记为"已被修改"
"""

import re
import os
import json
from typing import Any, Optional

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：定义注入检测模式
# - INJECTION_PATTERNS: 正则表达式列表
#   至少包含以下 8 种类型：
#   * 忽略指令: "ignore (all )?(previous|prior|above) instructions?"
#   * 忘记训练: "forget (your )?(training|instructions)"
#   * 覆盖系统: "override (system )?(prompt|instructions)"
#   * 角色扮演: "pretend (you are|to be)"
#   * DAN 攻击: "act as (if )?DAN"
#   * 系统伪造: r"\[SYSTEM\]" 或 "<system>"
#   * 越狱: "jailbreak|bypass (restrictions|filters)"
#   * 新角色: "you are now \w+ mode"
# - 模式不区分大小写
#
# 第2步：实现规则匹配函数
# - detect_patterns(user_input) → {"is_attack": bool, "matched": [...], "risk": str}
# - risk 计算: 1-2 个匹配 = "low", 3-4 = "medium", 5+ = "high"
# - 使用 re.search 逐个匹配
#
# 第3步：实现输入清洗函数
# - sanitize(user_input) → (cleaned_input, was_modified)
# - 策略:
#   * 移除 [SYSTEM]、[PROMPT] 等标签
#   * 替换已知的注入关键词为 [FILTERED]
#   * 截断异常长的输入（> 2000 字符）
# - 标记是否被修改
#
# 第4步：实现 LLM 安全审查
# - llm_safety_check(user_input, client) → dict
# - 设计审查 prompt:
#   * 分析输入是否包含注入/越狱企图
#   * 分析输入是否请求超出正常权限的操作
#   * 返回 is_safe, risk_level, concerns
# - 使用 response_format={"type": "json_object"}
#
# 第5步：实现 PromptInjectionGuard 类
# - __init__(use_llm_check=True, llm_client=None)
# - check(user_input):
#   * 第1层: 规则匹配
#   * 第2层: 输入清洗
#   * 第3层: LLM 审查（可选）
#   * 汇总结果: {is_safe, risk_level, original, sanitized, violations}
# - 如果任一层的风险为 high → 直接标记为 unsafe
#
# 第6步：测试
# - test_safe_inputs: 4 种正常客服查询
# - test_malicious_inputs: 4 种注入尝试
# - 验证检测器的准确率（不应误判正常输入）

# 提示：
# - 注入模式可能以各种编码/格式出现，纯规则匹配有局限
# - LLM 审查更好但更贵，建议先用规则快速过滤，可疑的再送 LLM
# - 这个检测器不能单独运行，需要集成到 Agent 的输入管道中
