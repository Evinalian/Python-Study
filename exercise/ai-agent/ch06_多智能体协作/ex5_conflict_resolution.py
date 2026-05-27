"""
章节：第06章 多智能体协作
题目：实现冲突检测和解决系统
类型：进阶练习

题目描述：
实现一个多 Agent 冲突检测和解决系统。在多 Agent 协作中，冲突是常态而非例外：
Agent 可能得出结论矛盾、职责重叠导致推诿、或陷入无限讨论循环。
这个系统负责检测这些冲突并自动或半自动地解决它们。

要求：
1. 实现循环检测器 (LoopDetector):
   - 记录 Agent 间的消息交互模式
   - 检测同一主题的高频来回
   - 检测消息内容的相似度（判定是否在"重复讨论"）
   - 当检测到循环时发出警告并触发打破机制
2. 实现冲突解决策略：
   a. 协调者仲裁 (CoordinatorDecides):
      - 将所有相关 Agent 的立场和论据收集起来
      - 协调者（或另一个中立 Agent）做最终决定
   b. 加权投票 (WeightedVote):
      - 根据每个 Agent 与冲突主题的专业相关度设定投票权重
      - 加权计算最终决定
   c. 重审与修订 (ReviseAndRetry):
      - 要求冲突双方重新审视自己的立场
      - 提供新的视角或补充信息
      - 尝试第二轮讨论
3. 实现职责重叠检测:
   - 比较两个 Agent Profile 的主要职责集合
   - 使用 Jaccard 相似度计算重叠度
   - 高重叠（>0.5）建议合并或重新划分边界
4. 实现冲突日志和审计:
   - 记录每次冲突的完整信息
   - 记录采用的解决策略和结果
   - 支持按 Agent、按类型、按时间段查询
5. 编写测试场景:
   - 场景A: 结论矛盾（两个 Agent 对同一数据得出相反结论）
   - 场景B: 职责重叠（两个 Agent 争抢同一个任务）
   - 场景C: 无限循环（两个 Agent 在某个细节上来回讨论）
   - 对每个场景应用合适的解决策略并验证效果

提示：
- 循环检测的关键指标：同一 topic 的来回次数、消息内容的余弦相似度
- 加权投票时，专业相关度可以用 Agent Profile 中的 skill_level 或 knowledge_domains 匹配度来确定
- 职责重叠检测是预防性措施：在分配任务前就检测，避免冲突发生
- 冲突日志使用 dataclass 结构化存储，支持 JSON 序列化导出
"""

import json
import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：定义数据结构
# - ConflictType 枚举: CONCLUSION_CONFLICT, ROLE_OVERLAP, DEADLOCK_LOOP, RESOURCE_CONFLICT
# - ResolutionStrategy 枚举: COORDINATOR_ARBITRATION, WEIGHTED_VOTE, REVISE_AND_RETRY, ESCALATE_TO_HUMAN
# - ConflictRecord dataclass: id, type, agents_involved, description, strategy_used, resolution, timestamp, resolved
# - AgentPosition dataclass: agent_name, stance, reasoning, confidence
#
# 第2步：实现 LoopDetector 类
# - __init__(max_rounds=5, similarity_threshold=0.8)
# - check(sender, receiver, content, topic) → {"is_looping": bool, "rounds": int, "warning": str}
#   * 消息历史维护: 按 (sender, receiver, topic) 三元组分组
#   * 同一 topic 的回合数 = 该 topic 下消息数 / 2
#   * 如果回合数 > max_rounds → 判定为循环
# - _calculate_similarity(msg1, msg2) → float:
#   * 简化实现：提取关键词集合，计算 Jaccard 相似度
# - break_loop(sender, receiver, topic) → str:
#   * 返回打破循环的建议
#
# 第3步：实现 ConflictResolver 类
# - __init__()
# - resolve(conflict_type, agents_positions, context, strategy) → ConflictRecord
# - _coordinator_arbitration(positions, context):
#   * 调用 LLM（或使用规则逻辑）做仲裁
#   * 输出: 仲裁结果 + 理由
# - _weighted_vote(positions, weights, context):
#   * positions: {agent_name: stance}
#   * weights: {agent_name: weight}
#   * 计算加权结果
# - _revise_and_retry(agents, positions, context):
#   * 给双方提供新的视角（调用 LLM）
#   * 要求重新评估
#   * 收集第二轮意见
#
# 第4步：实现 RoleOverlapDetector 类
# - detect_overlap(profile_a, profile_b) → dict:
#   * primary_duties: Jaccard(A, B)
#   * knowledge_domains: Jaccard(A, B)
#   * tools: Jaccard(A, B)
#   * overall_overlap: 综合得分
# - suggest_merge(profile_a, profile_b, overlap_report) → str:
#   * 如果重叠度 > 0.6，建议合并
#   * 如果重叠度在 0.3-0.6，建议明确边界
#   * 如果重叠度 < 0.3，无需处理
#
# 第5步：实现 ConflictLogger 类
# - __init__(): 初始化 conflicts 列表
# - log(record): 记录冲突
# - query(agent_name=None, conflict_type=None, time_range=None) → list[ConflictRecord]
# - summary() → dict: 统计各类型冲突数量、解决率
# - export_json(path): 导出审计日志
#
# 第6步：编写测试场景
# - 场景A 结论矛盾:
#   创建两个 Agent，分别给出相反的分析结论
#   测试加权投票策略（根据专业度加权）
# - 场景B 职责重叠:
#   创建两个高重叠度的 Profile
#   测试重叠检测和建议
# - 场景C 无限循环:
#   模拟 8 轮来回
#   测试循环检测和打断
# - 场景D 人为升级:
#   测试 ESCALATE_TO_HUMAN 策略（生成给人类的冲突摘要）

# 提示：
# - Jaccard 相似度用于比较两个集合，值在 [0, 1] 之间
# - 消息相似度可以用简单的词袋模型或 TF-IDF
# - 冲突解决中的"仲裁"可以简化为规则（如选择置信度更高的 Agent 的意见）
# - 在实际项目中，"升级给人类"意味着暂停执行并向用户展示冲突信息
# - 确保所有冲突记录包含足够的信息供审计
