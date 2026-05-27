"""
章节：第06章 多智能体协作
题目：Agent 角色匹配系统
类型：基础练习

题目描述：
实现一个基于能力画像的 Agent-任务匹配系统。在多 Agent 系统中，协调者需要
将每个子任务分配给最合适的 Agent。这个系统就是帮助做出这个决策的。

要求：
1. 设计 AgentProfile 数据类：
   - name, title, background（身份信息）
   - primary_duties, secondary_duties（职责）
   - knowledge_domains, skill_level（能力）
   - out_of_scope, escalation_policy（边界）
   - communication_style, preferred_collaborators（协作偏好）
   - output_format, output_detail_level（输出偏好）
2. 实现 match_score(profile, task_description) 方法：
   - 基于知识领域的匹配（高权重）
   - 基于主要职责的匹配（中权重）
   - 基于出界检测（负权重，匹配到 out_of_scope 则大幅降分）
   - 基于专业等级加权（skill_level 越高，匹配分越高）
3. 实现 find_best_agent(profiles, task_description, top_n) 方法：
   - 对所有 Profile 计算匹配分
   - 返回排名前 N 的结果
   - 如果最高分 < 阈值，返回"无合适 Agent"建议
4. 实现 profile_to_system_prompt(profile) 方法：
   - 将 Profile 转换为结构化的 System Prompt
   - 包含身份、职责、能力、边界、协作偏好等所有要素
5. 实现角色重叠检测：
   - 输入两个 Profile，计算它们的职责重叠度
   - 高重叠度意味两个 Agent 可能产生任务冲突
6. 创建至少 5 个不同类型的 Agent Profile 并用多种任务测试匹配效果

提示：
- 匹配分计算可以使用简单的关键词匹配（基础），也可以使用 embedding 相似度（进阶）
- 权重分配建议：知识领域 40%，职责 30%，工具匹配 20%，风格匹配 10%
- 角色重叠检测用 Jaccard 相似度（两个集合交集/并集）
- out_of_scope 匹配应产生显著的负分，确保边界清晰
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：定义 CommunicationStyle 枚举
# - CONCISE, DETAILED, SKEPTICAL, SUPPORTIVE
#
# 第2步：实现 AgentProfile 数据类
# - 包含完整的六组字段（身份、职责、能力、边界、协作、输出）
# - 实现 to_system_prompt() 方法，将所有字段组合成结构化的 System Prompt
# - 输出示例参考第4节中的模板
#
# 第3步：实现 match_score(profile, task_description) → float
# - 分词：将 task_description 转为小写，提取关键词
# - knowledge_domains 匹配：每个匹配的领域 +2.0 分
# - primary_duties 匹配：每个匹配的职责关键字 +1.0 分
# - skill_level 加权：高技能等级的领域匹配时额外加分
# - out_of_scope 检测：如果任务关键词匹配到 out_of_scope 项，-5.0 分
# - 确保分数不会低于 0
#
# 第4步：实现 find_best_agent(profiles, task_description, top_n=1) → list[tuple[AgentProfile, float]]
# - 计算所有 Profile 的匹配分
# - 按分数降序排列
# - 返回前 top_n 个
# - 如果最高分 < 阈值（如 2.0），打印警告
#
# 第5步：实现 detect_role_overlap(profile_a, profile_b) → dict
# - 计算 primary_duties 的 Jaccard 相似度: |A ∩ B| / |A ∪ B|
# - 计算 knowledge_domains 的 Jaccard 相似度
# - 计算 tools 的 Jaccard 相似度
# - 返回重叠度报告: {"duty_overlap": 0.6, "knowledge_overlap": 0.4, ...}
#
# 第6步：创建角色模板库
# - 至少 5 个预定义的 AgentProfile（如数据分析师、写作者、审查员、开发者、设计师）
# - 每个 Profile 要真实可信，有具体的背景和能力描述
#
# 第7步：测试
# - 用不同类型的任务测试匹配系统
# - 用重叠检测分析角色库中是否存在冗余角色
# - 打印匹配结果和角色重叠报告

# 提示：
# - Jaccard 相似度 = len(交集) / len(并集)
# - 匹配时把 out_of_scope 的负分设得足够大，确保不会被分配到不合适的 Agent
# - 在 to_system_prompt 中使用清晰的 Markdown 格式（H2 标题分隔各部分）
# - 重叠度 > 0.5 意味着两个角色可能可以合并
