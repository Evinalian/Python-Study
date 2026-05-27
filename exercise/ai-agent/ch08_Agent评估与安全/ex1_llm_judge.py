"""
章节：第08章 Agent评估与安全
题目：实现 LLM-as-Judge 评估器
类型：基础练习

题目描述：
实现一个基于 LLM 的自动评估器，用强模型（GPT-4o）评估 Agent 回复的质量。
这是 Agent 评估中最常用的方法——让一个强大的 LLM 充当"评委"。

要求：
1. 实现 LLMJudge 类：
   - __init__(judge_model="gpt-4o"): 初始化 Judge
   - evaluate(user_task, agent_output, criteria, reference_answer):
     调用 Judge LLM 评估 Agent 回复
   - 返回结构化的评估结果（JSON 格式）
2. 设计评估 prompt，包含至少 4 个评估维度：
   - 任务完成度：是否完成了用户要求的事项
   - 准确性：信息是否正确
   - 清晰度：表达是否清晰易懂
   - 完整性：是否遗漏重要信息
3. 每个维度独立评分（1-5 分），有具体的评价文字
4. 最后综合评分（1-10 分）和整体判定（pass/needs_improvement/fail）
5. 实现 batch_evaluate 方法，对多个测试用例批量评估
6. 生成评估报告摘要（各维度平均分、通过率）

提示：
- 使用 response_format={"type": "json_object"} 确保 Judge 返回结构化 JSON
- Judge 的 temperature 设为 0.0 确保评估一致性
- 评估 prompt 中要明确说明评分标准（什么是 5 分、什么是 1 分）
- 考虑提供参考答案让 Judge 对比评估（可选，能提升评估准确性）
- Judge 也可能有偏见，定期抽样人工评估做校准
"""

import os
import json
from typing import Any, Optional

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：实现 LLMJudge 类
# - __init__: 初始化 OpenAI client 和 judge_model
# - evaluate(user_task, agent_output, criteria=None, reference_answer=None):
#   * 构建评估 prompt（System + User）
#   * 在 User prompt 中包含：用户任务、Agent 回复、评估标准、参考答案（可选）
#   * 调用 client.chat.completions.create
#   * 使用 response_format={"type": "json_object"}
#   * 解析返回的 JSON 为 dict
#   * 返回评估结果
#
# 第2步：设计默认评估标准（至少 4 个）
# - task_completion: 任务完成度 (1-5)
#   5分=完美完成所有要求, 3分=部分完成, 1分=基本未完成
# - accuracy: 信息准确性 (1-5)
#   5分=所有信息正确, 3分=有小错误, 1分=严重错误
# - clarity: 表达清晰度 (1-5)
#   5分=结构清晰逻辑流畅, 3分=基本清晰, 1分=难以理解
# - completeness: 信息完整性 (1-5)
#   5分=无遗漏, 3分=轻微遗漏, 1分=关键信息缺失
#
# 第3步：设计评估 prompt 模板
# - System prompt: "你是一个严格的 Agent 评估专家..."
# - User prompt: 包含任务、回复、标准，要求输出结构化 JSON
# - 在 prompt 中明确说明评分标准
# - 要求给出 strengths 和 weaknesses
#
# 第4步：实现 batch_evaluate 方法
# - 输入: test_cases 列表，每个包含 task/agent_output/reference_answer(可选)
# - 对每个 case 调用 evaluate
# - 汇总所有结果
# - 计算各维度平均分
# - 计算整体通过率
#
# 第5步：测试
# - 准备 3-5 个 Agent 回复样本（可以手动构造）
# - 包含好的回复和差的回复
# - 运行评估
# - 验证评分是否合理（好回复分数高于差回复）

# 前置准备：
# - 需要安装的包：openai, python-dotenv
# - 需要设置的环境变量：OPENAI_API_KEY
