"""
章节：第05章 任务规划与分解
题目：实现一个基于 LLM 的简单规划器
类型：基础练习

题目描述：
实现一个基于 LLM 的规划器，将用户的自然语言任务分解为结构化的执行计划。
规划器接收用户任务和可用工具列表，输出一组包含依赖关系的子任务（JSON 格式）。

要求：
1. 定义至少 5 个可用工具（如查询数据、汇总、分析、发邮件等），每个工具有名称、描述、参数 schema
2. 设计规划器的 System Prompt，包含：角色设定、可用工具列表、规划原则、输出格式（JSON Schema）
3. 调用 LLM（使用 response_format={"type": "json_object"}）生成计划
4. 将 LLM 返回的 JSON 解析为 Python 数据结构
5. 打印计划摘要（子任务列表、依赖关系、并行机会）

提示：
- System Prompt 要明确要求 LLM 标注每个子任务的 depends_on 字段
- 规划原则包括：粒度控制（一个子任务 = 一次工具调用）、依赖分析、并行识别
- 在 prompt 中包含至少一个完整的计划示例（few-shot），让 LLM 理解期望的格式
- 使用 temperature=0.0 确保输出稳定
- 检查生成的计划中 depends_on 引用的任务 ID 是否都存在于 tasks 列表中
"""

import os
import json
import sys

# 提示：你需要导入 openai 库并初始化客户端
# from openai import OpenAI
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=...)

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义 AVAILABLE_TOOLS 字典，包含至少 5 个工具
#    每个工具需包含：description（功能描述）、parameters（参数 JSON Schema）
#    工具示例：get_data, process_data, analyze, generate_report, send_notification
#
# 2. 设计 build_planner_prompt() 函数，生成规划用的 System Prompt
#    必须包含：
#    - 角色设定（"你是一个任务规划专家..."）
#    - 工具列表（从 AVAILABLE_TOOLS 生成）
#    - 规划原则（粒度控制、依赖分析、并行识别）
#    - 输出格式（严格 JSON Schema，包含 tasks 数组）
#    - 至少一个完整的 Few-shot 示例计划
#
# 3. 实现 generate_plan(user_task, context) 函数
#    - 调用 LLM 的 chat.completions.create
#    - 使用 response_format={"type": "json_object"}
#    - 设置 temperature=0.0
#    - 解析返回的 JSON
#
# 4. 实现 print_plan_summary(plan_dict) 函数
#    - 打印每个子任务的 ID、描述、动作、依赖关系
#    - 识别并打印可以并行执行的任务组
#
# 5. 编写测试用例
#    - 用 2-3 个不同的用户任务测试规划器
#    - 观察 LLM 如何根据任务复杂度调整子任务数量
#
# 前置准备：
# - 需要安装的包：openai, python-dotenv
# - 需要设置的环境变量：OPENAI_API_KEY
#
# 提示：
# - 不要硬编码 API Key，始终用 os.environ.get("OPENAI_API_KEY") 读取
# - prompt 中包含 "JSON" 关键字才能使用 response_format={"type": "json_object"}
# - 对生成的计划做基本验证：检查 depends_on 引用的 task_id 是否存在
