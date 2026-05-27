"""
章节：第06章 多智能体协作
题目：用 CrewAI 构建一个代码审查团队
类型：进阶练习

题目描述：
使用 CrewAI 框架搭建一个专业的代码审查团队。团队由多个专业 Agent 组成，
各自从不同维度审查代码，最后由汇总 Agent 整合审查报告。

要求：
1. 安装并导入 CrewAI：
   pip install crewai
2. 创建以下专业 Agent：
   a. 代码质量审查员 (Code Quality Reviewer):
      - 检查代码风格 (PEP8 / Black)
      - 检查命名规范
      - 检查函数/类的职责是否单一
      - 检查注释和文档是否充分
   b. 安全审查员 (Security Reviewer):
      - 检查 SQL 注入风险
      - 检查 XSS/CSRF 漏洞
      - 检查敏感信息泄露（硬编码密钥等）
      - 检查依赖库的安全版本
   c. 性能审查员 (Performance Reviewer):
      - 检查算法复杂度
      - 检查不必要的 I/O 操作
      - 检查内存泄漏风险
      - 建议优化方案
   d. 汇总报告员 (Report Compiler):
      - 整合所有审查员的发现
      - 按严重程度分类（CRITICAL / HIGH / MEDIUM / LOW）
      - 生成结构化的审查报告
      - 给出总体评价和修改建议优先级
3. 使用层级执行模式 (Process.hierarchical)：
   - Manager Agent 负责协调整个审查流程
   - Manager 将代码分发给各审查员
   - 审查完成后由汇总报告员整合
4. 实现审查报告结构化输出：
   - 使用 Pydantic 模型定义输出格式
   - 包含总体评分、问题列表、改进建议
5. 测试：
   - 用一段包含多种问题的 Python 代码作为输入
   - 运行团队并观察各 Agent 的输出
   - 检查审查报告是否全面、准确

提示：
- CrewAI 的层级模式需要一个 Manager Agent（role 以 "Manager" 结尾）
- 各审查 Agent 的 task 可以设置 async_execution=True 实现并行审查
- 输出可以用 output_pydantic 约束格式
- 注意设置 allow_delegation=False 防止 Worker Agent 互相委托任务
- LLM 配置使用 config_list 格式
"""

import os
from typing import Optional

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：安装依赖
# pip install crewai pydantic
# 确保 OPENAI_API_KEY 环境变量已设置
#
# 第2步：定义输出数据结构（Pydantic）
# - class ReviewIssue: 单个问题
#   * file: 文件名/位置
#   * line: 行号（可选）
#   * severity: CRITICAL / HIGH / MEDIUM / LOW
#   * category: "code_quality" / "security" / "performance"
#   * description: 问题描述
#   * suggestion: 修改建议
# - class ReviewReport: 完整审查报告
#   * overall_score: 总体评分 (1-10)
#   * summary: 总体评价
#   * issues: List[ReviewIssue]
#   * strengths: 代码优点列表
#   * improvement_priority: 改进优先级列表
#
# 第3步：创建 Manager Agent
# - role 以 "Manager" 结尾（CrewAI 约定）
# - goal: "协调代码审查团队，确保全面审查和安全交付"
# - backstory: 技术主管背景，丰富的代码审查经验
# - allow_delegation=True
#
# 第4步：创建专业审查 Agent
# 对每个 Agent 定义：
# - role: 专业角色名称
# - goal: 该审查员的审查目标
# - backstory: 专业背景和经验
# - allow_delegation=False
# - verbose=True
#
# 第5步：定义 Task
# - Manager Task: 接收代码 → 分发给审查员 → 等待汇总
#   注意：这是唯一的顶层 Task，Manager 会自己分解
# - 在 description 中明确说明审查流程和各审查员的职责
#
# 第6步：组装 Crew
# - agents: [manager, code_quality, security, performance, compiler]
# - tasks: [manager_task]
# - process: Process.hierarchical
# - manager_agent: manager
# - verbose: True
#
# 第7步：准备测试代码
# - 编写一段约 100-200 行的 Python 代码
# - 刻意加入一些常见问题：
#   * SQL 拼接（安全风险）
#   * 硬编码密钥（安全风险）
#   * O(n²) 算法（性能问题）
#   * 变量命名不规范（质量问题）
#   * 缺少文档注释（质量问题）
#
# 第8步：运行和评估
# - crew.kickoff() 启动团队
# - 检查输出的审查报告是否全面覆盖了所有问题类型
# - 检查各 Agent 是否真正并行工作（查看日志时间戳）

# 提示：
# - CrewAI 层级模式中，只需要一个顶层 Task
# - Manager 会自动创建子 Task 分配给 Worker
# - 审查员的 Task 可以不用显式定义（Manager 动态分配）
# - 但也可以在 agents 的 goal 中描述审查维度
# - Manager LLM 的 temperature 设置低一些（0.0-0.2）确保分配决策稳定
