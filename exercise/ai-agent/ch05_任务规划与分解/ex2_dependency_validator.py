"""
章节：第05章 任务规划与分解
题目：实现依赖图验证器
类型：基础练习

题目描述：
实现一个依赖图验证器，输入一个计划（子任务列表），分析依赖关系的正确性。
验证器需要能够检测循环依赖、孤儿任务、不存在的依赖引用，并识别并行执行机会。

要求：
1. 实现循环依赖检测：使用 DFS 染色法（WHITE/GRAY/BLACK）检测依赖图中的环
2. 实现孤儿任务检测：找出既不被依赖也不依赖他人的孤立任务
3. 实现依赖引用验证：检查 depends_on 中引用的任务 ID 是否都存在于任务列表中
4. 实现并行机会识别：找出具有完全相同依赖集的任务组（这些任务可以并行执行）
5. 实现拓扑排序：按依赖顺序排列任务，用于确定执行顺序
6. 编写完整的 validate(plan) 函数，返回包含所有检查结果的报告字典

提示：
- 循环检测的关键：在 DFS 过程中，如果遇到 GRAY 状态的节点，说明存在环（遇到了尚未完成遍历的祖先节点）
- 孤儿任务不一定是错误（可能表示该任务是独立的顶层工作），但应该发出警告
- 并行任务组：如果 task_B 和 task_C 的 depends_on 都是 ["task_A"]，且它们之间没有依赖，则可并行
- 拓扑排序使用 Kahn 算法（BFS）或 DFS 都可以
- 返回的报告应该包含 warnings 列表，汇总所有发现的问题
"""

from typing import Any
from collections import defaultdict, deque

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 设计数据结构：SubTask 和 Plan
#    - 可以用 dataclass 或 NamedTuple
#    - SubTask 至少包含 id, description, depends_on 字段
#    - Plan 至少包含 tasks 列表
#
# 2. 实现 detect_cycles(plan)
#    - 构建依赖图（邻接表）
#    - 使用 WHITE/GRAY/BLACK 三色标记法进行 DFS
#    - 返回检测到的所有环（每个环是一个 task_id 列表）
#    - 注意：依赖图的方向——如果 B 依赖 A，边是 A → B
#
# 3. 实现 detect_orphan_tasks(plan)
#    - 跟踪所有被引用的任务 ID（出现在任何 depends_on 中的 ID）
#    - 找出既不在 depends_on 中，自身 depends_on 也为空的任务
#    - 注意：只有一个任务时不算孤儿
#
# 4. 实现 find_parallel_groups(plan)
#    - 按 depends_on 的排序元组对任务分组
#    - depends_on 完全相同且不互相依赖的任务属于同一并行组
#    - 返回包含多个任务的组
#
# 5. 实现 topological_sort(plan)
#    - Kahn 算法：计算入度，零入度入队，逐步弹出
#    - 如果排序后任务数少于总数 → 存在循环依赖
#    - 返回拓扑排序后的 task_id 列表
#
# 6. 实现 validate(plan) 主函数
#    - 调用以上所有检查函数
#    - 返回包含所有发现的报告字典：
#      { "total_tasks": int, "cycles": [...], "orphans": [...],
#        "parallel_groups": [...], "unresolved_deps": [...],
#        "topological_order": [...], "warnings": [...] }
#
# 7. 编写测试用例
#    - 正常计划（无问题）
#    - 有循环依赖的计划
#    - 有过早依赖/孤儿任务的计划
#    - 有并行机会的计划
#
# 提示：
# - 循环检测中注意：依赖图的"边方向"定义为 A→B 表示 B 依赖 A
# - 或者 A→B 表示 A 依赖 B 都可以，但要统一且注释清楚
# - 孤儿任务检测要先明确什么算"孤儿"，然后在测试用例中验证
# - 拓扑排序的结果可用于后续的执行顺序编排
