"""
练习 3-1: 手动追踪 ReAct 循环

给定以下场景，手动写出一个 ReAct 循环的完整 Thought-Action-Observation 序列。

场景：
用户问题: "请帮我查一下 Python 3.12 相比 3.11 有哪些重要的新特性，
          并给出一个你认为最实用的特性示例。"

TODO:
1. 不写代码，用文字描述一个完整的 ReAct 循环过程
   要求至少包含 3 轮 Thought-Action-Observation 循环
   最后给出 Final Answer

2. 每一轮中，详细写出：
   - Thought: LLM 在这一步分析什么？为什么决定采取这个 Action？
   - Action: 调用什么工具？参数是什么？
   - Observation: 工具返回了什么（模拟结果，但要合理真实）

3. 在 Final Answer 之后，写一段 100 字以上的分析（作为注释），讨论：
   - 为什么 LLM 在某一轮选择了搜索而不是直接回答？
   - 在第几轮 LLM 有了足够的信息来给出 Final Answer？为什么？
   - 如果 LLM 的第一轮搜索返回了空结果，后续几轮会怎么调整？

提示：
- 可以把 Thought / Action / Observation 写成带缩进的文本格式
- 思考：搜索关键词的选择本身就是一种"推理" —— 为什么搜 A 而不是 B？
"""


# ============================================================
# 以下是你需要填充的 ReAct 循环追踪
# ============================================================

# TODO: 在下方用注释写出完整的 ReAct 循环（包括至少 3 轮 + Final Answer）

# 用户问题: "请帮我查一下 Python 3.12 相比 3.11 有哪些重要的新特性，
#           并给出一个你认为最实用的特性示例。"
#
# --- 第 1 轮 ---
# Thought:
#
# Action:
#
# Observation:
#
# --- 第 2 轮 ---
# Thought:
#
# Action:
#
# Observation:
#
# --- 第 3 轮 ---
# Thought:
#
# Action:
#
# Observation:
#
# --- Final Answer ---
# ...


if __name__ == "__main__":
    # TODO: 打印你的分析输出
    pass
