"""
练习 3-5（进阶）: 实现 ReAct 循环的可视化追踪

实现一个函数 visualize_react_loop(history)，生成文本形式的可视化追踪。

TODO:
1. 实现 visualize_react_loop(history) 函数，参数 history 是一个列表，
   每个元素包含: {"thought": str, "action_raw": str, "observation": str, "confidence": str}

2. 可视化输出要求：
   a) 使用缩进和箭头符号清晰展示步骤层级：
      第1轮
        Thought → [思考内容的前100字符...]
          Action → search("北京天气")
            Observation → [搜索结果的前100字符...]
      第2轮
        Thought → ...
          Action → ...
            Observation → ...

   b) 长文本截断：Thought 和 Observation 超过 100 字符时截断并加 "..."
      但 action_raw 全量展示（通常较短）

   c) 异常高亮：以下情况用 [!!] 标记：
      - 工具调用失败（Observation 包含 "错误" 或 "error"）
      - 连续两轮使用了相同的 Action（重复调用同一工具同一参数）
      - 循环超过 5 轮仍未结束

   d) 统计摘要（在末尾输出）：
      ┌─────────────────────────────┐
      │ ReAct 循环统计              │
      ├─────────────────────────────┤
      │ 总循环次数:   5             │
      │ 有效步骤:     4             │
      │ 重复步骤:     1             │
      │ 工具调用失败: 0             │
      │ 平均 Confidence: 中         │
      └─────────────────────────────┘

3. 使用第 3 章的 run_react_agent 收集实际的 history 数据，
   测试你的可视化函数。

4. 额外挑战：用 Unicode 字符画一个时间线（timeline）风格的追踪图。
"""


# 请在此处编写你的代码


if __name__ == "__main__":
    # TODO: 实现可视化追踪并测试
    pass
