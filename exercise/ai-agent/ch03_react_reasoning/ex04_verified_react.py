"""
练习 3-4（进阶）: 实现带验证的 ReAct Agent

在基础 ReAct Agent 的基础上，添加验证机制。

TODO:
1. 修改第 3 章的 parse_react_output 函数或 run_react_agent 函数，
   在每个 Observation 之后，增加一个验证步骤：
   - LLM 需要输出 "Confidence: [高/中/低]" 来评估搜索结果的可靠性
   - 评估维度：信息来源是否可靠、信息是否足够具体、信息是否直接回答问题

2. 实现验证后的自动响应逻辑：
   - Confidence 为"高"：继续正常流程
   - Confidence 为"中"：记录这一轮的搜索结果，但尝试用不同关键词再搜一次
   - Confidence 为"低"：放弃当前搜索结果，用完全不同的搜索策略重试
   - 如果连续 3 次 Confidence 为"低"：告诉用户无法找到可靠信息，
     并给出基于已有最佳信息的回答

3. 创建以下测试场景：
   a) 搜索一个常见话题（Confidence 应为"高"）
   b) 搜索一个冷门话题（模拟返回空结果或不相关信息，Confidence 应为"低"）
   c) 搜索一个有歧义的话题（返回部分相关部分不相关的结果，Confidence 应为"中"）

4. 测试并记录：
   - 每个场景下 Agent 的 Confidence 判断是否准确
   - 当 Confidence 为"低"时，Agent 是否进行了有效的重试
   - "连续 3 次低 Confidence"的兜底机制是否正常工作

提示：
- 需要在 system prompt 中加入 Confidence 的输出格式要求和评估标准
- 修改 parse_react_output 来解析 Confidence 字段
- 思考：LLM 对自己"不知道什么"的判断能力如何？它能否正确评估搜索结果的可靠性？
"""


# 请在此处编写你的代码


if __name__ == "__main__":
    # TODO: 实现带验证的 ReAct Agent 并测试
    pass
