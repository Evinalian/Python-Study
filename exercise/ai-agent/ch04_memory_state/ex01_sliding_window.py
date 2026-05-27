"""
练习 4-1: 实现滑动窗口策略

实现一个 SlidingWindowMemory 类，管理对话历史的 token 限制。

TODO:
1. 实现 SlidingWindowMemory 类：

   a) __init__(self, max_tokens: int = 4000)
      - max_tokens: messages 列表允许的最大 token 数

   b) add_message(self, message: dict)
      - 添加一条消息到内部消息列表

   c) get_messages(self) -> List[dict]
      - 返回当前的消息列表
      - 返回前自动调用 _enforce_limit() 确保 token 数不超限

   d) _enforce_limit(self)
      - 计算 messages 的总 token 数
      - 如果没有 tiktoken，使用简单估算:
        中文: 1 字符 ≈ 1.5 token
        英文: 1 字符 ≈ 0.3 token
        混合: 字符数 * 0.8 (粗略折中)
      - 注意：精确估算时，OpenAI 的一条消息还有固定的 overhead token
        (每条消息约 3-4 token 的格式开销，role 字段等)
      - 如果总 token 数超过 max_tokens:
        * 保留 system message（永远不删）
        * 从最老的非 system message 开始丢弃
        * 直到 token 数降到 max_tokens 的 80% 以下
        * 丢弃后，在 system message 之后插入一条提示消息:
          "[注意: 由于上下文限制，部分早期对话已被省略。共省略了 N 条消息。如有需要，请询问。]"

   e) estimate_tokens(self, text: str) -> int
      - 估算一段文本的 token 数

2. 测试：
   - 创建一段长对话（至少 20 条消息），包含中文和英文内容
   - 分别设置 max_tokens=2000 和 max_tokens=8000，观察裁剪差异
   - 验证 system message 是否始终被保留
   - 验证裁剪提示消息是否正确显示省略的消息数量

3. 分析（作为注释）：
   - 你的估算方法的准确性如何？和实际 token 数可能有多大的偏差？
   - 除了简单丢弃最老的消息，还有什么更好的裁剪策略？
"""


# 请在此处编写你的代码


if __name__ == "__main__":
    # TODO: 实现 SlidingWindowMemory 并测试
    pass
