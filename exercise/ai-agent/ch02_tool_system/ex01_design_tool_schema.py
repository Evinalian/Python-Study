"""
练习 2-1: 设计一个完整的工具 Schema

为你选择的三个工具分别设计完整的 JSON Schema。

TODO:
1. 选择三个工具，例如：
   - 天气查询 (query_weather)
   - 新闻搜索 (search_news)
   - 单位换算 (convert_unit)

2. 为每个工具设计完整的 JSON Schema，要求：
   - description 包含适用场景和不适用场景（不少于 50 字）
   - 参数类型选择合理，添加适当的约束条件
   - query_weather 的参数至少包含 city (string) 和 date (string, 可选)
   - search_news 至少包含一个 enum 类型的参数（如 category: 科技/财经/体育/娱乐）
   - convert_unit 至少包含一个 array 类型的参数或多个数值参数

3. 写一段文字说明你的设计思路（作为注释或输出），特别是：
   - description 中为什么那样写（哪部分是告诉 LLM "该用"，哪部分是告诉 LLM "不该用"）
   - 为什么选这些类型和约束（为什么某参数用 enum 而不是 string？）
   - 你认为这个 Schema 有什么可以改进的地方？

4. 额外挑战：写一个简单的测试，用你的 Schema 构造一个 OpenAI Function Calling 请求，
   测试 LLM 是否能正确理解并填充参数。
"""


# 请在此处编写你的代码


if __name__ == "__main__":
    # TODO: 设计 Schema 并测试
    pass
