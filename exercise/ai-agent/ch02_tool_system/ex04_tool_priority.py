"""
练习 2-4（进阶）: 实现工具调用的优先级排序

有时多个工具都可以解决同一个问题。设计一个优先级机制，引导 LLM 选择更合适的工具。

TODO:
1. 为 Tool 数据类添加 priority 字段（1-10，数字越小优先级越高）

2. 创建两组功能重叠的工具，例如：
   - search_web (priority=2, 通用网页搜索) vs specialized_weather_api (priority=1, 专用天气 API)
   - 两者都能查天气，但 specialized_weather_api 更准确
   - generic_calculator (priority=3) vs scientific_calculator (priority=2) vs math_engine (priority=1)

3. 实现两种优先级引导策略：
   策略 A: 在 system prompt 中按优先级列出工具推荐顺序
   策略 B: 为高优先级工具在 description 中加入 "推荐优先使用" 标记

4. 编写测试：
   - 用户问 "北京天气怎么样？" —— 验证 Agent 优先使用 specialized_weather_api
   - 用户问 "计算 sin(30°) + cos(60°)" —— 验证 Agent 优先使用 math_engine
   - 用户问 "今天有什么新闻？" —— search_web 应该被使用（专用工具不适用）

5. 记录测试结果，分析：
   - 哪种引导策略（A 或 B）更有效？
   - 优先级机制是否会影响 LLM 的灵活性（当高优先级工具不适用时）？

提示：
- 可以在 system prompt 中加入类似 "工具推荐顺序: math_engine > scientific_calculator > generic_calculator"
- 注意不要让优先级机制过于"强硬"，LLM 仍应能根据实际情况灵活选择
"""


# 请在此处编写你的代码


if __name__ == "__main__":
    # TODO: 实现优先级机制并测试
    pass
