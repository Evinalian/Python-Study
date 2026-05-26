"""
练习 1: 单工具天气助手

场景:
    实现一个天气查询助手，包含两个工具：
    1. get_current_weather: 查询当前天气
    2. get_forecast: 查询未来天气

要求:
    1. 为两个工具设计完整的 JSON Schema（name, description, parameters）
       - get_current_weather 参数: city (必填), unit (选填, celsius/fahrenheit)
       - get_forecast 参数: city (必填), days (选填, 1-7)

    2. 实现模拟的天气函数:
       - get_current_weather 返回模拟数据（如 {"city": "...", "temperature": ..., "condition": "..."}）
       - get_forecast 返回模拟列表数据

    3. 实现 run_conversation(user_query) 函数:
       - 完整的 Function Calling 循环
       - 支持模型在一次回复中调用 0、1 或多个工具
       - 调用完成后生成自然语言回复

    4. 测试 3 个场景:
       - "北京现在天气怎么样？"
       - "上海未来3天天气预报"
       - "你好"（不需要调用工具）

TODO:
    1. 定义 TOOLS 列表（两个工具的 Schema）
    2. 实现 get_current_weather(city, unit)
    3. 实现 get_forecast(city, days)
    4. 实现 run_conversation(user_query) —— 完整的执行循环
    5. 运行测试场景
"""

import os
import json
from datetime import datetime, timedelta
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ============================================================
# TODO 1: 定义两个工具的 Schema
# ============================================================
TOOLS = [
    # TODO: get_current_weather 的 Schema
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "get_current_weather",
    #         "description": "...",
    #         "parameters": { ... }
    #     }
    # },
    #
    # TODO: get_forecast 的 Schema
]

# ============================================================
# TODO 2: 实现 get_current_weather
# ============================================================
def get_current_weather(city: str, unit: str = "celsius") -> dict:
    """
    查询指定城市的当前天气。

    参数:
        city: 城市名称
        unit: 温度单位，celsius(摄氏度) 或 fahrenheit(华氏度)

    返回:
        {"city": "...", "temperature": ..., "unit": "...", "condition": "...", "humidity": ..., "wind": "..."}
    """
    # TODO: 为至少 5 个城市提供模拟数据
    # 提示: 用字典存储各城市的天气数据，找不到时返回 "暂无数据"
    pass


# ============================================================
# TODO 3: 实现 get_forecast
# ============================================================
def get_forecast(city: str, days: int = 3) -> dict:
    """
    查询指定城市未来几天的天气预报。

    参数:
        city: 城市名称
        days: 预报天数（1-7）

    返回:
        {"city": "...", "forecast": [{"date": "...", "condition": "...", "temp_high": ..., "temp_low": ...}, ...]}
    """
    # TODO: 生成 days 天的模拟天气预报数据
    # 提示: 用 datetime 生成未来日期，随机或预设天气状况
    pass


# ============================================================
# TODO 4: 实现完整的对话处理函数
# ============================================================
def run_conversation(user_query: str) -> str:
    """
    处理用户查询，自动调用工具并返回自然语言回复。

    实现步骤:
    1. 初始化 messages，包含 system prompt 和 user query
    2. 调用 API（带 tools 参数）
    3. 检查 response 中是否有 tool_calls:
       a. 有: 执行工具 → 将结果以 tool role 追加 → 再次调用 API → 返回回复
       b. 无: 直接返回回复
    4. 处理多个 tool_calls（并行调用情况）
    """
    # TODO: 实现完整的 Function Calling 循环
    # 提示: 参考教程中的 ToolExecutor 类，或简单的 if/else 判断
    pass


# ============================================================
# TODO 5: 测试
# ============================================================
if __name__ == "__main__":
    test_queries = [
        "北京现在天气怎么样？",
        "上海未来3天天气预报",
        "你好",
        "深圳今天天气如何？顺便看看未来5天预报",
        "东京现在天气如何？",  # 没有数据的城市
    ]

    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"用户: {q}")
        # reply = run_conversation(q)
        # print(f"助手: {reply}")

    print("\n请完成 TODO 后取消注释运行。")
