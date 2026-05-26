"""
练习 2: 流式 + Function Calling

场景:
    实现流式天气查询助手，在流式模式下正确处理 tool_calls。

核心挑战:
    在流式模式下，tool_calls 是分多个 chunk 传输的:
    - Chunk A: delta.tool_calls[0].id = "call_xxx"
    - Chunk B: delta.tool_calls[0].function.name = "get"
    - Chunk C: delta.tool_calls[0].function.name = "_weather"  (需要累积)
    - Chunk D-K: delta.tool_calls[0].function.arguments = 分段传输的 JSON
    - 最后一个 chunk: finish_reason = "tool_calls"

    你需要把这些片段累积起来，拼成完整的 tool_call。

要求:
    1. 实现 accumulate_tool_calls(stream) 函数:
       - 接收流式 response
       - 累积所有 tool_calls（支持多个并行调用）
       - 返回完整的 tool_calls 列表

    2. 实现 stream_with_tools(user_query) 函数:
       - 流式调用 API（可能产生 tool_calls）
       - 如果产生 tool_calls: 累积 → 执行 → 再次流式调用获取最终回复
       - 如果直接回复文本: 流式打印

    3. 工具: get_weather(city), get_time()

TODO:
    1. 定义 TOOLS 列表和函数
    2. 实现 accumulate_tool_calls(stream)
    3. 实现 stream_with_tools(user_query)
    4. 测试: 天气查询、时间查询
"""

import os
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# TODO 1: 定义工具
# ============================================================
def get_weather(city: str) -> dict:
    """模拟天气查询"""
    # TODO: 为至少 3 个城市提供模拟数据
    pass


def get_time() -> dict:
    """获取当前时间"""
    # TODO: 用 datetime.now() 返回当前时间字符串
    pass


TOOLS = [
    # TODO: get_weather Schema
    # TODO: get_time Schema
]

FUNCTION_MAP = {"get_weather": get_weather, "get_time": get_time}


# ============================================================
# TODO 2: 累积 tool_calls
# ============================================================
def accumulate_tool_calls(stream):
    """
    从流式 response 中累积 tool_calls。

    流式模式下 tool_calls 的传输特点:
    - 每个 tool_call 有一个固定的 index
    - id 可能在第一个 chunk 给出
    - function.name 可能分多个 chunk 到达
    - function.arguments 是 JSON 字符串，分多个 chunk 到达

    返回:
        accumulated: dict[int, dict]
        {
            0: {"id": "call_xxx", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"北京"}'}},
        }

    提示:
    accumulated = {}
    for chunk in stream:
        if chunk.choices[0].delta.tool_calls:
            for tc_delta in chunk.choices[0].delta.tool_calls:
                idx = tc_delta.index
                if idx not in accumulated:
                    accumulated[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                # 累积 id, name, arguments
    """
    # TODO: 实现累积逻辑
    pass


# ============================================================
# TODO 3: 流式 + 工具调用
# ============================================================
def stream_with_tools(user_query: str):
    """
    流式调用，正确处理 tool_calls。

    流程:
    1. 流式调用 API (tools=TOOLS
    2. 同时做两件事:
       a. 如果有 content: 打印出来
       b. 如果有 tool_calls: 累积 (但不能同时发生)
    3. 如果产生了 tool_calls:
       a. 执行工具
       b. 将结果追加到 messages
       c. 再次流式调用获取最终回复
    4. 打印最终回复
    """
    # TODO: 实现完整流程
    pass


if __name__ == "__main__":
    print("=== 流式 Function Calling ===\n")

    # 测试1: 工具调用
    # stream_with_tools("北京现在天气怎么样？")

    # 测试2: 直接对话
    # stream_with_tools("你好，你叫什么名字？")

    # 测试3: 并行工具调用
    # stream_with_tools("北京和上海的天气分别怎么样？")

    print("请完成 TODO 后取消注释运行。")
