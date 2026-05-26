"""
练习 4: 通用 ToolExecutor 框架

场景:
    实现一个通用的 ToolExecutor 类，支持任意工具集的注册和执行。
    这是生产级 Function Calling 的核心组件。

要求:
    1. 实现 ToolExecutor 类:
       - __init__: 注册 tools schema 和 function_map
       - execute(messages): 执行对话循环直到模型不再调用工具
       - run(system_prompt, user_query): 便捷方法

    2. 支持功能:
       - 并行工具调用处理
       - 最大轮次限制（默认 10，防止无限循环）
       - 单次工具调用超时（默认 30 秒）
       - 错误记录到日志
       - 返回完整的 messages 历史（方便调试）

    3. 实现工具注册:
       - register(name, schema, func): 动态注册新工具
       - 运行时添加/移除工具

    4. 测试: 用 ToolExecutor 运行三个不同的工具集组合
"""

import os
import json
import time
import threading
from typing import Callable
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# TODO 1: 实现 ToolExecutor 类
# ============================================================
class ToolExecutor:
    """
    通用的 Function Calling 执行器。

    职责:
    - 管理工具注册表
    - 执行对话循环
    - 处理并行调用、超时、错误
    """

    def __init__(self, max_rounds: int = 10, tool_timeout: float = 30.0):
        """
        初始化执行器。

        参数:
            max_rounds: 最大工具调用轮次
            tool_timeout: 单个工具调用的超时时间（秒）
        """
        # TODO: 初始化
        # self.tool_schemas = []   # 所有工具的 Schema 列表
        # self.function_map = {}   # {"func_name": callable}
        # self.max_rounds = max_rounds
        # self.tool_timeout = tool_timeout
        # self.logs = []           # 执行日志
        pass

    def register(self, schema: dict, func: Callable) -> None:
        """
        注册一个工具。

        参数:
            schema: 工具的 JSON Schema（包含 type: "function" 的完整定义）
            func: 工具对应的可调用函数
        """
        # TODO:
        # 1. 从 schema 中提取 function name
        # 2. 将 schema 追加到 self.tool_schemas
        # 3. 将 func 注册到 self.function_map[name]
        pass

    def unregister(self, name: str) -> None:
        """移除一个工具"""
        # TODO:
        # 1. 从 self.tool_schemas 中移除
        # 2. 从 self.function_map 中移除
        pass

    def execute(
        self,
        messages: list[dict],
        model: str = "gpt-4o",
        temperature: float = 0.0,
    ) -> list[dict]:
        """
        执行对话循环。

        参数:
            messages: 初始 messages 列表（至少包含 system + user）
            model: 模型名称
            temperature: 生成参数

        返回:
            完整的 messages 列表（包含所有工具调用和结果）
        """
        # TODO: 实现核心循环
        # for round_num in range(self.max_rounds):
        #     1. 调用 API
        #     2. 检查 tool_calls
        #        - 无: 追加 assistant 消息，break
        #        - 有:
        #          a. 追加 assistant 消息（含 tool_calls）
        #          b. 遍历执行每个 tool_call
        #             - 使用 execute_with_timeout 防止超时
        #             - try/except 捕获异常
        #             - 追加 tool 消息（成功或错误）
        #          c. 记录日志
        # else: 达到最大轮次，记录警告
        pass

    def run(
        self,
        system_prompt: str,
        user_query: str,
        model: str = "gpt-4o",
    ) -> str:
        """
        便捷方法: 运行一次对话并返回最终回复。

        参数:
            system_prompt: System 消息内容
            user_query: 用户消息内容
            model: 模型名称

        返回:
            最终回复文本
        """
        # TODO: 构建初始 messages，调用 self.execute()，提取最终回复
        pass

    def get_logs(self) -> list[dict]:
        """获取执行日志"""
        # TODO: 返回日志列表
        pass


# ============================================================
# TODO 2: 工具函数（供测试用）
# ============================================================
def calculator(expression: str) -> dict:
    """安全的数学计算器"""
    # TODO: 实现基本计算（只允许数字和 +-*/() ）
    pass


def get_time() -> dict:
    """获取当前时间"""
    # TODO: 用 datetime.now() 返回当前时间
    pass


def text_tools(text: str, operation: str) -> dict:
    """
    文本工具: 支持 uppercase, lowercase, reverse, count_words
    """
    # TODO: 根据 operation 执行对应操作
    pass


# ============================================================
# TODO 3: 定义工具 Schema
# ============================================================
CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "TODO: 写 description（含触发条件）",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "TODO"}
            },
            "required": ["expression"],
        },
    },
}

GET_TIME_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "TODO",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

TEXT_TOOLS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "text_tools",
        "description": "TODO",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "TODO"},
                "operation": {
                    "type": "string",
                    "description": "TODO",
                    "enum": ["uppercase", "lowercase", "reverse", "count_words"],
                },
            },
            "required": ["text", "operation"],
        },
    },
}


# ============================================================
# TODO 4: 测试
# ============================================================
if __name__ == "__main__":
    # 测试1: 基础使用
    print("=== 测试1: 基础计算器 ===\n")
    # executor = ToolExecutor(max_rounds=3)
    # executor.register(CALCULATOR_SCHEMA, calculator)
    # executor.register(GET_TIME_SCHEMA, get_time)
    #
    # reply = executor.run(
    #     system_prompt="你是助手，可以计算和查时间。",
    #     user_query="计算 (15 + 23) * 7 的结果，并告诉我现在几点",
    # )
    # print(f"回复: {reply}")

    # 测试2: 运行时动态切换工具
    print("\n=== 测试2: 动态注册/移除工具 ===\n")
    # executor.unregister("get_time")
    # reply = executor.run(
    #     system_prompt="你是助手，只能计算不能查时间。",
    #     user_query="现在几点？",
    # )
    # print(f"回复: {reply}")

    # 测试3: 文本处理工具
    print("\n=== 测试3: 文本工具 ===\n")
    # executor.register(TEXT_TOOLS_SCHEMA, text_tools)
    # reply = executor.run(
    #     system_prompt="你是文本处理助手。",
    #     user_query="把 'Hello World' 反转并转为大写",
    # )
    # print(f"回复: {reply}")

    print("\n请完成 TODO 后取消注释运行。")
