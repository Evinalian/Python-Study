"""
练习 1: 终端流式对话

场景:
    实现一个简单的终端聊天程序，支持流式显示和多轮对话。

要求:
    1. 实现 stream_chat(messages) 函数:
       - 调用 OpenAI API（stream=True）
       - 逐 token 打印（flush=True 确保即时显示）
       - 返回完整的回复文本

    2. 实现多轮对话循环:
       - 维护 messages 列表（包含 system + 历史对话）
       - 用户输入 → 流式生成回复 → 追加到 messages
       - 支持 /exit 退出, /clear 清空对话, /history 查看历史

    3. 流式打印的用户体验优化:
       - 打印 "助手: " 前缀
       - 每个 token 之间无额外空格
       - 完成后打印换行

TODO:
    1. 实现 stream_chat(messages) 函数
    2. 实现对话主循环（含命令处理: /exit, /clear, /history）
    3. 测试: 多轮对话 + 上下文记忆
"""

import os
import sys
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# TODO 1: 实现流式聊天函数
# ============================================================
def stream_chat(messages: list[dict]) -> str:
    """
    流式调用 OpenAI API，逐 token 打印并返回完整回复。

    参数:
        messages: 对话历史列表

    返回:
        完整的 assistant 回复文本

    实现提示:
    1. client.chat.completions.create(model="gpt-4o", messages=messages, stream=True)
    2. for chunk in response:
    3.     delta = chunk.choices[0].delta
    4.     if delta.content:
    5.         print(delta.content, end="", flush=True)
    6.         full_content += delta.content
    7. print()  # 最后的换行
    """
    # TODO: 实现流式调用
    pass


# ============================================================
# TODO 2: 实现对话主循环
# ============================================================
def main():
    """
    对话主循环。

    功能:
    - 维护 messages 列表
    - 逐轮对话（用户输入 → AI 流式回复 → 记录历史）
    - 命令:
      /exit   : 退出程序
      /clear  : 清空对话历史（保留 system prompt）
      /history: 显示对话历史摘要

    实现步骤:
    1. 初始化 messages = [system prompt]
    2. 循环:
       a. 获取用户输入 (input("你: "))
       b. 检查命令
       c. 将用户消息加入 messages
       d. 打印 "助手: "，调用 stream_chat(messages)
       e. 将 assistant 回复加入 messages
    """
    # TODO:
    # system_prompt = "你是一个有帮助的AI助手。回答简洁准确。"
    # messages = [{"role": "system", "content": system_prompt}]
    #
    # print("终端流式对话 (输入 /exit 退出, /clear 清空, /history 查看历史)\n")
    #
    # while True:
    #     user_input = input("你: ").strip()
    #     if not user_input:
    #         continue
    #     if user_input == "/exit":
    #         break
    #     elif user_input == "/clear":
    #         messages = [{"role": "system", "content": system_prompt}]
    #         print("对话已清空\n")
    #         continue
    #     elif user_input == "/history":
    #         # 打印历史摘要
    #         ...
    #         continue
    #
    #     messages.append({"role": "user", "content": user_input})
    #     print("助手: ", end="", flush=True)
    #     reply = stream_chat(messages)
    #     messages.append({"role": "assistant", "content": reply})
    #     print()
    pass


if __name__ == "__main__":
    # main()
    print("请完成 TODO 后取消注释运行。")
