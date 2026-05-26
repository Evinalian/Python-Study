"""
练习 5: 终端 UI 流式对话客户端

场景:
    使用 Python 的 rich 库构建一个更漂亮的终端对话客户端。

要求:
    1. 使用 rich 库实现:
       - 彩色输出（用户消息蓝色，AI 回复绿色）
       - Markdown 渲染（代码块、加粗、列表）
       - 对话历史可滚动
       - 状态提示（"正在生成..." / "已完成"）

    2. 流式输出:
       - AI 回复以流式方式逐 token 显示
       - 使用 rich.live.Live 实现实时更新
       - 完成后渲染为完整的 Markdown

    3. 交互功能:
       - 支持多轮对话
       - Ctrl+C 中断生成
       - /help 查看命令列表

安装依赖:
    pip install rich

TODO:
    1. 实现 ChatUI 类（封装 rich 渲染逻辑）
    2. 实现流式生成 + Markdown 渲染
    3. 实现交互循环（含命令处理）
    4. 实现 Ctrl+C 中断
"""

import os
import sys
import signal
from openai import OpenAI

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("请先安装 rich: pip install rich")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

console = Console() if RICH_AVAILABLE else None

# ============================================================
# TODO 1: ChatUI 类
# ============================================================
class ChatUI:
    """
    基于 rich 的聊天界面管理器。

    功能:
    - 彩色输出用户/AI消息
    - 流式渲染AI回复
    - Markdown 渲染
    """

    def __init__(self, system_prompt: str = "你是一个有帮助的AI助手。"):
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]
        self.interrupted = False

    def print_user(self, text: str):
        """打印用户消息（蓝色面板）"""
        # TODO: 使用 rich Panel, style="blue"
        pass

    def print_assistant_header(self):
        """打印助手回复的起始标签"""
        # TODO: 打印 "助手:" 标签
        pass

    def stream_reply(self) -> str:
        """
        流式获取并实时渲染 AI 回复。

        返回:
            完整的回复文本

        提示:
        1. 调用 API (stream=True)
        2. 使用 rich.live.Live 实时更新
        3. 收集完整文本
        4. 完成后渲染为 Markdown
        """
        # TODO: 实现流式渲染
        pass

    def print_error(self, msg: str):
        """打印错误消息（红色）"""
        # TODO: console.print(f"[red]{msg}[/red]")
        pass

    def print_info(self, msg: str):
        """打印信息消息（黄色）"""
        # TODO: console.print(f"[yellow]{msg}[/yellow]")
        pass

    def add_to_history(self, role: str, content: str):
        """添加到对话历史"""
        # TODO: messages.append({"role": role, "content": content})
        pass

    def run(self):
        """
        运行对话循环。

        命令:
        - /exit: 退出
        - /clear: 清空历史
        - /help: 帮助
        - Ctrl+C: 中断当前生成
        """
        # TODO: 实现主循环
        pass


# ============================================================
# TODO: 信号处理（Ctrl+C 中断生成）
# ============================================================
# 提示: 使用 signal.signal(signal.SIGINT, handler) 捕获 Ctrl+C


if __name__ == "__main__":
    if not RICH_AVAILABLE:
        print("请先安装 rich: pip install rich")
        sys.exit(1)

    # chat = ChatUI()
    # chat.run()

    print("请完成 TODO 后取消注释运行。")
