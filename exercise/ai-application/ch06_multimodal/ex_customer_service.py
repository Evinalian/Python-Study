"""
进阶练习: 多模态客服原型
======================

需求:
  模拟一个简单的多模态客服系统，支持语音输入和图片输入。

  工作流程:
    1. 用户选择输入模式: 语音 或 图片 或 文字
    2. 语音模式: .mp3/.wav 文件 → Whisper 转文字 → 继续处理
    3. 图片模式: 图片文件 → GPT-4o 视觉理解 → 继续处理
    4. 文字模式: 直接输入文字 → 继续处理
    5. LLM 理解用户问题（带系统提示: 扮演客服角色）
    6. 生成文字回复
    7. TTS 将回复朗读出来（保存为 .mp3）
    8. 返回: 文字回复 + 语音回复文件路径

要求:
  1. 实现 CustomerServiceAgent 类:
     - __init__(self, system_prompt: str): 初始化，设置客服角色
     - process_text(self, text: str) -> str: 处理文字输入
     - process_voice(self, audio_path: str) -> str: 语音→文字→处理
     - process_image(self, image_path: str, question: str | None) -> str:
       图片→视觉理解→处理。如果 question 为 None，用默认问题"请描述这张图片中的问题"
     - reply(self, text: str) -> tuple[str, str]: 返回 (文字回复, 语音文件路径)

  2. 系统提示示例:
     "你是一位专业的电商客服，负责处理退换货、物流查询、产品咨询。
      你需要: (1) 礼貌友好 (2) 先确认用户问题再回答 (3) 给出具体的解决方案
      (4) 如果需要用户提供更多信息（如订单号/截图），请明确说明"

  3. 实现交互式命令行:
     - 显示菜单: [1] 文字输入 [2] 语音输入 [3] 图片输入 [4] 退出
     - 循环处理，直到用户选择退出
     - 每次回复后显示文字回复，并提示语音文件保存路径
     - 支持对话历史（保留最近 10 轮对话）

  4. 错误处理:
     - 文件不存在
     - 不支持的文件格式
     - API 调用失败（打印错误并让用户重试）
     - TTS 文本过长时自动分段

TODO:
  - [ ] 实现 CustomerServiceAgent 类
  - [ ] 实现 process_text / process_voice / process_image 方法
  - [ ] 实现 reply 方法（含 TTS）
  - [ ] 实现对话历史管理（最近 10 轮）
  - [ ] 实现交互式命令行循环

提示:
  - 使用 messages 列表维护对话历史，role 分别为 user 和 assistant
  - TTS 用 tts-1 模型，音色用 "nova" 或 "alloy"
  - 语音识别用 whisper-1 的 verbose_json 格式，取 text 字段
  - 图片处理时先用 PIL 做预处理（缩放到短边 768 内）
  - 对话历史长度超过 20 条（10 轮）时，保留最近 20 条
  - 可以使用 threading 在后台做 TTS（让用户不用等语音生成）
"""
import os
import base64
import io
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class CustomerServiceAgent:
    """多模态客服代理"""

    def __init__(self, system_prompt: str):
        # TODO: 初始化对话历史，第一条为 system 角色
        pass

    def process_text(self, text: str) -> str:
        # TODO: 将用户文字加入历史，调用 LLM 获取回复，将回复加入历史，返回回复
        pass

    def process_voice(self, audio_path: str) -> str:
        # TODO: 验证文件存在 → Whisper 转文字 → 调用 process_text
        pass

    def process_image(self, image_path: str, question: str | None = None) -> str:
        # TODO: 加载图片 → 预处理 → 调用 GPT-4o vision → 调用 process_text
        pass

    def reply(self, text: str) -> tuple[str, str]:
        # TODO: 生成 TTS 语音并保存 → 返回 (文字回复, 语音文件路径)
        pass


def interactive_loop():
    # TODO: 创建 agent → 显示菜单 → 循环处理用户选择
    pass


def main():
    # TODO: 初始化 agent 并启动交互循环
    pass


if __name__ == "__main__":
    main()
