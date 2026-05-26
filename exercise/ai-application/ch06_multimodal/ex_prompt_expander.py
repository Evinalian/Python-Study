"""
练习: DALL-E Prompt 扩展生成器
=============================

需求:
  用户输入一段简短的中文描述（可能不完整、口语化），
  系统先用 GPT-4o 将其扩展为结构化的英文 DALL-E Prompt，
  再调用 DALL-E 3 生成图片并保存。

要求:
  1. 实现 expand_prompt(user_input: str) -> dict:
     - 调用 GPT-4o，将中文简短描述扩展为包含以下字段的 JSON:
       { "subject": "主体", "environment": "环境", "style": "风格",
         "lighting": "光照", "composition": "构图", "colors": "色调",
         "full_prompt": "完整英文Prompt" }
     - 使用 response_format={"type": "json_object"} 确保 JSON 输出

  2. 实现 generate_image(full_prompt: str, style: str, output_path: str) -> str:
     - 根据 style 字段追加风格修饰词到 prompt
     - 调用 DALL-E 3 生成图片
     - 支持 --size 参数（1024x1024 / 1792x1024 / 1024x1792）
     - 支持 --quality 参数（standard / hd）

  3. 实现交互式命令行界面:
     - 程序启动后提示用户输入描述
     - 显示扩展后的 Prompt 结构
     - 询问用户是否满意（y/n），不满意则让用户说"要改什么"，重新扩展
     - 满意后生成图片
     - 显示生成的图片路径和 API 返回的 revised_prompt

  4. 错误处理:
     - 处理内容审核不通过的情况（content_filter 错误）
     - 处理网络超时
     - 在扩展 prompt 失败时给出友好的错误提示

TODO:
  - [ ] 实现 expand_prompt(user_input) 调用 GPT-4o 扩展 prompt
  - [ ] 实现 generate_image(full_prompt, style, size, quality, output_path)
  - [ ] 实现交互循环：输入 → 扩展 → 确认 → 生成
  - [ ] 实现 main() 串联全部流程

提示:
  - GPT-4o 的系统提示中给出 JSON schema 可以提高输出质量
  - DALL-E 3 的 revised_prompt 字段显示了模型自己改写的 prompt，可以打印出来供用户学习
  - style 字段额外追加到 prompt 中可以补偿 DALL-E 3 改写造成的风格丢失
  - 使用 requests 库下载生成的图片 URL
"""
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def expand_prompt(user_input: str) -> dict:
    # TODO: 调用 GPT-4o，用 response_format json_object 扩展 prompt
    pass


def generate_image(full_prompt: str, style: str, size: str, quality: str, output_path: str) -> str:
    # TODO: 调用 DALL-E 3 生成图片并下载到 output_path
    pass


def interactive_loop():
    # TODO: 实现交互循环
    pass


def main():
    # TODO: 解析命令行参数（--size, --quality, --output），启动交互循环
    pass


if __name__ == "__main__":
    main()
