"""
练习: Gradio 多模型对比工具
===========================

需求:
  构建一个 Gradio 应用，允许用户:
    - 选择两个不同的模型
    - 输入同一个问题
    - 同时查看两个模型的回答（并排展示）
    - 保存对比结果到本地文件

要求:
  1. 界面布局:
     - 顶部: 标题和说明
     - 左侧面板: 模型选择、参数设置（温度、max_tokens）、输入框
     - 右侧面板: 模型A回答（上）、模型B回答（下）
     - 底部: 对比分析按钮 + 保存按钮

  2. 功能:
     - 模型选择: 下拉框，选项至少包含 gpt-4o, gpt-4o-mini, gpt-4.1-mini
     - 支持流式输出（两个模型同时流式显示）
     - "开始对比" 按钮触发两个模型并行调用
     - "AI 分析差异" 按钮: 调用第三个模型来分析两个回答的差异
     - "保存结果" 按钮: 将问题和两个回答保存为 TXT 文件

  3. 技术点:
     - 使用 gr.Blocks 构建自定义布局
     - 使用 threading.Thread 或 asyncio 并行调用两个模型
     - 使用 gr.State 存储对话历史
     - 使用 yield 实现流式更新

  4. 可选增强:
     - 添加主题切换
     - 支持多轮对比（历史记录列表）
     - 模型响应时间对比（显示每个模型的耗时）

TODO:
  - [ ] 设计界面布局（gr.Blocks + gr.Row + gr.Column）
  - [ ] 实现 call_model(model, messages, temperature, max_tokens) 调用单个模型
  - [ ] 实现 compare_models() 并行调用两个模型
  - [ ] 实现 analyze_differences() 让 AI 分析两个回答的差异
  - [ ] 实现 save_result() 保存对比结果
  - [ ] 实现流式输出（用 yield 分段更新）

提示:
  - 并行调用: 使用 concurrent.futures.ThreadPoolExecutor 或 asyncio.gather
  - 流式更新: 两个模型用两个单独的生成器，交替 yield 更新
  - 响应时间: 用 time.time() 记录每个模型的开始和结束时间
  - 分析差异: prompt 中包含"模型A的回答"和"模型B的回答"，让 GPT 做对比
"""
import os
import time
import json
from datetime import datetime
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_model_stream(model: str, messages: list, temperature: float, max_tokens: int):
    """流式调用单个模型，yield 每个文本片段"""
    # TODO: 实现流式调用
    pass


def compare_models(prompt, model_a, model_b, temperature, max_tokens):
    """并行调用两个模型并流式返回结果"""
    # TODO: 实现并行比较
    pass


def analyze_differences(prompt, response_a, response_b):
    """让 AI 分析两个回答的差异"""
    # TODO: 实现差异分析
    pass


def save_result(prompt, model_a, response_a, model_b, response_b):
    """保存对比结果到文件"""
    # TODO: 实现保存
    pass


def build_interface():
    """构建 Gradio 界面"""
    # TODO: 用 gr.Blocks 构建完整界面
    pass


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860)
