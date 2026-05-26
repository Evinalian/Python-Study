"""
练习 4: 并行翻译对比

场景:
    同时将一段文本翻译成英文、日文、韩文，三个流并行运行。

要求:
    1. 实现 async_translate(text, target_lang, label) 函数:
       - 使用 AsyncOpenAI
       - 流式调用，每个 token 标注语言标签
       - 返回完整翻译

    2. 实现 parallel_translate(text, langs) 函数:
       - 使用 asyncio.gather 并行运行多个翻译
       - 所有翻译同时流式输出（各行交错显示）
       - 打印总结: 每个语言的翻译长度和耗时

    3. 打印格式:
       [英文] Artificial intelligence is...
       [日文] 人工知能は...
       [韩文] 인공지능은...
       三个流同时打印，行与行之间可能交错

TODO:
    1. 实现 async_translate(text, target_lang, label)
    2. 实现 parallel_translate(text, langs)
    3. 测试: 翻译一段中文到三种语言
"""

import os
import asyncio
import time
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# TODO 1: 异步流式翻译
# ============================================================
async def async_translate(text: str, target_lang: str, label: str) -> dict:
    """
    流式翻译到指定语言。

    参数:
        text: 原文
        target_lang: 目标语言（如 "英文", "日文", "韩文"）
        label: 显示标签（如 "[英文]", "[日文]"）

    返回:
        {"label": label, "content": "翻译内容", "elapsed": 耗时秒数}

    提示:
    1. system_prompt = f"你是专业翻译。将以下内容翻译为{target_lang}。只输出译文不要解释。"
    2. stream=True 流式调用
    3. 每个 token 打印为 f"{label} {token}"
    4. 记录开始和结束时间
    """
    # TODO: 实现
    pass


# ============================================================
# TODO 2: 并行流式翻译
# ============================================================
async def parallel_translate(text: str, langs: list[tuple[str, str]]):
    """
    并行翻译到多种语言。

    参数:
        text: 待翻译文本
        langs: [(target_lang, label), ...]
               例如 [("英文", "[EN]"), ("日文", "[JP]"), ("韩文", "[KR]")]

    流程:
    1. 创建多个 asyncio Task
    2. 用 asyncio.gather 并发执行
    3. 收集所有结果
    4. 打印总结报告

    提示:
    tasks = [async_translate(text, lang, label) for lang, label in langs]
    results = await asyncio.gather(*tasks)
    """
    # TODO: 实现
    pass


if __name__ == "__main__":
    text = "人工智能正在深刻改变我们的生活方式，从智能手机到自动驾驶，AI技术无处不在。"
    langs = [("英文", "[EN]"), ("日文", "[JP]"), ("韩文", "[KR]")]

    print(f"原文: {text}\n")
    print("并行翻译中...\n")

    # TODO: 取消注释
    # asyncio.run(parallel_translate(text, langs))

    print("\n请完成 TODO 后运行。")
