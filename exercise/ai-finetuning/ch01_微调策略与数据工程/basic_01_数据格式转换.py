"""
练习 1: 数据格式转换 —— 将多源数据统一为 Chat Template

场景:
    你拿到了三份不同来源的微调数据：
    - 来源 A（CSV）: 列名为 question/answer/category
    - 来源 B（JSON）: 每个对象有 input/output 字段
    - 来源 C（JSONL）: 每行有 prompt/completion 字段

    你需要将它们统一转换为标准的 Chat Template 格式:
    {
        "messages": [
            {"role": "system", "content": "系统指令"},
            {"role": "user", "content": "用户输入"},
            {"role": "assistant", "content": "期望回复"}
        ]
    }

要求:
    1. 为三个来源分别编写转换函数
    2. 统一注入 system_prompt（"你是一个有帮助的中文AI助手。"）
    3. 过滤掉空的或太短的问答对
    4. 检测并报告转换过程中的异常数据
    5. 最终输出为统一的 JSONL 文件

TODO:
    1. 实现 convert_csv(source_path, system_prompt) 函数:
       - 读取 CSV，列名为 question/answer/category
       - 保留 category 字段到输出中（作为 metadata）
       - 过滤 answer 长度 < 20 字符的数据

    2. 实现 convert_json(source_path, system_prompt) 函数:
       - 读取 JSON 数组，字段为 input/output
       - 处理可能的嵌套 JSON 结构
       - 检测 input/output 字段缺失的情况

    3. 实现 convert_jsonl(source_path, system_prompt) 函数:
       - 逐行读取 JSONL，字段为 prompt/completion
       - 处理可能的格式错误（某行不是合法 JSON）
       - 报告解析失败的行数

    4. 实现 merge_and_save(sources_config, output_path) 函数:
       - sources_config 是一个列表，每项指定 {"path": ..., "type": "csv"/"json"/"jsonl"}
       - 依次调用对应的转换函数
       - 合并所有转换结果
       - 保存为新 JSONL 文件

    5. 实现 generate_stats(dataset) 函数:
       - 统计总数据量
       - 统计 instruction 和 response 的长度分布（平均、中位数、最小、最大）
       - 统计各类别的数量（如果有 category 字段）
       - 打印统计报告

    6. 思考题（注释回答）:
       - 为什么统一 Chat Template 格式很重要？
       - 如果 training 和 inference 时使用的 Chat Template 不一致会怎样？
"""
import os
import json
import csv
from typing import Optional

# ============================================================
# TODO 1: CSV 转换函数
# ============================================================
def convert_csv(source_path: str, system_prompt: str = "") -> list[dict]:
    """
    将 CSV 文件（question/answer/category）转换为标准 messages 格式。

    参数:
        source_path: CSV 文件路径
        system_prompt: 全局 system prompt

    返回:
        [{"messages": [...], "metadata": {"category": "..."}}, ...]
    """
    # TODO: 读取 CSV 文件
    # TODO: 遍历每一行
    # TODO: 跳过 answer 为空的或长度 < 20 的行
    # TODO: 构建 messages 格式
    # TODO: 保留 category 到 metadata
    pass


# ============================================================
# TODO 2: JSON 转换函数
# ============================================================
def convert_json(source_path: str, system_prompt: str = "") -> list[dict]:
    """
    将 JSON 数组文件（input/output）转换为标准 messages 格式。

    参数:
        source_path: JSON 文件路径
        system_prompt: 全局 system prompt

    返回:
        [{"messages": [...]}, ...]
    """
    # TODO: 读取 JSON 文件（假设是 JSON 数组）
    # TODO: 遍历数组中的每个对象
    # TODO: 检查 input/output 字段是否存在
    # TODO: 记录缺失字段的数据条数
    # TODO: 构建 messages 格式
    pass


# ============================================================
# TODO 3: JSONL 转换函数
# ============================================================
def convert_jsonl(source_path: str, system_prompt: str = "") -> list[dict]:
    """
    将 JSONL 文件（prompt/completion）转换为标准 messages 格式。

    参数:
        source_path: JSONL 文件路径
        system_prompt: 全局 system prompt

    返回:
        [{"messages": [...]}, ...]
    """
    # TODO: 逐行读取 JSONL
    # TODO: try-except 捕获 JSON 解析错误
    # TODO: 记录解析失败的行号和内容
    # TODO: 检查 prompt/completion 字段
    # TODO: 构建 messages 格式
    pass


# ============================================================
# TODO 4: 合并与保存函数
# ============================================================
def merge_and_save(sources_config: list[dict], output_path: str) -> list[dict]:
    """
    合并多个来源的数据并保存为 JSONL。

    参数:
        sources_config: 数据源配置列表
            [{"path": "data/a.csv", "type": "csv"}, ...]
        output_path: 输出 JSONL 文件路径

    返回:
        合并后的数据列表
    """
    # TODO: 定义 type -> 转换函数的映射
    # TODO: 遍历 sources_config，调用对应转换函数
    # TODO: 合并所有结果
    # TODO: 保存为 JSONL（每行一个 JSON 对象，ensure_ascii=False）
    pass


# ============================================================
# TODO 5: 统计报告函数
# ============================================================
def generate_stats(dataset: list[dict]) -> dict:
    """
    生成数据集的统计报告。

    返回:
        {
            "total": 总数据量,
            "instruction_length": {"mean": ..., "median": ..., "min": ..., "max": ...},
            "response_length": {"mean": ..., "median": ..., "min": ..., "max": ...},
            "categories": {"category1": count1, ...} 或 None
        }
    """
    # TODO: 统计总数据量
    # TODO: 计算 instruction 长度（每个样本的最后一个 user message）
    # TODO: 计算 response 长度（每个样本的最后一个 assistant message）
    # TODO: 统计 category 分布（如果 metadata 中有 category）
    pass


# ============================================================
# TODO 6: 思考题
# ============================================================
"""
Q1: 为什么统一 Chat Template 格式很重要？
A1: TODO

Q2: 如果 training 和 inference 时使用的 Chat Template 不一致会怎样？
A2: TODO

Q3: 为什么需要过滤 answer 太短的数据？
A3: TODO
"""


# ============================================================
# 测试代码（完成 TODO 后取消注释运行）
# ============================================================
if __name__ == "__main__":
    SYSTEM_PROMPT = "你是一个有帮助的中文AI助手。"

    # 创建示例数据用于测试
    # (在实际运行前，先手写几个示例 CSV/JSON/JSONL 文件)
    print("请先完成所有 TODO，然后准备三个测试数据文件并运行。")
    print("示例配置:")
    print("""
    sources = [
        {"path": "test_data/source_a.csv", "type": "csv"},
        {"path": "test_data/source_b.json", "type": "json"},
        {"path": "test_data/source_c.jsonl", "type": "jsonl"},
    ]
    all_data = merge_and_save(sources, "output/unified_data.jsonl")
    stats = generate_stats(all_data)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    """)
