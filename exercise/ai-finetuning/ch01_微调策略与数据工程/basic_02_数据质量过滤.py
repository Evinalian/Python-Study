"""
练习 2: 数据质量过滤 —— 规则过滤 + 评分筛选

场景:
    你从开源社区下载了一个包含 20000 条中文指令数据的数据集。
    但仔细检查后发现，数据质量参差不齐：
    - 有些指令太短（如"你好"），几乎没有学习价值
    - 有些回复太长且包含重复内容（水文）
    - 有些数据包含乱码或非中文内容
    - 有些回复明显是敷衍的（如"我不知道"、"这个问题很难回答"）

    你需要编写一套过滤系统，用规则筛掉低质数据。

要求:
    1. 实现可配置的多层过滤规则
    2. 每层过滤都打印输入的条数和过滤掉的条数
    3. 最后用基于规则的质量评分系统给每条数据打分
    4. 只保留评分 >= 阈值的优质数据
    5. 生成过滤统计报告

TODO:
    1. 实现 FilterStats 数据类：
       - 记录每个过滤步骤的名称、输入数、输出数
       - 实现 total_removed() 方法
       - 实现 summary() 方法打印完整统计

    2. 实现长度过滤器 length_filter(data, min_inst, max_inst, min_resp, max_resp):
       - 指令长度在 [min_inst, max_inst] 之间
       - 回复长度在 [min_resp, max_resp] 之间
       - 建议默认值: min_inst=10, max_inst=2000, min_resp=20, max_resp=8000

    3. 实现语言过滤器 language_filter(data, target_lang="zh"):
       - 使用简单的启发式规则判断是否为中文（中文字符占比 > 50%）
       - 过滤掉非中文指令
       - (进阶: 使用 fasttext 或 lingua 库)

    4. 实现敷衍回复过滤器 low_quality_filter(data):
       - 过滤包含敷衍模式的回复（如"我不知道"、"很难说"、"这个我不太清楚"）
       - 过滤回复中独特字符占比 < 30% 的数据（可能是重复内容）
       - 过滤回复中感叹号/questions 符号占比过高（> 15%）的数据

    5. 实现质量评分函数 score_quality(instruction, response) -> float (0-100):
       - 指令信息量（长度、是否包含问号/疑问词）（25分）
       - 回复信息量（长度、段落数）（30分）
       - 回复多样性（独特字符占比）（20分）
       - 结构规范性（是否有标点、是否有分段）（25分）

    6. 组装完整 pipeline: load -> length_filter -> language_filter -> low_quality_filter -> score_quality -> filter_by_score -> save

    7. 思考题（注释回答）:
       - 过滤太激进会有什么问题？
       - 如何为不同的微调场景（代码生成 vs 闲聊）调整过滤参数？
"""
import re
import json
from typing import Optional
from dataclasses import dataclass


# ============================================================
# TODO 1: FilterStats 数据类
# ============================================================
@dataclass
class FilterStats:
    """记录每个过滤步骤的统计信息。"""
    # TODO: 定义 steps 列表，每项为 {"name": str, "input": int, "output": int}
    # TODO: 实现 add_step(name, input_count, output_count) 方法
    # TODO: 实现 total_removed() -> int 方法
    # TODO: 实现 summary() 方法，打印类似:
    #   ===== 过滤统计 =====
    #   初始数据量:        20000
    #   长度过滤后:        18000  (-2000)
    #   语言过滤后:        16500  (-1500)
    #   ...
    #   最终保留率:        70.0%
    pass


# ============================================================
# TODO 2: 长度过滤器
# ============================================================
def length_filter(
    data: list[dict],
    min_inst: int = 10,
    max_inst: int = 2000,
    min_resp: int = 20,
    max_resp: int = 8000,
) -> list[dict]:
    """按指令和回复的长度范围过滤数据。"""
    # TODO: 遍历 data
    # TODO: 提取 instruction 和 response（兼容多种字段名）
    # TODO: 检查长度是否在范围内
    # TODO: 返回通过过滤的数据
    pass


# ============================================================
# TODO 3: 语言过滤器
# ============================================================
def is_mostly_chinese(text: str, threshold: float = 0.5) -> bool:
    """判断文本是否主要为中文（中文字符占比 > threshold）。"""
    # TODO: 统计文本中 Unicode 中文字符（一-鿿）的数量
    # TODO: 计算中文字符占比
    # TODO: 如果文本太短（<10 字符），直接返回 True 不过滤
    pass


def language_filter(data: list[dict]) -> list[dict]:
    """过滤掉指令非中文的数据。"""
    # TODO: 遍历 data
    # TODO: 提取 instruction
    # TODO: 用 is_mostly_chinese 判断
    # TODO: 返回通过过滤的数据
    pass


# ============================================================
# TODO 4: 低质量回复过滤器
# ============================================================
# 预定义的敷衍模式列表
DISMISSIVE_PATTERNS = [
    r"我不知道",
    r"我不清楚",
    r"这个我不太?了解",
    r"很难说",
    r"无法确定",
    r"我没有?办法回答",
    r"这个问题.*?复杂",
    r"作为AI.*?无法",
]

def low_quality_filter(data: list[dict]) -> list[dict]:
    """过滤低质量的回复。"""
    # TODO: 遍历 data
    # TODO: 检查回复是否匹配任何 DISMISSIVE_PATTERNS
    # TODO: 计算回复的独特字符占比（set(text) / len(text)）
    # TODO: 如果独特字符占比 < 0.3，视为低质量
    # TODO: 计算感叹号占比（感叹号数 / 总字符数）
    # TODO: 如果感叹号占比 > 0.15，视为低质量
    # TODO: 返回通过过滤的数据
    pass


# ============================================================
# TODO 5: 质量评分
# ============================================================
def score_quality(instruction: str, response: str) -> float:
    """
    对单条数据计算质量评分（0-100）。

    评分维度:
    - 指令信息量（25分）：长度、是否包含疑问词
    - 回复信息量（30分）：长度、段落数
    - 回复多样性（20分）：独特字符占比
    - 结构规范性（25分）：标点使用、分段
    """
    score = 0.0
    # TODO: 实现四个维度的评分逻辑
    # TODO: 返回 0-100 之间的分数
    pass


def filter_by_quality(data: list[dict], min_score: float = 40.0) -> list[dict]:
    """过滤掉质量分数低于阈值的样本。"""
    # TODO: 为每条数据计算质量分数
    # TODO: 打印评分分布统计（0-20, 20-40, 40-60, 60-80, 80-100 各有多少条）
    # TODO: 只保留分数 >= min_score 的数据
    pass


# ============================================================
# TODO 6: 组装完整 Pipeline
# ============================================================
def run_filter_pipeline(
    input_path: str,
    output_path: str,
    min_score: float = 40.0,
) -> list[dict]:
    """
    运行完整的过滤 pipeline。

    流程: 加载 -> 长度过滤 -> 语言过滤 -> 低质量过滤 -> 评分 -> 质量过滤 -> 保存
    """
    # TODO: 从 JSONL 加载数据
    # TODO: 创建 FilterStats 对象
    # TODO: 依次执行每个过滤步骤，记录统计数据
    # TODO: 打印 FilterStats.summary()
    # TODO: 保存过滤后的数据
    # TODO: 返回过滤后的数据
    pass


# ============================================================
# TODO 7: 思考题
# ============================================================
"""
Q1: 过滤太激进会有什么问题？
A1: TODO

Q2: 如何为不同的微调场景（代码生成 vs 闲聊）调整过滤参数？
A2: TODO

Q3: 为什么不能只依赖规则过滤（还需要后续的质量评分）？
A3: TODO
"""


if __name__ == "__main__":
    print("请完成所有 TODO，准备一个 JSONL 测试文件（每行一个 JSON 对象），然后运行。")
    # 示例调用:
    # result = run_filter_pipeline("raw_data.jsonl", "filtered_data.jsonl", min_score=40)
    # print(f"过滤完成，保留 {len(result)} 条数据")
