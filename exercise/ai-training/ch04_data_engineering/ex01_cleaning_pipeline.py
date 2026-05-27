"""
练习 1：实现完整的数据清洗 pipeline
===================================

编写一个 Python 脚本，实现对 JSONL 格式文本数据的分阶段清洗。

你需要完成:
1. 定义清洗规则（每个规则是一个函数）:
   - 长度过滤: 丢弃字符数 < min_length 或 > max_length 的文档
   - 语言检测: 使用 langdetect 库，丢弃非目标语言的文档
   - 重复率: 丢弃 n-gram 重复率 > threshold 的文档
   - 特殊字符比: 丢弃特殊字符占比 > threshold 的文档
   - HTML 残留: 丢弃含 HTML 标签的文档
2. 实现一个 Pipeline 类，依次应用规则
3. 每步过滤后记录: 保留数量、丢弃数量、丢弃比例
4. 在清洗日志中记录被丢弃文档的原因分布

思考题:
- 过滤规则的顺序重要吗？为什么？
- 如果先做去重再做长度过滤 vs 先做长度过滤再做去重，有什么不同？
- 为什么需要记录过滤统计，而不仅仅是输出干净数据？
"""

import json
import re
import os
from collections import Counter


# TODO: 实现各个过滤规则函数
def filter_by_length(text, min_length=100, max_length=100000):
    """TODO: 返回 (passed, reason)"""
    pass


def filter_by_language(text, target_lang='en'):
    """TODO: 使用 langdetect 检测语言，返回 (passed, reason)"""
    pass


def filter_by_repetition(text, n=4, max_ratio=0.3):
    """TODO: 返回 (passed, reason)"""
    pass


def filter_by_special_chars(text, max_ratio=0.3):
    """TODO: 返回 (passed, reason)"""
    pass


def filter_html_residue(text):
    """TODO: 检查 HTML 标签残留，返回 (passed, reason)"""
    pass


# TODO: 实现 Pipeline 类
class CleaningPipeline:
    """
    数据清洗流水线。

    TODO: 实现 __init__ 和 run 方法。

    __init__:
        self.rules = []  # 过滤规则列表，每个是 (name, func)

    run:
        对每条文档依次应用规则，记录每步的过滤统计。
    """

    def __init__(self):
        pass  # TODO: 你的代码

    def add_rule(self, name, rule_func):
        """TODO: 添加过滤规则"""
        pass

    def run(self, input_file, output_file, log_file):
        """
        TODO: 执行清洗 pipeline。

        参数:
            input_file: JSONL 输入文件（每行一个 JSON，含 'text' 字段）
            output_file: 输出文件路径
            log_file: 日志文件路径

        返回:
            stats: dict，包含每一步的过滤统计
        """
        pass


if __name__ == "__main__":
    # TODO: 创建测试数据并测试 pipeline
    pass
