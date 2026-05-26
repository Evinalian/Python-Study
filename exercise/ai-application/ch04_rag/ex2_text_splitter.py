"""
练习 2: 文本切块对比

场景:
    对比 CharacterTextSplitter 和 RecursiveCharacterTextSplitter 的效果。
    用同一篇文本测试，分析两种切分策略的差异。

要求:
    1. 实现 CharacterTextSplitter 类（固定大小 + overlap）
    2. 实现 RecursiveCharacterTextSplitter 类（递归分隔符）
    3. 准备一篇测试文本（至少 2000 字）
    4. 用两种 splitter 切分同一文本
    5. 对比分析:
       - chunk 数量和平均大小
       - 是否在句子中间断开（统计末尾非标点符号的 chunk 比例）
       - 语义完整性（主观判断）

TODO:
    1. 实现 CharacterTextSplitter
    2. 实现 RecursiveCharacterTextSplitter
    3. 实现 evaluate_chunks(chunks) 评估函数
    4. 运行对比并输出报告
"""

import re


# ============================================================
# TODO 1: CharacterTextSplitter
# ============================================================
class CharacterTextSplitter:
    """
    固定大小字符切分器。

    参数:
        chunk_size: 每块最大字符数
        chunk_overlap: 相邻块重叠字符数
        separator: 分隔符
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150, separator: str = "\n\n"):
        # TODO
        pass

    def split(self, text: str) -> list[str]:
        # TODO: 实现固定大小切分逻辑
        # 1. 如果文本长度 <= chunk_size，直接返回
        # 2. 按 separator 分割
        # 3. 累积片段直到接近 chunk_size
        # 4. 添加 overlap
        pass


# ============================================================
# TODO 2: RecursiveCharacterTextSplitter
# ============================================================
class RecursiveCharacterTextSplitter:
    """
    递归字符切分器。

    分隔符优先级: "\\n\\n" → "\\n" → "。" → "，" → " " → ""
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        # TODO
        pass

    def split(self, text: str) -> list[str]:
        # TODO: 实现递归切分逻辑
        # 1. 用当前分隔符切分
        # 2. 如果某个片段太长，递归用下一级分隔符
        # 3. 添加 overlap
        pass


# ============================================================
# TODO 3: 评估函数
# ============================================================
def evaluate_chunks(chunks: list[str], label: str) -> dict:
    """
    评估切块质量。

    返回:
    {
        "label": "CharacterTextSplitter",
        "chunk_count": 5,
        "avg_length": 750,
        "min_length": 200,
        "max_length": 1200,
        "std_length": 180,
        "mid_sentence_ratio": 0.3,    # 在句子中间断开的比例
        "overlap_ratio": 0.15,         # 内容重叠比例
        "empty_count": 0               # 空 chunk 数量
    }

    提示:
    - mid_sentence: 文本末尾不是 。！？.!? 等标点
    - 检查 chunk 之间的内容重叠
    """
    # TODO: 实现评估
    pass


def compare_splitters(text: str) -> None:
    """
    对比两种 splitter 的效果并打印报告。
    """
    # TODO:
    # 1. 创建两个 splitter 实例（使用相同的 chunk_size 和 overlap）
    # 2. 分别切分
    # 3. 调用 evaluate_chunks 评估
    # 4. 打印对比表格
    pass


if __name__ == "__main__":
    # 准备测试文本
    test_text = """
    TODO: 准备一篇至少 2000 字的测试文本。
    可以是教程文章中摘录的几段，也可以是新闻文章。
    确保包含: 标题、段落、列表、代码块（如果可能）。
    要求文本有自然段落结构，以便验证切分器是否在合理位置断开。
    """

    # compare_splitters(test_text * 5)  # 重复5次以增加长度
    print("请完成 TODO 后运行。")
