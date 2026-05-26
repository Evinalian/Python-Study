"""
章节：第4章 字典与集合
题目：用 defaultdict 实现倒排索引
类型：进阶练习

题目描述：
倒排索引（Inverted Index）是搜索引擎的核心数据结构：给定一些文档，
对每个词，记录它出现在哪些文档中。

输入：
    docs = {
        1: "Python is great for data science",
        2: "Python is also used for web development",
        3: "Java is popular for enterprise applications",
    }

期望输出（倒排索引）：
    {
        "python": {1, 2},
        "is": {1, 2, 3},
        "great": {1},
        ...
    }

提示：
- 使用 defaultdict(set) 自动创建空集合作为缺省值
- 分词：text.lower().split() 简单按空格拆分
- 用 add() 往 set 里添加文档 ID
"""

from collections import defaultdict

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 build_inverted_index(docs) 函数
# 2. 使用 defaultdict(set) 构建词→文档ID集合的映射
# 3. 对每个文档分词（转小写 + split），将 doc_id 加入对应词的集合
#
# 提示：
# - defaultdict(set) 的缺省值是一个空 set
# - 用 index[word].add(doc_id) 即可
#
# 完成后，运行 python advanced_01_倒排索引.py 测试你的代码。


def build_inverted_index(docs):
    """
    构建倒排索引：词 → 包含该词的所有文档 ID 集合

    参数:
        docs: dict，键为 doc_id，值为文档文本

    返回:
        defaultdict，键为词（小写），值为 doc_id 集合
    """
    pass  # TODO: 实现此函数
