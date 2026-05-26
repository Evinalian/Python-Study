"""
练习 1: 通用文档加载器

场景:
    实现一个通用文档加载器，支持 PDF、Word、TXT 三种格式。
    自动识别扩展名，提取文本和元数据，返回统一格式。

要求:
    1. 实现 UnifiedLoader 类:
       - register(extensions, loader_func): 注册加载器
       - load(file_path): 自动选择加载器
       - load_directory(dir_path): 批量加载目录

    2. 实现三种加载器:
       - load_pdf(file_path): 使用 PyMuPDF 提取文本
       - load_docx(file_path): 使用 python-docx 提取文本
       - load_txt(file_path): 使用内置 open() 读取

    3. 统一返回格式:
       [{"text": "内容", "metadata": {"source": "文件名", "page": 页数, "format": "pdf/docx/txt"}}]

TODO:
    1. 实现 UnifiedLoader 类
    2. 实现三种加载器函数
    3. 注册并测试
"""

import os
from typing import Callable


class UnifiedLoader:
    """
    通用文档加载器。

    职责:
    - 管理不同格式的加载器
    - 根据文件扩展名自动分发
    - 统一返回格式
    """

    def __init__(self):
        # TODO: 初始化 _loaders 字典
        # self._loaders: dict[str, Callable] = {}
        pass

    def register(self, extensions: list[str], loader: Callable):
        """
        注册一个加载器。

        参数:
            extensions: 文件扩展名列表，如 [".pdf", ".PDF"]
            loader: 加载函数，签名为 def loader(file_path: str) -> list[dict]
        """
        # TODO: 将每个扩展名映射到 loader
        pass

    def load(self, file_path: str) -> list[dict]:
        """
        加载单个文件。

        返回格式:
        [{"text": "...", "metadata": {"source": "...", "format": "...", "page": ...}}, ...]
        """
        # TODO:
        # 1. 提取扩展名
        # 2. 查找对应的 loader
        # 3. 调用 loader
        # 4. 为每个文档补充元数据
        pass

    def load_directory(self, dir_path: str, recursive: bool = False) -> list[dict]:
        """
        加载目录中所有支持的文件。

        返回:
            所有文档的列表
        """
        # TODO: 扫描目录，load 每个支持的文件，合并结果
        pass


# ============================================================
# TODO: 实现 PDF 加载器
# ============================================================
def load_pdf(file_path: str) -> list[dict]:
    """
    使用 PyMuPDF 加载 PDF。

    返回: [{"text": "第1页内容", "metadata": {"page": 1}}, ...]

    提示:
    1. import fitz
    2. doc = fitz.open(file_path)
    3. for page in doc: text = page.get_text()
    """
    # TODO: 实现
    pass


# ============================================================
# TODO: 实现 Word 加载器
# ============================================================
def load_docx(file_path: str) -> list[dict]:
    """
    使用 python-docx 加载 Word 文档。

    返回: [{"text": "全部文本内容", "metadata": {"page": 1}}]

    提示:
    1. from docx import Document
    2. doc = Document(file_path)
    3. 遍历 doc.paragraphs，提取文本
    4. 也尝试提取 doc.tables
    """
    # TODO: 实现
    pass


# ============================================================
# TODO: 实现 TXT 加载器
# ============================================================
def load_txt(file_path: str) -> list[dict]:
    """
    加载纯文本文件。

    提示:
    1. 尝试多种编码: utf-8, gbk, latin-1
    2. 如果文件太大（>10000字），按段落切分
    """
    # TODO: 实现
    pass


if __name__ == "__main__":
    loader = UnifiedLoader()
    # TODO: 注册加载器
    # loader.register([".pdf"], load_pdf)
    # loader.register([".docx"], load_docx)
    # loader.register([".txt", ".md", ".py"], load_txt)

    # TODO: 测试
    # docs = loader.load("test.pdf")
    # print(f"加载了 {len(docs)} 个文档片段")

    print("请完成 TODO 后运行。")
