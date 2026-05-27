# 第04章 RAG架构深入

## 学习目标

完成本章后，你将能够：

1. 理解 RAG 的整体架构：离线索引管道 + 在线检索 + 增强生成
2. 用多种方式加载不同格式的文档（PDF、Word、网页、Markdown）
3. 掌握 4 种文本切块策略，理解 chunk_size 和 chunk_overlap 的调参原理
4. 实现稠密检索、稀疏检索、混合检索三种检索策略
5. 使用 Cross-Encoder 做 ReRank 重排序提升检索精度
6. 处理检索后的去重、过滤、上下文分配
7. 设计 RAG 的生成 prompt，包含引用标注和 fallback 策略
8. 构建一个完整的端到端文档问答系统

## 前置知识

- Python 基础：文件 I/O、列表推导式、异常处理
- LLM API 调用基础（第01章 Prompt Engineering）
- 对向量（Vector）有基本概念（一组浮点数表示语义）
- 了解余弦相似度的基本思想（两个向量方向的接近程度）
- pip 安装第三方库的能力

---

## 1. RAG 整体架构

### 1.1 什么是 RAG —— 给 LLM 装上"外挂硬盘"

传统 LLM 的知识完全来自训练数据，存在两个根本性限制：**知识截止日期**（训练后发生的事一概不知）和**私有知识盲区**（你的公司文档、产品手册从未进入训练集）。RAG（Retrieval-Augmented Generation，检索增强生成）解决的就是这两个问题——它让 LLM 在回答问题之前，先去一个外部知识库中"查资料"，然后基于查到的资料生成回答。

通俗比喻：传统 LLM 像一个只靠记忆回答的专家——记性好但知识有截止日期，且没见过你公司的内部文件。RAG 给这个专家配了一个实时联网的图书馆——每次被提问，先去图书馆翻相关书籍，然后基于书里的内容给出有据可查的回答。

```
传统 LLM:
  用户: "公司最新的请假政策是什么？"
  LLM: "抱歉，我的训练数据截止到..."  ← 没有私有知识

RAG:
  用户: "公司最新的请假政策是什么？"
    ↓
  [检索阶段] 从公司文档库中搜索最相关的片段
    ↓
  找到: "员工每年享有5天带薪病假，3天事假..."
    ↓
  [生成阶段] 基于检索到的内容生成回答
    ↓
  LLM: "根据公司政策，员工每年享有5天带薪病假..."
```

### 1.2 两阶段架构 —— 离线管道 + 在线查询

RAG 系统在架构上清晰地分为两个阶段，它们有着完全不同的性能要求和触发时机。

**离线索引阶段（Offline Indexing）**：发生在用户提问之前，是一个批处理任务。它的目标是把你拥有的所有文档（PDF、Word、网页等）转换成可供快速检索的索引。这个阶段的完整流水线是：加载文档 -> 文本清洗 -> 切块（chunking）-> Embedding（向量化）-> 存入向量数据库。离线阶段不需要实时响应，可以花较多时间做高质量的文本处理，而且只需要在文档更新时重新执行。

**在线查询阶段（Online Retrieval + Generation）**：发生在用户提问时，是一个实时任务。它的目标是在毫秒级时间内从海量 chunk 中找到最相关的那几条，然后组装成 prompt 喂给 LLM 生成回答。在线阶段的流水线是：用户问题 -> Query Embedding -> 向量检索（召回 top-k）-> ReRank 重排序 -> 上下文组装 -> LLM 生成回答。

把这两个阶段分开设计，是一种典型的"空间换时间"策略——离线阶段花时间把文档处理好、存成容易查的格式，在线阶段就可以在毫秒级完成检索。如果没有离线阶段的预处理，每次查询都要现场加载和切分所有文档，响应时间将是不可接受的。

```
┌─────────────────────────────────────────────────────┐
│                 离线索引阶段 (Offline)                │
│                                                      │
│  文档 → 加载 → 切块 → Embedding → 向量数据库         │
│                                                      │
│  这个阶段只需要执行一次（或文档更新时重新执行）        │
│  ⏱ 性能目标: 分钟级                                  │
│  🔄 触发时机: 文档导入/更新时                          │
└─────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────┐
│                 在线查询阶段 (Online)                 │
│                                                      │
│  用户问题 → Embedding → 检索 → 排序 → 生成回答       │
│                                                      │
│  每次查询都执行                                       │
│  ⏱ 性能目标: 毫秒-秒级                               │
│  🔄 触发时机: 用户每次提问时                            │
└─────────────────────────────────────────────────────┘
```

### 1.3 架构图代码化

下面用 Python 代码勾勒 RAG 系统的整体骨架，后续各节将逐一展开每个组件的实现。

```python
"""
RAG 系统架构概览 —— 用代码描述整个流程。
这是一个高层视图，后续各节展开每个组件。
"""
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    """文档对象 —— RAG 中的基本数据单元"""
    content: str               # 文本内容
    metadata: dict             # 元数据（来源、页码、标题等）
    embedding: list[float] = None  # 向量表示（加载后计算）


class RAGPipeline:
    """
    RAG 管道的完整流程。

    离线阶段:
        1. DocumentLoader 加载文档
        2. TextSplitter 切分成块
        3. EmbeddingModel 计算向量
        4. VectorStore 存储向量 + 文本

    在线阶段:
        5. QueryEmbedding 计算查询向量
        6. VectorStore.search 检索 top-k
        7. ReRanker 重排序
        8. Generator 基于检索结果生成回答
    """

    def __init__(self):
        self.documents: list[Document] = []
        self.chunks: list[Document] = []

    def offline_index(self, file_paths: list[str]):
        """离线索引阶段"""
        # 1. 加载文档
        # 2. 切块
        # 3. 计算向量
        # 4. 存入数据库
        pass

    def online_query(self, question: str) -> str:
        """在线查询阶段"""
        # 5. 查询向量化
        # 6. 检索 top-k
        # 7. 排序
        # 8. 生成回答
        pass


print("RAG = 检索(Retrieval) + 增强(Augmented) + 生成(Generation)")
print("核心思想: 先查后答，而不是全靠模型记忆")
```

---

## 2. 文档加载

### 2.1 PDF 文档加载 —— 三种方案的场景选择

PDF 是最常见的文档格式，但也是加载难度最高的——因为它既有"扫描版 PDF"（图片嵌入文字），也有"电子版 PDF"（原生文字），还可能是两者的混合。不同的 PDF 加载库各有优势领域：

- **PyMuPDF (fitz)**：速度和内存效率最高，对原生文字 PDF 的提取效果最好。适合大多数纯文字 PDF 场景。如果文档量很大（数千页），PyMuPDF 是最佳选择。
- **pdfplumber**：表格提取能力最强，能保留表格的结构信息。适合财务报表、数据报告等表格密集型 PDF。文本提取速度比 PyMuPDF 慢，但更准确。
- **unstructured**：通用性最强，不仅支持 PDF，还支持 Word、HTML、Markdown 等多种格式。自动识别文档结构元素（标题、段落、列表、表格），适合排版复杂的文档。

选择建议：大部分项目以 PyMuPDF 为主（覆盖 80% 的文档），遇到表格需求时补充 pdfplumber，需要多格式支持时引入 unstructured。

```python
"""
PDF 文档加载 —— 三种常用库的对比和使用。

PyMuPDF (fitz): 速度快，支持复杂排版
pdfplumber:  表格提取能力最强
unstructured: 支持多种格式，自动识别文档结构
"""
import os


# ============================================================
# 方式1: PyMuPDF (fitz) —— 推荐用于大多数场景
# ============================================================
def load_pdf_with_pymupdf(file_path: str) -> list[dict]:
    """
    使用 PyMuPDF 加载 PDF，每页作为一个文档块。

    安装: pip install PyMuPDF

    参数:
        file_path: PDF 文件路径

    返回:
        [{"page": 1, "text": "页面内容...", "metadata": {...}}, ...]
    """
    try:
        import fitz  # PyMuPDF 的导入名
    except ImportError:
        print("请安装 PyMuPDF: pip install PyMuPDF")
        return []

    doc = fitz.open(file_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()

        # 提取页面元数据
        pages.append(
            {
                "page": page_num + 1,
                "text": text,
                "metadata": {
                    "source": file_path,
                    "page_number": page_num + 1,
                    "total_pages": len(doc),
                    "char_count": len(text),
                },
            }
        )

    doc.close()
    print(f"[PyMuPDF] 加载完成: {len(pages)} 页, 总字符数: {sum(p['metadata']['char_count'] for p in pages)}")
    return pages


# ============================================================
# 方式2: pdfplumber —— 推荐用于表格密集型 PDF
# ============================================================
def load_pdf_with_pdfplumber(file_path: str) -> list[dict]:
    """
    使用 pdfplumber 加载 PDF，尤其适合提取表格。

    安装: pip install pdfplumber

    与 PyMuPDF 的区别:
    - pdfplumber.table 提取: 高精度表格识别
    - pdfplumber 的文本提取更慢但更准确（对复杂排版）
    """
    try:
        import pdfplumber
    except ImportError:
        print("请安装 pdfplumber: pip install pdfplumber")
        return []

    pages = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            # 尝试提取表格
            tables = page.extract_tables()

            pages.append(
                {
                    "page": page_num + 1,
                    "text": text,
                    "tables": tables,  # pdfplumber 特有的表格数据
                    "metadata": {
                        "source": file_path,
                        "page_number": page_num + 1,
                        "has_tables": len(tables) > 0,
                    },
                }
            )

    print(f"[pdfplumber] 加载完成: {len(pages)} 页")
    return pages


# ============================================================
# 方式3: unstructured —— 支持多格式的通用加载器
# ============================================================
def load_pdf_with_unstructured(file_path: str) -> list[dict]:
    """
    使用 unstructured 库加载 PDF，自动识别文档元素。

    安装: pip install "unstructured[pdf]"

    特点:
    - 自动识別标题、段落、列表、表格
    - 统一的 Document 对象
    - 支持 PDF、Word、HTML、Markdown 等多种格式
    """
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError:
        print("请安装 unstructured: pip install 'unstructured[pdf]'")
        return []

    elements = partition_pdf(file_path)

    pages = []
    for element in elements:
        pages.append(
            {
                "type": str(element.__class__.__name__),  # Title, NarrativeText, Table 等
                "text": str(element),
                "metadata": {
                    "source": file_path,
                    "page_number": element.metadata.page_number if element.metadata else None,
                },
            }
        )

    print(f"[unstructured] 加载完成: {len(pages)} 个文档元素")
    return pages


if __name__ == "__main__":
    # 创建测试 PDF 的代码（如果不存在的话）
    print("PDF 加载对比:")
    print("  1. PyMuPDF: 速度快，适合纯文本 PDF")
    print("  2. pdfplumber: 表格提取强，适合报表类 PDF")
    print("  3. unstructured: 自动识别结构，适合排版复杂的 PDF")
    print("\n实际项目中，建议以 PyMuPDF 为主，遇到表格时补充 pdfplumber")
```

### 2.2 Word 文档加载

Word 文档（.docx）的结构比 PDF 更规整——它是基于 XML 的标记语言，天然区分段落、表格和标题样式。这种结构信息在 RAG 中非常宝贵——如果切块时能知道"这是一个标题"、"这是一个表格"，就可以避免在标题和正文之间切断语义。

```python
"""
Word 文档 (.docx) 加载。

主要库: python-docx
特点: 可以区分段落、表格、标题样式
"""
import os


def load_docx(file_path: str) -> list[dict]:
    """
    使用 python-docx 加载 Word 文档。

    安装: pip install python-docx

    参数:
        file_path: .docx 文件路径

    返回:
        [{"type": "paragraph"/"table", "text": "...", "style": "Heading 1", ...}, ...]
    """
    try:
        from docx import Document
    except ImportError:
        print("请安装 python-docx: pip install python-docx")
        return []

    doc = Document(file_path)
    elements = []

    # 遍历文档的 body 元素（按出现顺序）
    from docx.oxml.ns import qn

    for block in doc.element.body:
        # 处理段落
        if block.tag == qn("w:p"):
            para = None
            for p in doc.paragraphs:
                if p._element is block:
                    para = p
                    break
            if para and para.text.strip():
                elements.append(
                    {
                        "type": "paragraph",
                        "text": para.text,
                        "style": para.style.name if para.style else "Normal",
                        "metadata": {"source": file_path},
                    }
                )

        # 处理表格
        elif block.tag == qn("w:tbl"):
            for table in doc.tables:
                if table._element is block:
                    # 将表格转换为文本
                    rows = []
                    for row in table.rows:
                        cells = [cell.text for cell in row.cells]
                        rows.append(" | ".join(cells))
                    table_text = "\n".join(rows)

                    elements.append(
                        {
                            "type": "table",
                            "text": table_text,
                            "metadata": {"source": file_path, "rows": len(rows)},
                        }
                    )
                    break

    print(f"[python-docx] 加载完成: {len(elements)} 个元素")
    return elements


def load_docx_simple(file_path: str) -> str:
    """
    最简单的方式: 只提取纯文本，忽略格式。
    如果不需要区分段落/表格，这是最直接的方法。
    """
    try:
        from docx import Document
    except ImportError:
        return ""

    doc = Document(file_path)
    full_text = []

    for para in doc.paragraphs:
        if para.text.strip():
            # 保留标题结构
            if para.style.name.startswith("Heading"):
                level = para.style.name.split()[-1]
                full_text.append(f"{'#' * int(level)} {para.text}")
            else:
                full_text.append(para.text)

    return "\n\n".join(full_text)


if __name__ == "__main__":
    print("Word 文档加载说明:")
    print("  - 简单场景用 load_docx_simple() 提取纯文本")
    print("  - 需要区分段落/表格用 load_docx()")
    print("  - 保留 Heading 样式有助于切块时保持语义完整性")
```

### 2.3 网页加载

网页加载的核心挑战不是"怎么获取 HTML"（requests 一行搞定），而是"怎么从 HTML 中提取正文"。现代网页充满了导航栏、广告、侧边栏、页脚等噪声内容，如果不加清理直接提取文本，噪声会严重干扰后续的检索质量。

关键步骤：(1) 用 BeautifulSoup 解析 DOM；(2) 移除 `<script>`、`<style>`、`<nav>`、`<footer>`、`<aside>` 等非内容标签；(3) 优先从 `<main>`、`<article>` 等语义标签中提取正文；(4) 合并多余空行。

```python
"""
网页加载 —— 从 URL 抓取内容并提取正文。

核心思路:
1. requests 获取 HTML
2. BeautifulSoup 解析 DOM
3. 移除无关元素（script, style, nav, footer）
4. 提取正文文本
"""
import re


def load_webpage(url: str) -> dict:
    """
    从网页加载文本内容。

    安装: pip install requests beautifulsoup4

    参数:
        url: 网页 URL

    返回:
        {"title": "...", "text": "...", "url": url, "metadata": {...}}
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("请安装依赖: pip install requests beautifulsoup4")
        return {}

    # 设置 User-Agent 避免被拒绝
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding  # 自动检测编码
    except requests.RequestException as e:
        print(f"请求失败: {e}")
        return {"error": str(e), "url": url}

    soup = BeautifulSoup(response.text, "html.parser")

    # 提取标题
    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)

    # 移除不需要的元素
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # 提取正文
    # 策略1: 优先使用语义标签
    main_content = soup.find("main") or soup.find("article") or soup.body

    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # 清理文本: 合并多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return {
        "title": title,
        "text": text,
        "url": url,
        "metadata": {
            "source": url,
            "char_count": len(text),
            "title": title,
        },
    }


def load_multiple_webpages(urls: list[str]) -> list[dict]:
    """批量加载多个网页"""
    results = []
    for url in urls:
        print(f"加载: {url}")
        result = load_webpage(url)
        if "error" not in result:
            results.append(result)
    return results


if __name__ == "__main__":
    print("网页加载说明:")
    print("  1. requests 获取 HTML")
    print("  2. BeautifulSoup 解析 + 移除无关标签")
    print("  3. 优先提取 <main>/<article> 语义标签")
    print("  4. 清理多余空行")
```

### 2.4 统一文档加载器

当项目需要支持多种文档格式时，为每种格式写一个独立的加载函数会导致代码分散。策略模式（Strategy Pattern）是解决这个问题的经典方案——把每种文件格式对应一个加载策略，注册到一个统一的加载器中，运行时根据扩展名自动匹配。

```python
"""
统一文档加载器 —— 根据文件扩展名自动选择加载方法。

设计模式: Strategy Pattern（策略模式）
每种文件格式对应一个加载策略，运行时自动匹配。
"""
import os
from typing import Callable


class UnifiedLoader:
    """
    统一的文档加载器。

    注册不同文件类型的加载策略，根据扩展名自动分发。
    """

    def __init__(self):
        self._loaders: dict[str, Callable] = {}

    def register(self, extensions: list[str], loader: Callable):
        """注册一个文档加载器"""
        for ext in extensions:
            self._loaders[ext.lower()] = loader

    def load(self, file_path: str) -> list[dict]:
        """
        根据文件扩展名自动加载文档。

        参数:
            file_path: 文件路径

        返回:
            文档内容列表
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext not in self._loaders:
            raise ValueError(f"不支持的文件类型: {ext}。支持的类型: {list(self._loaders.keys())}")

        print(f"使用 {self._loaders[ext].__name__} 加载 {file_path}")
        return self._loaders[ext](file_path)

    def load_directory(self, dir_path: str, recursive: bool = True) -> list[dict]:
        """
        加载目录中所有支持的文档。

        参数:
            dir_path: 目录路径
            recursive: 是否递归子目录

        返回:
            所有文档的内容列表
        """
        import glob

        all_docs = []
        supported_exts = list(self._loaders.keys())

        for ext in supported_exts:
            pattern = os.path.join(dir_path, f"**/*{ext}") if recursive else os.path.join(dir_path, f"*{ext}")
            for file_path in glob.glob(pattern, recursive=recursive):
                try:
                    docs = self.load(file_path)
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"加载 {file_path} 失败: {e}")

        print(f"共加载 {len(all_docs)} 个文档片段")
        return all_docs


# 使用示例
if __name__ == "__main__":
    loader = UnifiedLoader()

    # 注册各种文件类型的加载器
    # loader.register([".pdf"], load_pdf_with_pymupdf)
    # loader.register([".docx"], load_docx)
    # loader.register([".txt", ".md"], load_text_file)  # 纯文本加载器
    # loader.register([".html", ".htm"], load_html_file)  # 本地 HTML 加载器

    # 加载单个文件
    # docs = loader.load("report.pdf")

    # 加载整个目录
    # all_docs = loader.load_directory("./documents/")

    print("UnifiedLoader 设计思路:")
    print("  1. 注册: 每种文件格式对应一个加载函数")
    print("  2. 自动匹配: 根据扩展名选择加载器")
    print("  3. 批量加载: load_directory 扫描整个目录")
```

---

## 3. 文本切块策略

### 3.1 为什么需要切块 —— 三个根本原因

把长文档切成小块不是可有可无的操作，而是由三个硬性约束决定的。

**第一个约束：上下文窗口有限**。虽然 GPT-4o 支持 128K token 的上下文，但在 RAG 场景中，上下文窗口是共享资源——你需要同时放入 system prompt、检索到的多个 chunk、用户问题、输出预留空间。如果把整个 100 页 PDF（10 万 token）作为"一个 chunk"塞进去，就没空间放其他内容了，而且模型对超长上下文中部的信息关注度会下降（"Lost in the Middle" 现象）。

**第二个约束：检索精度**。Embedding 模型将文本映射为向量时，本质上是将文本的语义压缩到固定维度（如 1536 维）。文本越长，这个压缩的"信息密度"越高，但区分度越低。如果一整个文档作为一个 embedding，它包含了"公司介绍 + 产品说明 + 技术支持 + 附录"的所有语义混合，对于"如何重置密码"这种具体问题，这个混合向量的相似度不会很高。切成小块后，每个块有一个明确的单一语义，检索精度大幅提升。

**第三个约束：成本**。每个检索到的 token 都要作为 prompt 的一部分计费。如果每次检索返回的是 5000 字的整页文档，其中只有 200 字和问题相关，你为 4800 字的噪声付了费。切块让你只把最相关的内容喂给 LLM。

### 3.2 固定大小切块 —— 最简单的入口

固定大小切块按指定的字符数切分文本，是最容易理解和实现的切块方式。适合格式规整、段落结构清晰的文档。缺点是不考虑语义边界——可能在句子中间切断。

```python
"""
固定大小切块 —— 最简单的切块方式。

按固定字符数切分，不考虑语义边界。
适合: 格式规整、段落清晰的文档。
不适合: 代码、表格、对话。
"""


class CharacterTextSplitter:
    """
    固定大小字符切分器。

    参数:
        chunk_size: 每块的目标字符数
        chunk_overlap: 相邻块之间的重叠字符数
        separator: 分隔符（默认按段落分）
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separator: str = "\n\n"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split(self, text: str) -> list[str]:
        """
        将文本切分成多个 chunk。

        算法:
        1. 如果文本长度 <= chunk_size，直接返回
        2. 用 separator 分割文本
        3. 累积片段直到接近 chunk_size
        4. 保留 chunk_overlap 长度的重叠
        """
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # 按分隔符切分
        segments = text.split(self.separator)
        chunks = []
        current_chunk = ""
        current_length = 0

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            segment_length = len(segment)

            # 如果当前 chunk 加上新片段不会超限
            if current_length + segment_length <= self.chunk_size:
                if current_chunk:
                    current_chunk += self.separator + segment
                else:
                    current_chunk = segment
                current_length += segment_length
            else:
                # 保存当前 chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # 如果单个片段就超过 chunk_size，需要进一步切分
                if segment_length > self.chunk_size:
                    # 按句子切分（简化版）
                    sub_chunks = self._split_long_segment(segment)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                    current_length = 0
                else:
                    # 新片段作为新 chunk 的开始
                    current_chunk = segment
                    current_length = segment_length

        # 保存最后一个 chunk
        if current_chunk:
            chunks.append(current_chunk)

        # 添加重叠: 每个 chunk 的结尾包含下一个 chunk 的开头
        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_long_segment(self, text: str) -> list[str]:
        """处理超过 chunk_size 的单个片段"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # 尽量在标点符号处断开
            if end < len(text):
                # 在 chunk_size 范围内找最后一个句号/问号/感叹号
                for punct in ["。", "！", "？", "\n", ".", "!", "?"]:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start + self.chunk_size // 2:
                        end = last_punct + 1
                        break
            chunks.append(text[start:end])
            start = end - self.chunk_overlap if end < len(text) else end
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """在 chunk 之间添加重叠"""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped = []
        for i in range(len(chunks)):
            if i == 0:
                overlapped.append(chunks[i])
            else:
                # 前一个 chunk 的结尾作为当前 chunk 的开头
                prev_end = chunks[i - 1][-self.chunk_overlap :] if len(chunks[i - 1]) > self.chunk_overlap else chunks[i - 1]
                overlapped.append(prev_end + "\n\n" + chunks[i])

        return overlapped


if __name__ == "__main__":
    sample_text = """
第一段: 人工智能是计算机科学的一个分支，它企图了解智能的实质，
并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。

第二段: 机器学习是实现人工智能的一种方法。通过从数据中学习规律，
机器学习算法可以在没有明确编程的情况下做出预测或决策。
深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的复杂模式。

第三段: 近年来，大语言模型(LLM)的兴起推动了人工智能的新一轮发展。
GPT-4、Claude等模型在各种任务上展现出接近甚至超越人类的能力。
然而，LLM也面临幻觉、偏见、隐私等挑战。
"""

    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split(sample_text)

    print(f"原文长度: {len(sample_text)}")
    print(f"切分成 {len(chunks)} 个 chunk:\n")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ({len(chunk)} 字) ---")
        print(chunk)
        print()
```

### 3.3 递归字符切块 —— 保留语义边界的进阶方案

递归字符切块比固定大小切块更智能的地方在于：它不是简单地"数到 1000 字就切一刀"，而是按照分隔符优先级**尝试**在语义边界处切分。优先在段落边界（`\n\n`）切，如果切出来的块还是太大，就降到行边界（`\n`），再降到句子边界（`。`），最坏情况逐字切。这种"嵌套降级"策略最大程度保证了每个 chunk 的语义完整性。

```python
"""
递归字符切块 —— LangChain 风格的 RecursiveCharacterTextSplitter。

与固定大小切块的区别:
- 不是简单地按固定字符数切
- 而是按优先级递减的分隔符尝试切分
- 如果按 "\\n\\n" 切出来太大，改用 "\\n"
- 如果按 "\\n" 切出来还太大，改用 "。" 
- 如果还太大，改用 "，" 甚至逐字符切
"""


class RecursiveCharacterTextSplitter:
    """
    递归字符切分器。

    分隔符优先级（从粗到细）:
    1. "\\n\\n" (段落)
    2. "\\n"   (行)
    3. "。"    (句子)
    4. "，"    (子句)
    5. ""      (逐字符)

    每一层尝试用当前分隔符切分，如果分出的片段太长则降到下一层。
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "，", " ", ""]

    def split(self, text: str) -> list[str]:
        """递归切分文本"""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """递归切分核心逻辑"""
        # 文本已经足够短，直接返回
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # 用当前分隔符切分
        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            splits = text.split(separator)
        else:
            # 最后的分隔符是 ""，即逐字符切分
            splits = list(text)

        # 合并片段直到接近 chunk_size
        chunks = []
        current_chunk = ""

        for split in splits:
            if not split.strip():
                continue

            # 如果当前片段本身就超过 chunk_size，递归用下一层分隔符
            if len(split) > self.chunk_size and remaining_separators:
                # 先把当前累积的保存
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                # 递归处理超长片段
                sub_chunks = self._split_text(split, remaining_separators)
                chunks.extend(sub_chunks)
                continue

            # 尝试把 split 加入 current_chunk
            test_chunk = current_chunk + (separator if current_chunk else "") + split

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # 超了，保存当前
                if current_chunk:
                    chunks.append(current_chunk)

                # 如果单个 split 就太大且没有更细的分隔符
                if len(split) > self.chunk_size and not remaining_separators:
                    # 强制按字符切
                    for i in range(0, len(split), self.chunk_size):
                        chunks.append(split[i : i + self.chunk_size])
                    current_chunk = ""
                else:
                    current_chunk = split

        # 最后一个
        if current_chunk:
            chunks.append(current_chunk)

        # 如果 separator 切出来的块都太大，降到更细的分隔符
        if remaining_separators and any(len(c) > self.chunk_size for c in chunks):
            return self._split_text(text, remaining_separators)

        return chunks


if __name__ == "__main__":
    sample = """
人工智能概述

人工智能(AI)是计算机科学的一个分支。它致力于创造能够执行通常需要人类智能的任务的系统。

机器学习是AI的核心方法之一。通过算法从数据中学习，而不是通过显式编程来执行任务。

深度学习使用多层神经网络。它在图像识别、语音识别和自然语言处理方面取得了突破性进展。

大语言模型(LLM)如GPT-4代表了AI的最新进展。这些模型可以生成文本、翻译语言、编写代码等。

然而LLM也面临挑战。幻觉问题、偏见问题、隐私问题都需要解决。

未来AI将更加普及。从自动驾驶到医疗诊断，AI正在改变世界。
"""

    splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=30)
    chunks = splitter.split(sample)

    print(f"原文长度: {len(sample)}")
    print(f"递归切分: {len(chunks)} 个 chunk\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1} ({len(chunk)} 字):")
        print(f"  [{chunk[:100]}{'...' if len(chunk)>100 else ''}]")
```

### 3.4 chunk_size 和 chunk_overlap 的场景选择指南

`chunk_size` 和 `chunk_overlap` 不是拍脑袋设定的魔法数字，而是需要根据文档类型和任务目标来调整的参数。核心权衡：

- **chunk_size 太小**（如 200 字）：检索精度高（每个 chunk 语义单一），但信息碎片化——检索到的 chunk 可能只包含"Python 是一种..."，缺了后面的"解释型语言"，上下文不完整。
- **chunk_size 太大**（如 5000 字）：上下文完整，但检索精度下降——chunk 中混合了多个语义，embedding 向量模糊，且检索结果中大量内容与问题无关。
- **chunk_overlap 太小**（如 0-50）：信息可能在 chunk 边界丢失——如果关键概念恰好出现在两个 chunk 的分界处，两个 chunk 都只有片段。
- **chunk_overlap 太大**（如 500）：信息冗余，相同内容出现在多个 chunk 中，token 消耗增加，且检索结果中重复内容多。

不同场景的推荐值：

| 文档类型 | chunk_size | chunk_overlap | 原因 |
|----------|-----------|---------------|------|
| 技术文档 | 800-1200 | 100-200 | 段落结构清晰，中等块保留完整概念 |
| 法律合同 | 500-800 | 100-150 | 条款独立性较强，小块有助于精确检索 |
| 学术论文 | 1000-1500 | 150-250 | 推理链长，大块保留论证过程 |
| 对话记录 | 300-500 | 50-100 | 单条对话短，小块保持问答对应 |
| 代码仓库 | 1500-3000 | 0 | 函数/类不应被切断，重叠导致代码不完整 |

```python
"""
chunk_size 和 chunk_overlap 的参数调优指南。

这两个参数直接影响 RAG 的检索质量和生成质量，需要根据文档类型和任务调整。
"""


class ChunkParamGuide:
    """
    chunk_size 和 chunk_overlap 的调参建议。

    核心权衡:
    - chunk_size 太小: 信息碎片化，缺少上下文
    - chunk_size 太大: 检索精度下降，噪声增加
    - chunk_overlap 太小: 关键信息可能落在边界上
    - chunk_overlap 太大: 信息冗余，token 浪费
    """

    RECOMMENDATIONS = {
        "技术文档": {
            "chunk_size": "800-1200",
            "chunk_overlap": "100-200",
            "reason": "技术文档段落结构清晰，中等块大小可以保留完整概念",
        },
        "法律合同": {
            "chunk_size": "500-800",
            "chunk_overlap": "100-150",
            "reason": "条款独立性较强，较小块有助于精确检索特定条款",
        },
        "学术论文": {
            "chunk_size": "1000-1500",
            "chunk_overlap": "150-250",
            "reason": "推理链较长，需要较大的块保留论证过程",
        },
        "对话记录": {
            "chunk_size": "300-500",
            "chunk_overlap": "50-100",
            "reason": "单条对话较短，小块可以保持问题-回答对应关系",
        },
        "代码仓库": {
            "chunk_size": "1500-3000",
            "chunk_overlap": "0",
            "reason": "函数/类不应被切断，重叠可能导致代码不完整",
        },
    }

    @staticmethod
    def grid_search_suggestion(base_chunk_size: int) -> list[tuple[int, int]]:
        """
        为 A/B 测试生成候选参数组合。

        参数:
            base_chunk_size: 根据文档类型推荐的基准值

        返回:
            [(chunk_size, chunk_overlap), ...] 建议测试的组合
        """
        candidates = []
        for ratio in [0.5, 0.75, 1.0, 1.5, 2.0]:
            size = int(base_chunk_size * ratio)
            for overlap_ratio in [0.1, 0.15, 0.2, 0.25]:
                overlap = int(size * overlap_ratio)
                candidates.append((size, overlap))
        return candidates

    @staticmethod
    def evaluate_chunk_quality(chunks: list[str]) -> dict:
        """
        评估切块质量的简单指标。

        返回:
            {
                "avg_length": 平均长度,
                "std_length": 长度标准差,
                "too_short_ratio": 过短块比例 (<100字),
                "too_long_ratio": 过长块比例 (>2000字),
                "mid_sentence_ratio": 以非句号结尾的比例,
            }
        """
        if not chunks:
            return {}

        lengths = [len(c) for c in chunks]
        avg = sum(lengths) / len(lengths)
        variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)
        std = variance ** 0.5

        too_short = sum(1 for l in lengths if l < 100) / len(lengths)
        too_long = sum(1 for l in lengths if l > 2000) / len(lengths)

        # 检查是否在句子中间断开
        mid_sentence = sum(
            1 for c in chunks if c.strip() and c.strip()[-1] not in "。！？.!?\n"
        ) / len(chunks)

        return {
            "avg_length": int(avg),
            "std_length": int(std),
            "too_short_ratio": f"{too_short:.1%}",
            "too_long_ratio": f"{too_long:.1%}",
            "mid_sentence_ratio": f"{mid_sentence:.1%}",
            "total_chunks": len(chunks),
        }


if __name__ == "__main__":
    print("=== Chunk 参数调优指南 ===\n")
    for doc_type, params in ChunkParamGuide.RECOMMENDATIONS.items():
        print(f"{doc_type}:")
        print(f"  chunk_size={params['chunk_size']}, overlap={params['chunk_overlap']}")
        print(f"  原因: {params['reason']}\n")

    print("=== A/B 测试建议组合 (基准 1000) ===\n")
    candidates = ChunkParamGuide.grid_search_suggestion(1000)
    for size, overlap in candidates[:6]:  # 只展示前6组
        print(f"  (chunk_size={size:>4}, overlap={overlap:>3})")
```

---

## 4. 检索策略

### 4.1 稠密检索（Embedding + 余弦相似度）—— 语义匹配

稠密检索是目前 RAG 系统的主流检索方式。它的核心思想是：把文本转换为固定维度的向量（embedding），然后通过计算查询向量和文档向量之间的余弦相似度来度量语义相关性。

稠密检索的**优势**在于语义匹配——"汽车"和"轿车"在词面上完全不同，但在向量空间中距离很近，因为它们的语义相似。这使得它能处理同义词、口语化表达和跨语言匹配。

稠密检索的**劣势**在于：(1) 需要 embedding 模型（增加了计算成本和调用延迟）；(2) 对精确关键词匹配不如 BM25——搜索产品编号 "AB-12345" 时，embedding 模型可能不理解这个编号的含义；(3) embedding 质量高度依赖模型，不同模型对同一文本的编码结果差异可能很大。

```python
"""
稠密检索 (Dense Retrieval)。

原理:
1. 将所有文档块转换为向量（embedding）
2. 将用户查询转换为向量
3. 计算查询向量与每个文档块向量的余弦相似度
4. 返回相似度最高的 top-k 个文档块

优势: 语义匹配，同义词也能检索到
劣势: 需要 embedding 模型，处理精确关键词匹配不如 BM25
"""
import numpy as np
from typing import List, Tuple


class DenseRetriever:
    """
    稠密检索器。

    使用 embedding 模型将文本转为向量，通过余弦相似度检索。
    """

    def __init__(self, embedding_model):
        """
        参数:
            embedding_model: 一个包含 embed(texts) 方法的对象
                             (如 OpenAIEmbeddings, SentenceTransformer 等)
        """
        self.embedding_model = embedding_model
        self.documents: list[dict] = []  # [{"text": ..., "embedding": [...], "metadata": {...}}, ...]

    def add_documents(self, documents: list[dict]):
        """
        添加文档到索引。

        参数:
            documents: [{"text": "内容", "metadata": {...}}, ...]
        """
        if not documents:
            return

        texts = [doc["text"] for doc in documents]
        embeddings = self.embedding_model.embed(texts)

        for doc, emb in zip(documents, embeddings):
            self.documents.append(
                {
                    "text": doc["text"],
                    "embedding": emb,
                    "metadata": doc.get("metadata", {}),
                }
            )

        print(f"已索引 {len(self.documents)} 个文档块")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        检索与查询最相关的 top_k 个文档。

        参数:
            query: 用户查询
            top_k: 返回数量

        返回:
            [{"text": "...", "score": 0.95, "metadata": {...}}, ...]
            按相似度降序排列
        """
        if not self.documents:
            return []

        # 1. 查询向量化
        query_embedding = self.embedding_model.embed([query])[0]

        # 2. 计算余弦相似度
        scored_docs = []
        query_vec = np.array(query_embedding)

        for doc in self.documents:
            doc_vec = np.array(doc["embedding"])
            similarity = self._cosine_similarity(query_vec, doc_vec)
            scored_docs.append(
                {
                    "text": doc["text"],
                    "score": float(similarity),
                    "metadata": doc["metadata"],
                }
            )

        # 3. 排序
        scored_docs.sort(key=lambda x: x["score"], reverse=True)

        return scored_docs[:top_k]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度。

        cos(a, b) = (a · b) / (||a|| * ||b||)

        返回值范围: [-1, 1]，越接近 1 越相似
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


# 模拟 Embedding 模型（实际使用时替换为 OpenAI 或 SentenceTransformer）
class MockEmbedding:
    """
    模拟 Embedding 模型（仅用于测试 DenseRetriever 的逻辑）。

    实际项目中替换为:
    - OpenAI: embeddings.create(model="text-embedding-3-small", input=texts)
    - SentenceTransformer: model.encode(texts)
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        生成随机向量（仅用于演示）。
        实际项目应调用真实的 embedding API。
        """
        np.random.seed(42)  # 确定性随机（演示用）
        return [np.random.randn(self.dim).tolist() for _ in texts]


if __name__ == "__main__":
    print("=== 稠密检索演示 ===\n")

    # 准备文档
    docs = [
        {"text": "Python是一种解释型、面向对象的编程语言", "metadata": {"source": "wiki"}},
        {"text": "JavaScript主要用于Web前端开发", "metadata": {"source": "wiki"}},
        {"text": "Python在数据科学领域广泛应用，如NumPy、Pandas", "metadata": {"source": "blog"}},
        {"text": "Web开发中常用的后端语言包括Python(Django)和Java(Spring)", "metadata": {"source": "blog"}},
        {"text": "机器学习的三大框架: PyTorch, TensorFlow, Scikit-learn", "metadata": {"source": "paper"}},
    ]

    # 初始化检索器
    retriever = DenseRetriever(MockEmbedding(dim=64))
    retriever.add_documents(docs)

    # 搜索
    results = retriever.search("Python数据分析", top_k=3)
    for i, r in enumerate(results):
        print(f"  {i + 1}. [score={r['score']:.3f}] {r['text'][:60]}...")
```

Cosine similarity 的计算看似简单，但有一个重要的实现细节：**embedding 必须做归一化**。如果 query embedding 和 doc embedding 来自不同的模型（或同一模型的不同版本），它们的模长可能不同。直接用点积而不除以模长，会导致模长较大的向量天然占据优势，而不是语义上更相关的向量。

### 4.2 稀疏检索（BM25）—— 关键词精确匹配

BM25 是经典的文本检索算法，基于词频（TF）和逆文档频率（IDF）。与稠密检索的"语义"路线不同，BM25 走的是"统计"路线——一个词如果在查询中出现，且在少数文档中频繁出现（说明它是这些文档的特征词），就给高分。

BM25 与稠密检索是互补关系，不是替代关系。BM25 擅长的场景：精确关键词匹配（产品编号、人名、地名、代码标识符）、专业术语搜索、不可替代词（如 "OAuth2.0" 不能被 embedding 模型"理解"其语义）。BM25 不擅长的场景：同义词搜索、口语化查询、多语言匹配。

```python
"""
稀疏检索 —— BM25 算法。

BM25 (Best Match 25) 是经典的文本检索算法，基于词频和逆文档频率。

与稠密检索的区别:
- 稠密: 语义匹配（"汽车" 和 "轿车" 可以匹配）
- 稀疏: 关键词匹配（只有包含 "汽车" 的文档才会被匹配）
- BM25 优势: 精确匹配、可解释、不需要 embedding 模型
- BM25 劣势: 无法处理同义词、口语化表达
"""
import math
from collections import Counter


class BM25Retriever:
    """
    BM25 检索器（纯 Python 实现）。

    参数:
        k1: 词频饱和度参数（默认 1.5）
        b: 文档长度归一化参数（默认 0.75）
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[dict] = []
        self._avgdl: float = 0.0          # 平均文档长度
        self._doc_freqs: dict = {}         # {term: 包含该词的文档数}
        self._doc_lengths: list[int] = []   # 每篇文档的长度
        self._term_freqs: list[dict] = []   # 每篇文档的词频

    def add_documents(self, documents: list[dict]):
        """
        添加文档并建立索引。

        参数:
            documents: [{"text": "...", "metadata": {...}}, ...]
        """
        self.documents = documents
        self._doc_lengths = []
        self._term_freqs = []
        self._doc_freqs = {}

        for doc in documents:
            tokens = self._tokenize(doc["text"])
            tf = Counter(tokens)
            self._term_freqs.append(tf)
            self._doc_lengths.append(len(tokens))

            # 更新文档频率
            for term in tf:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        self._avgdl = sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0
        print(f"BM25 索引完成: {len(self.documents)} 篇文档, 平均长度 {self._avgdl:.0f}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        BM25 检索。

        返回:
            [{"text": "...", "score": ..., "metadata": {...}}, ...]
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        N = len(self.documents)

        for i, doc in enumerate(self.documents):
            score = 0.0
            doc_len = self._doc_lengths[i]
            tf = self._term_freqs[i]

            for token in query_tokens:
                if token not in self._doc_freqs:
                    continue

                # IDF
                df = self._doc_freqs[token]
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))

                # TF (带饱和度)
                term_freq = tf.get(token, 0)
                tf_component = (term_freq * (self.k1 + 1)) / (
                    term_freq + self.k1 * (1 - self.b + self.b * doc_len / self._avgdl)
                )

                score += idf * tf_component

            scores.append(score)

        # 排序
        scored = [
            {
                "text": doc["text"],
                "score": score,
                "metadata": doc.get("metadata", {}),
            }
            for doc, score in zip(self.documents, scores)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)

        return scored[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """
        中文分词（简化版：按字分词）。

        实际项目建议使用 jieba 分词:
        import jieba
        return list(jieba.cut(text))
        """
        # 简化: 按字 + 双字组合分词
        chars = list(text.replace(" ", ""))
        tokens = chars.copy()  # 单字
        # 双字组合
        for i in range(len(chars) - 1):
            tokens.append(chars[i] + chars[i + 1])
        return tokens


if __name__ == "__main__":
    docs = [
        {"text": "Python机器学习入门教程", "metadata": {"category": "技术"}},
        {"text": "JavaScript前端开发指南", "metadata": {"category": "技术"}},
        {"text": "机器学习在医疗诊断中的应用", "metadata": {"category": "论文"}},
        {"text": "Python数据分析实战", "metadata": {"category": "书籍"}},
        {"text": "深度学习框架PyTorch使用指南", "metadata": {"category": "技术"}},
    ]

    bm25 = BM25Retriever()
    bm25.add_documents(docs)

    print("\n=== BM25 检索测试 ===\n")

    queries = ["Python", "机器学习", "深度学习框架"]
    for q in queries:
        print(f"查询: '{q}'")
        results = bm25.search(q, top_k=3)
        for i, r in enumerate(results):
            print(f"  {i + 1}. [score={r['score']:.3f}] {r['text']}")
        print()
```

### 4.3 混合检索（Hybrid Search）—— 两者之长

单一检索方式都有盲区。混合检索通过加权融合稠密检索和 BM25 的分数，取两者之长。核心公式：`final_score = alpha * dense_norm + (1-alpha) * sparse_norm`。

alpha 的选择反映了你对"语义匹配 vs 关键词匹配"的偏好。alpha=0.7 偏向语义（大多数通用场景），alpha=0.3 偏向关键词（代码搜索、编号搜索），alpha=0.5 平衡。

混合检索的实现中，关键步骤是**分数归一化**。稠密检索的 cosine similarity 范围是 [-1, 1]，BM25 的分数可能从 0 到几十甚至上百。两个不同量纲的分数不能直接加权——必须先做 min-max normalization 归一化到 [0, 1]。

```python
"""
混合检索 —— 稠密 + 稀疏加权融合。

原理:
1. 分别用稠密检索和 BM25 检索获取各自的结果
2. 将两者的分数归一化
3. 加权融合: final_score = alpha * dense_score + (1-alpha) * sparse_score

优势: 兼顾语义匹配和关键词精确匹配
"""
import numpy as np
from typing import List


class HybridRetriever:
    """
    混合检索器。

    结合稠密检索（语义匹配）和 BM25（关键词匹配）的优势。
    """

    def __init__(self, dense_retriever, bm25_retriever, alpha: float = 0.7):
        """
        参数:
            dense_retriever: 稠密检索器（需要 .search(query, top_k) 方法）
            bm25_retriever: BM25 检索器（需要 .search(query, top_k) 方法）
            alpha: 稠密权重的系数 (0.0 = 纯BM25, 1.0 = 纯稠密)
        """
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.alpha = alpha

    def search(self, query: str, top_k: int = 5, retrieval_k: int = 20) -> list[dict]:
        """
        混合检索。

        参数:
            query: 用户查询
            top_k: 最终返回数量
            retrieval_k: 两路初检数量（通常较大，allowing more candidates for fusion）

        返回:
            融合排序后的结果
        """
        # 1. 两路检索（初检数可以比最终返回数大）
        dense_results = self.dense.search(query, top_k=retrieval_k)
        bm25_results = self.bm25.search(query, top_k=retrieval_k)

        # 2. 分数归一化 (min-max normalization)
        dense_results = self._normalize_scores(dense_results, "dense_score")
        bm25_results = self._normalize_scores(bm25_results, "bm25_score")

        # 3. 构建 text → 分数的映射
        fused: dict[str, dict] = {}

        for r in dense_results:
            key = r["text"]  # 用文本作为 key（生产环境应用 doc_id）
            fused[key] = {
                "text": r["text"],
                "metadata": r.get("metadata", {}),
                "dense_score": r["dense_score"],
                "bm25_score": 0.0,
            }

        for r in bm25_results:
            key = r["text"]
            if key in fused:
                fused[key]["bm25_score"] = r["bm25_score"]
            else:
                fused[key] = {
                    "text": r["text"],
                    "metadata": r.get("metadata", {}),
                    "dense_score": 0.0,
                    "bm25_score": r["bm25_score"],
                }

        # 4. 加权融合
        for item in fused.values():
            item["score"] = (
                self.alpha * item["dense_score"] + (1 - self.alpha) * item["bm25_score"]
            )

        # 5. 排序
        sorted_results = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

        return sorted_results[:top_k]

    def _normalize_scores(self, results: list[dict], score_key: str) -> list[dict]:
        """Min-Max 归一化分数到 [0, 1]"""
        if not results:
            return results

        scores = [r["score"] for r in results]
        min_s = min(scores)
        max_s = max(scores)

        for r in results:
            if max_s > min_s:
                r[score_key] = (r["score"] - min_s) / (max_s - min_s)
            else:
                r[score_key] = 1.0

        return results


if __name__ == "__main__":
    print("=== 混合检索原理 ===\n")
    print("final_score = α * dense_norm + (1-α) * sparse_norm")
    print()
    print("α 的选择:")
    print("  α = 0.7: 偏语义匹配（大多数场景推荐）")
    print("  α = 0.5: 平衡")
    print("  α = 0.3: 偏关键词匹配（如代码搜索）")
```

### 4.4 ReRank 重排序 —— 粗检+精排两阶段

粗检（Embedding/BM25/Hybrid）的目标是**召回**——从海量 chunk 中快速缩小范围到 top-M（如 top-20）。精排（Cross-Encoder）的目标是**精度**——对这 20 个候选做更准确的相关性判断，筛选出最终 top-K（如 top-5）。

为什么需要两阶段？因为 Cross-Encoder 虽然精度高（query 和 doc 联合编码，比各自独立编码更准确），但速度慢——它需要对每个 (query, doc) 对做一次前向传播。如果直接从 10000 个 chunk 中用 Cross-Encoder 打分，延迟将是秒级甚至十秒级，不可接受。两阶段方案中，粗检在毫秒级将范围缩小到 20，Cross-Encoder 只对 20 个候选打分，总延迟可控。

```python
"""
ReRank 重排序 —— 用 Cross-Encoder 对初检结果精排。

原理:
1. 粗检（Bi-Encoder/Embedding）: 快速从海量文档中召回 top-k 候选
2. 精排（Cross-Encoder）: 对 query-候选对 做更精确的相关性判断

为什么需要 ReRank?
- Bi-Encoder 速度快但精度有限（query 和 doc 独立编码）
- Cross-Encoder 精度高但速度慢（query 和 doc 联合编码）
- ReRank 兼顾两者: 粗检快速缩小范围，精排只在候选集上运行
"""


class SimpleCrossEncoder:
    """
    简化的 Cross-Encoder 实现（演示原理）。

    实际项目使用:
    - HuggingFace: model = CrossEncoder("BAAI/bge-reranker-large")
    - Cohere: co.rerank(query=query, documents=docs)
    - Voyage: voyage.rerank(query=query, documents=docs)
    """

    def __init__(self):
        # 实际项目中: from sentence_transformers import CrossEncoder
        # self.model = CrossEncoder("BAAI/bge-reranker-base")
        pass

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """
        对候选文档重排序。

        参数:
            query: 用户查询
            candidates: 粗检返回的候选文档列表
            top_k: 返回数量

        返回:
            重排序后的文档列表（带新的 rerank_score）

        实际实现:
            pairs = [[query, doc["text"]] for doc in candidates]
            scores = self.model.predict(pairs)
            # 按 rerank_score 排序
        """
        # 这是演示代码，实际项目用上面的 HuggingFace CrossEncoder
        print(f"[ReRank] 对 {len(candidates)} 个候选重排序，取 top-{top_k}")
        # 模拟: 返回原顺序（实际会重新排序）
        for doc in candidates:
            doc["rerank_score"] = doc.get("score", 0.0)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]


class RerankPipeline:
    """
    完整的检索 + 重排序管道。

    流程:
    1. 粗检（Hybrid/Dense/BM25）: 召回 top-M 候选 (M > K)
    2. ReRank（Cross-Encoder）: 精排取 top-K (K <= M)
    3. 返回最终结果
    """

    def __init__(self, retriever, reranker, retrieval_k: int = 20, final_k: int = 5):
        self.retriever = retriever
        self.reranker = reranker
        self.retrieval_k = retrieval_k  # 粗检索取多少
        self.final_k = final_k          # 最终返回多少

    def search(self, query: str) -> list[dict]:
        """检索 + 重排序"""
        # 第1阶段: 粗检索
        print(f"[Pipeline] 第1阶段: 粗检索 (top-{self.retrieval_k})")
        candidates = self.retriever.search(query, top_k=self.retrieval_k)

        # 第2阶段: 精排
        print(f"[Pipeline] 第2阶段: ReRank (top-{self.final_k})")
        final_results = self.reranker.rerank(query, candidates, top_k=self.final_k)

        return final_results


if __name__ == "__main__":
    print("=== ReRank 管道 ===\n")
    print("阶段1 (粗检): 从 N 个文档中快速召回 top-20 ~ top-100")
    print("阶段2 (精排): Cross-Encoder 对召回文档精确打分，取 top-3 ~ top-5")
    print()
    print("推荐 ReRank 模型:")
    print("  - BAAI/bge-reranker-v2-m3:  中文友好，效果最好")
    print("  - BAAI/bge-reranker-base:    轻量，延迟低")
    print("  - Cohere Rerank API:         商业方案，效果稳定")
```

---

## 5. 检索后处理

检索返回的文档不一定都是高质量的，需要后处理来提升最终输入 LLM 的上下文质量。三个核心步骤：

**相似度阈值过滤**：如果检索到的文档相似度都很低（如最高分才 0.3），说明知识库中可能没有相关信息。此时硬把低分文档塞给 LLM 反而可能导致"依据不相关内容编造答案"。应该设置阈值（如 0.6），低于阈值的直接过滤，触发 fallback。

**去重**：检索到的多个 chunk 可能来自同一文档的相邻位置，内容高度重叠。如果不去重，LLM 看到的上下文中有大量重复信息，既浪费 token 又可能干扰模型判断。Jaccard 相似度是简单有效的去重指标。

**上下文窗口分配**：LLM 的上下文窗口是有限的。当检索到很多相关 chunk 时，需要在窗口中合理分配空间——按分数从高到低依次放入，直到窗口满。长尾的低分 chunk 可能被截断或丢弃。

```python
"""
检索后处理 —— 过滤、去重、上下文分配。

检索返回的文档不一定都是高质量的，需要后处理。
"""


class PostRetrievalProcessor:
    """
    检索后处理器。

    三个核心步骤:
    1. 相似度阈值过滤: 去掉相关性太低的文档
    2. 去重: 去掉内容高度重复的文档
    3. 上下文窗口分配: 在有限窗口内分配文档
    """

    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold

    def filter_by_threshold(self, results: list[dict]) -> list[dict]:
        """
        过滤掉相似度低于阈值的文档。

        如果所有文档都低于阈值 → 触发 fallback
        """
        filtered = [r for r in results if r["score"] >= self.similarity_threshold]

        if not filtered:
            print("[警告] 所有文档相似度都低于阈值，返回 top-1 作为 fallback")
            return results[:1] if results else []

        print(f"[过滤] {len(results)} → {len(filtered)} (阈值={self.similarity_threshold})")
        return filtered

    def deduplicate(self, results: list[dict], min_jaccard: float = 0.8) -> list[dict]:
        """
        基于 Jaccard 相似度去重。

        如果两个文档的文本高度重叠，只保留分数高的那个。
        """
        if len(results) <= 1:
            return results

        kept = []
        for i, doc in enumerate(results):
            is_duplicate = False
            doc_tokens = set(doc["text"])

            for kept_doc in kept:
                kept_tokens = set(kept_doc["text"])
                intersection = doc_tokens & kept_tokens
                union = doc_tokens | kept_tokens

                if union:
                    jaccard = len(intersection) / len(union)
                    if jaccard >= min_jaccard:
                        is_duplicate = True
                        break

            if not is_duplicate:
                kept.append(doc)

        if len(kept) < len(results):
            print(f"[去重] {len(results)} → {len(kept)} (Jaccard阈值={min_jaccard})")

        return kept

    def allocate_context_window(
        self, results: list[dict], max_tokens: int = 4000
    ) -> tuple[list[dict], int]:
        """
        在有限的上下文窗口内分配文档。

        策略: 按分数从高到低分配，每个文档约占总窗口的 1/top_k。
        遇到超长的文档可能会截断。

        参数:
            results: 排序后的文档列表
            max_tokens: 总 token 预算

        返回:
            (分配的文档列表, 使用的 token 数)
        """
        allocated = []
        tokens_used = 0
        # 粗略估计: 1 token ≈ 0.5 中文字符 ≈ 1 英文字符
        chars_per_token = 0.5

        for doc in results:
            estimated_tokens = len(doc["text"]) * chars_per_token
            remaining = max_tokens - tokens_used

            if remaining <= 0:
                break

            if estimated_tokens <= remaining:
                # 完整放入
                allocated.append(doc)
                tokens_used += int(estimated_tokens)
            else:
                # 截断放入
                max_chars = int(remaining / chars_per_token)
                truncated_doc = doc.copy()
                truncated_doc["text"] = doc["text"][:max_chars] + "...(truncated)"
                allocated.append(truncated_doc)
                tokens_used += remaining
                break

        print(f"[窗口分配] {len(results)} → {len(allocated)} 个文档, 使用约 {tokens_used} tokens")
        return allocated, tokens_used

    def process(self, results: list[dict], max_tokens: int = 4000) -> list[dict]:
        """完整的后处理流水线"""
        # 步骤1: 阈值过滤
        results = self.filter_by_threshold(results)

        # 步骤2: 去重
        results = self.deduplicate(results)

        # 步骤3: 窗口分配
        results, _ = self.allocate_context_window(results, max_tokens)

        return results


if __name__ == "__main__":
    print("=== 检索后处理流水线 ===\n")
    processor = PostRetrievalProcessor(similarity_threshold=0.6)

    print("后处理步骤:")
    print("  1. 相似度阈值过滤 → 去掉不相关的")
    print("  2. Jaccard 去重 → 去掉重复内容")
    print("  3. 上下文窗口分配 → 控制 token 预算")
```

---

## 6. 生成阶段

### 6.1 RAG 生成 Prompt 模板 —— 从检索结果到回答

RAG 的生成 prompt 设计和普通对话有显著不同。普通对话中，模型可以自由使用训练知识；RAG 中，你**只希望**模型使用检索到的资料，而非它自己的"记忆"。如果 prompt 设计不当，模型会混合使用检索结果和训练知识，产生"假装来自资料但实际是自己编的"幻觉。

高质量的 RAG prompt 包含四个要素：(1) System Prompt 中明确角色设定和严格的"只用资料"约束；(2) 检索到的上下文标注来源（让模型可以引用）；(3) 用户问题；(4) fallback 策略——当资料不足时怎么办。

```python
"""
RAG 的生成阶段 —— 将检索结果与用户问题组合，生成最终回答。

高质量的 RAG prompt 包含:
1. System Prompt: 角色定义 + 回答规则
2. 检索到的上下文（标注来源）
3. 用户问题
4. 引用要求
"""
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


RAG_SYSTEM_PROMPT = """\
你是一个基于知识库的问答助手。你的回答必须基于提供的文档内容。

# 回答规则
1. 只使用【参考资料】中提供的信息回答问题
2. 如果参考资料不足以回答问题，明确告知用户"根据现有资料无法回答"
3. 引用参考资料时，标出来源（如：根据[文档1]第X页...）
4. 不要编造参考资料中没有的信息
5. 如果资料中有矛盾，指出矛盾并标注各自的来源

# 禁止
- 不要使用你的训练数据中的知识（除非资料中也有）
- 不要说"根据我的知识"或"据我所知"
- 不要忽略资料中相互矛盾的部分"""


class RAGGenerator:
    """
    RAG 生成器 —— 基于检索结果生成回答。
    """

    def __init__(
        self,
        system_prompt: str = RAG_SYSTEM_PROMPT,
        model: str = "gpt-4o",
        include_sources: bool = True,
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.include_sources = include_sources

    def build_prompt(
        self,
        question: str,
        retrieved_docs: list[dict],
    ) -> list[dict]:
        """
        构建 RAG 的 messages。

        结构:
        1. System Prompt（回答规则）
        2. User Message（检索结果 + 用户问题）
        """
        # 构建参考资料部分
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            source_info = ""
            meta = doc.get("metadata", {})

            if meta.get("source"):
                source_info += f"来源: {meta['source']}"
            if meta.get("page_number"):
                source_info += f", 第{meta['page_number']}页"
            if meta.get("title"):
                source_info += f", 标题: {meta['title']}"

            context_parts.append(
                f"--- 文档{i + 1} ({source_info}) ---\n"
                f"相关性分数: {doc.get('score', 'N/A')}\n"
                f"{doc['text']}"
            )

        context_text = "\n\n".join(context_parts)

        user_message = f"""\
## 参考资料

{context_text}

## 用户问题

{question}

## 要求
- 基于以上参考资料回答
- 标注引用来源（如 [文档1]）
- 如果资料不足，请明确说明"""

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

    def generate(self, question: str, retrieved_docs: list[dict]) -> dict:
        """
        生成 RAG 回答。

        返回:
            {
                "answer": "回答文本",
                "sources": ["来源1", "来源2"],
                "has_sufficient_info": True/False,
            }
        """
        if not retrieved_docs:
            return {
                "answer": "抱歉，没有找到相关信息来回答您的问题。",
                "sources": [],
                "has_sufficient_info": False,
            }

        messages = self.build_prompt(question, retrieved_docs)

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
        )

        answer = response.choices[0].message.content

        # 提取来源（简化版）
        sources = []
        for i, doc in enumerate(retrieved_docs):
            meta = doc.get("metadata", {})
            source_str = meta.get("source", f"文档{i + 1}")
            if meta.get("page_number"):
                source_str += f" (第{meta['page_number']}页)"
            sources.append(source_str)

        # 判断资料是否足够（检查回答中是否有 fallback 关键词）
        fallback_phrases = ["无法回答", "资料不足", "没有相关信息", "根据现有资料无法"]
        has_sufficient = not any(phrase in answer for phrase in fallback_phrases)

        return {
            "answer": answer,
            "sources": sources,
            "has_sufficient_info": has_sufficient,
        }

    def generate_with_fallback(self, question: str, retrieved_docs: list[dict]) -> dict:
        """
        带 fallback 机制的生成。

        当检索不到相关资料时:
        - 方案A: 告知用户资料不足
        - 方案B: 建议用户换一种问法
        - 方案C: 提供近似匹配的结果
        """
        result = self.generate(question, retrieved_docs)

        if not result["has_sufficient_info"]:
            # Fallback: 建议用户改述问题
            result["fallback_suggestion"] = (
                "建议您尝试: \n"
                "1. 使用更具体的关键词\n"
                "2. 缩短问题长度\n"
                "3. 使用文档中可能出现的术语"
            )

        return result


if __name__ == "__main__":
    # 模拟检索结果
    mock_docs = [
        {
            "text": "员工每年享有5天带薪病假。病假需要提供医院证明。超过5天需要HR审批。",
            "score": 0.92,
            "metadata": {"source": "员工手册.pdf", "page_number": 15},
        },
        {
            "text": "年假计算: 入职满1年的员工享有5天年假，满3年10天，满5年15天。",
            "score": 0.85,
            "metadata": {"source": "员工手册.pdf", "page_number": 12},
        },
    ]

    generator = RAGGenerator()

    print("=== RAG 生成 Prompt 预览 ===\n")
    messages = generator.build_prompt("我的病假有多少天？", mock_docs)
    for msg in messages:
        print(f"[{msg['role']}]: {msg['content'][:300]}...")
        print()

    print("(取消注释以下代码以实际生成)")
    # result = generator.generate("我的病假有多少天？", mock_docs)
    # print(f"回答: {result['answer']}")
    # print(f"来源: {result['sources']}")
```

---

## 基础练习

### 练习 1: 文档加载器
**场景**: 实现一个通用文档加载器，支持 PDF、Word、TXT 三种格式。
**要求**: 自动识别扩展名，提取文本和元数据，返回统一格式。
**文件**: `exercise/ai-application/ch04_rag/ex1_document_loader.py`

### 练习 2: 文本切块比较
**场景**: 对比 CharacterTextSplitter 和 RecursiveCharacterTextSplitter 的效果。
**要求**: 用同一文本测试，对比 chunk 数量、大小分布、语义完整性。
**文件**: `exercise/ai-application/ch04_rag/ex2_text_splitter.py`

### 练习 3: BM25 检索器
**场景**: 从头实现 BM25 检索器（不依赖现成库）。
**要求**: 支持中文分词（jieba），实现 TF-IDF 和 BM25 两种打分。
**文件**: `exercise/ai-application/ch04_rag/ex3_bm25.py`

## 进阶练习

### 练习 4: 端到端文档问答
**场景**: 构建一个完整的文档问答系统（加载→切块→索引→检索→生成）。
**要求**: 支持上传 PDF 文件，回答关于文档内容的问题。
**文件**: `exercise/ai-application/ch04_rag/ex4_rag_pipeline.py`

### 练习 5: 多策略对比评估
**场景**: 设计评估实验，对比不同检索策略在相同测试集上的效果。
**要求**: Dense vs BM25 vs Hybrid，记录 MRR 和 Hit Rate。
**文件**: `exercise/ai-application/ch04_rag/ex5_retrieval_eval.py`

---

## 常见错误

### 错误 1: chunk_size 太小导致信息碎片化

```python
# 错误: 100字的chunk可能只包含半句话
splitter = CharacterTextSplitter(chunk_size=100)
# 检索时匹配到 "Python是一种..." 但不知道后面在说什么

# 修正: 根据文档类型选择合适的 chunk_size
splitter = CharacterTextSplitter(chunk_size=800)  # 技术文档 800-1200
```

### 错误 2: 检索时只用稠密不用稀疏，导致精确匹配失败

```python
# 错误: 用户搜 "AB-12345"，稠密检索返回了一堆不相关的文档
# 因为 embedding 模型不理解产品编号

# 修正: 混合检索确保精确关键词也能匹配
hybrid = HybridRetriever(dense, bm25, alpha=0.5)
```

### 错误 3: RAG 回答时模型忽略检索结果

```python
# 错误: System prompt 没有强调"只基于参考资料"
# 模型可能结合自身知识和检索结果输出，产生幻觉
system = "你是一个助手。回答用户问题。"
# → 模型可能用自己的知识"补充"检索结果中没有的信息

# 修正: 明确要求只使用参考资料
system = """只使用参考资料中的信息回答。如果资料中没有，说'无法回答'。
禁止使用任何外部知识。"""
```

### 错误 4: 忘记对 embedding 做归一化

```python
# 错误: 直接做点积作为相似度
similarity = np.dot(query_vec, doc_vec)  # 长度不同的向量不可比

# 修正: 用余弦相似度（内积/模长）
cos_sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
```

### 错误 5: 生成 prompt 太长导致上下文溢出

```python
# 错误: 所有检索结果一股脑塞进 prompt
# 结果: prompt tokens > 模型上下文窗口 → 截断 → 丢失信息

# 修正: 使用 PostRetrievalProcessor 做上下文窗口分配
processor = PostRetrievalProcessor()
docs = processor.process(retrieval_results, max_tokens=4000)
```

### 错误 6: 切块时丢失页码/章节等元数据

```python
# 错误: 只保存文本内容，不保留来源信息
chunks = [chunk_text for chunk_text in splitter.split(text)]
# 生成回答时无法标注引用来源

# 修正: 每个 chunk 保留元数据
chunks = [{"text": t, "metadata": {"page": i, "source": file}} for ...]
```

---

## 本章小结

本章深入学习了 RAG 架构的完整体系。回顾关键理解：

- **RAG 的本质与两阶段架构**：RAG 给 LLM 装上了"外挂硬盘"——回答前先查资料，用查到的内容作为依据。架构分为离线索引阶段（加载 -> 切块 -> Embedding -> 存储）和在线查询阶段（Query Embedding -> 检索 -> ReRank -> 生成）。离线是"空间换时间"，在线是毫秒级响应。两个阶段完全解耦，各自的性能目标和技术选型独立。

- **文档加载的场景选择**：PDF 主力用 PyMuPDF（速度），表格用 pdfplumber（精度），多格式用 unstructured（通用性）。保留元数据（页码、来源、标题）对后续的引用标注至关重要——没有元数据的 RAG 是无法追溯的。

- **切块策略的核心权衡**：chunk_size 的选值是在"检索精度"和"上下文完整性"之间做 trade-off。技术文档 800-1200，法律合同 500-800，代码仓库 1500-3000 且 overlap=0。递归切块通过在分隔符优先级层级间降级，最大程度保留了语义边界。chunk_overlap 防止关键信息恰好落在边界上。

- **检索策略的组合艺术**：稠密检索（语义匹配）和 BM25（关键词精确匹配）是互补关系，不是替代关系。混合检索通过分数归一化+加权融合取两者之长。ReRank（Cross-Encoder）实现"粗检+精排"两阶段——粗检快速缩小范围（毫秒级），精排精确打分（秒级）。整个检索管道是"召回率+精度"的工程平衡。

- **生成阶段的规范设计**：RAG prompt 的核心约束是"只用资料，不用训练知识"——这需要严格的 system prompt 措辞（包含 fallback 策略和来源引用要求）和结构化的上下文组装。来源标注不仅是引用规范，更是可追溯性的保障——用户可以根据引用追溯到原始文档的特定位置。

- **后处理的必要性**：阈值过滤防止低质量内容污染 LLM 输入，Jaccard 去重避免信息冗余浪费 token，上下文窗口分配确保有限的窗口预算用在最相关的内容上。

下一章将深入学习向量数据库实战，将本章的检索部分用专业的向量数据库（ChromaDB、Milvus）来实现。
