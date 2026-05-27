# 第07章 LangChain 与 LlamaIndex

## 学习目标

1. 理解 LangChain 的核心架构设计：Model I/O、Retrieval、Chains、Agents 四大模块
2. 掌握 LCEL（LangChain Expression Language）的管道语法与 Runnable 接口
3. 能独立构建 RAG（检索增强生成）应用和工具调用 Agent
4. 理解 LlamaIndex 的数据索引与查询引擎架构
5. 能对本地文档集建立索引并使用多种查询模式检索
6. 掌握 LangChain vs LlamaIndex 的选型决策

---

## 前置知识

- Python 异步编程基础（`async/await`）：本章部分代码使用异步
- OpenAI API：Chat Completion、Embedding、Function Calling
- 向量数据库基础概念：embedding、相似度检索、Chroma（见第05章）
- Python 类型注解（`typing` 模块）：提高代码可读性

---

## 7.0 前言：为什么需要框架——从"手动挡"到"自动挡"

在深入学习 LangChain 和 LlamaIndex 之前，我们先回答一个根本性的问题：**为什么需要框架？直接用 OpenAI SDK 不行吗？**

答案是可以，但代价很大。想象你需要构建一个这样的应用：用户提问 → 从知识库中搜索相关文档 → 将文档作为参考资料注入 Prompt → LLM 生成回答 → 如果问题涉及计算，调用计算器工具 → 如果问题需要查天气，调用天气 API → 最后汇总所有结果返回给用户。用纯 OpenAI SDK 写，你需要手动管理对话历史、手动拼接 Prompt、手动解析 LLM 的"工具调用请求"、手动执行工具、手动把工具结果注回对话、手动处理每一个环节的错误……代码很快就会变得难以维护。

LangChain 和 LlamaIndex 解决的就是这个"编排"问题。它们提供了一套标准化的组件（积木）和连接方式（榫卯），让你能像搭乐高一样构建 LLM 应用。但这两个框架的设计哲学完全不同，理解它们的差异是本章的核心。

**LangChain 的设计哲学是"通用编排层"**：它把自己定位为 LLM 应用的操作系统——不关心你的数据是什么、你的任务是什么，只提供标准化的接口让你把模型、数据、工具、记忆这些组件串联起来。LangChain 的核心抽象是 Chain（链）：一个组件输出到下一个组件输入的管道。这种设计让它极其灵活，但也让它显得"什么都能做，但什么都不精"。

**LlamaIndex 的设计哲学是"数据优先的索引层"**：它从数据出发，核心问题始终是"我有一个数据集合，如何让 LLM 高效地从中提取信息？"LlamaIndex 的核心抽象是 Index（索引）：数据被结构化地组织成各种索引形式（向量索引、树索引、关键词索引等），然后通过查询引擎来检索。这种聚焦让它在这一个领域做得比 LangChain 更深，但也意味着它不擅长 Agent、工具调用等"非检索"任务。

**一个关键的决策：什么时候用框架，什么时候直接用 SDK？** 如果你的应用只有"用户提问 → LLM 回答"这一条简单的路径，框架反而增加了学习成本——直接用 OpenAI SDK 写几十行代码就解决了。当你面临以下任一情况时，才值得引入框架：（1）需要管理对话历史和记忆；（2）需要从多个数据源检索；（3）需要 LLM 自主调用工具（Agent）；（4）需要复杂的多步推理链；（5）需要流式输出的中间处理。简而言之：框架的价值在于管理复杂度，而不是增加复杂度。

带着这个理解，我们来深入学习这两个框架。

---

## 7.1 LangChain 核心架构

### 7.1.1 LangChain 的四大支柱

LangChain 将 LLM 应用涉及的所有环节抽象为四个核心模块。这四个模块的关系不是平行的，而是有一条清晰的数据流向：

**Model I/O** 是入口和出口：Prompt Template 格式化输入 → Chat Model 调用 LLM → Output Parser 解析输出。这是最基础的模块，任何 LangChain 应用都绕不开它。

**Retrieval** 是外部知识的入口：Document Loader 加载文档 → Text Splitter 切分文本 → Embedding 向量化 → Vector Store 存储向量 → Retriever 检索。这个五层架构对应了 RAG 的完整流程。

**Chains** 是编排的核心：把多个步骤串联起来。LangChain 旧版使用 LLMChain 和 SequentialChain，新版推荐使用 LCEL（管道语法）。Chains 的价值在于把分散的组件组合成一条清晰的流水线。

**Agents** 是智能的顶点：让 LLM 自主决定"用哪个工具、以什么顺序、怎么判断结果是否足够"。Agent 本质上是一个在 Thought→Action→Observation 之间循环的推理引擎。

这四个模块的组织关系如下：

```
+-----------------------------------------------------+
|                    LangChain                        |
+---------------+---------------+---------------------+
|  Model I/O    |  Retrieval    |  Chains & Agents    |
|  ---------    |  ---------    |  ---------------    |
|  - Prompt     |  - Loader     |  - LLMChain         |
|  - Chat Model |  - Splitter   |  - SequentialChain  |
|  - Output     |  - Embedding  |  - RouterChain      |
|    Parser     |  - VectorStore|  - AgentExecutor    |
|               |  - Retriever  |  - Tools            |
+---------------+---------------+---------------------+
|                   LCEL (管道语法)                    |
+-----------------------------------------------------+
```

### 7.1.2 安装与第一个 LangChain 程序

```bash
pip install langchain langchain-openai langchain-community
pip install chromadb  # 向量数据库（第05章已安装可跳过）
pip install tiktoken  # token 计数
```

LangChain 对 OpenAI SDK 的封装遵循一个核心思路：**把所有东西都变成 Runnable 对象**。一个 Runnable 对象有一个统一的 `invoke()` 方法，输入数据，返回结果。ChatOpenAI 是最基础也是最重要的 Runnable——它封装了 Chat Completion API，但与之交互的不是原始的 dict，而是 LangChain 的消息对象（SystemMessage、HumanMessage、AIMessage）。

```python
"""
ex_7_1_hello_langchain.py: LangChain 入门——最简单的 LLM 调用
演示：LangChain 如何封装 OpenAI 的 Chat Completion
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ====== 方式一：使用 LangChain 的 ChatOpenAI ======
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 初始化模型
# 对比：原来的 openai.OpenAI() 变成了 ChatOpenAI()
llm = ChatOpenAI(
    model="gpt-4o",                      # 模型名
    temperature=0.7,                     # 温度（0-2），越高越随机
    max_tokens=500,                      # 最大输出 token
    api_key=os.getenv("OPENAI_API_KEY"),  # 可省略（自动读环境变量）
    # base_url=os.getenv("OPENAI_BASE_URL"),  # 如需代理
)

# 调用模型
# LangChain 使用 Message 对象，而不是 dict
messages = [
    SystemMessage(content="你是一位中国古典文学教授，回答要引经据典。"),
    HumanMessage(content="请解释'道可道，非常道'的含义。"),
]

response = llm.invoke(messages)  # invoke() 是 LangChain 的统一调用接口
print(response.content)  # .content 获取文本内容
# 输出类型是 AIMessage，它包含更多元数据:
# print(response.response_metadata) # 含 token 用量、模型名等
# print(response.usage_metadata)     # input_tokens, output_tokens
```

**关键概念**：

- `ChatOpenAI` 是 LangChain 对 OpenAI Chat Completion API 的封装，它返回 `AIMessage` 而非字符串。
- `invoke()` 是 LangChain 的统一调用接口（所有 Runnable 对象都有这个方法）。
- `SystemMessage`、`HumanMessage`、`AIMessage` 是 LangChain 的三种消息类型（对应 API 的 system/user/assistant）。

### 7.1.3 Model I/O 详解 —— 从输入到输出的标准化管道

Model I/O 处理三个环节的连接：**Prompt Template** 负责把变量填入模板生成最终 Prompt → **Chat Model** 负责调用 LLM → **Output Parser** 负责把 LLM 的文本输出解析为结构化的 Python 对象。

#### Prompt Template——让 Prompt 可编程

手工拼接 Prompt 字符串（`f"翻译为{lang}: {text}"`）在小项目里能用，但当你有几十种不同的 Prompt 模板、每个模板有不同的变量时，就会变得混乱。LangChain 的 Prompt Template 提供了两个关键能力：**变量管理**（自动校验变量的完整性）和**消息角色管理**（区分 system/human/ai 消息）。Few-Shot Prompt Template 更进一步——它让你用几个"输入→期望输出"的示例来教会模型输出的格式，而不需要在 Prompt 中写大段的格式说明文字。

```python
"""
ex_7_2_prompt_template.py: LangChain Prompt Template 详解
演示：ChatPromptTemplate、Few-Shot、自定义模板
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ====== 1. 基础 ChatPromptTemplate ======
# 使用 from_messages 构建消息模板
# 花括号 { } 中的变量会在调用时被填入
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位{role}。请用{language}回答问题。"),
    ("human", "{question}"),
])

# 填入变量，生成最终的消息列表
messages = prompt.format_messages(
    role="Python 编程导师",
    language="中文",
    question="解释什么是闭包？",
)
print("=== 填入变量后的消息 ===")
for msg in messages:
    print(f"[{msg.type}] {msg.content}")
# 输出:
# [system] 你是一位Python 编程导师。请用中文回答问题。
# [human] 解释什么是闭包？

# 直接调用
response = llm.invoke(messages)
print(f"\n回复: {response.content[:100]}...")

# 也可以用管道语法一步完成（详见 7.2 LCEL）:
# chain = prompt | llm
# response = chain.invoke({"role": "...", "language": "...", "question": "..."})


# ====== 2. 含对话历史的模板 ======
# MessagesPlaceholder 可以插入一个消息列表（用于对话历史）
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是一位有用的助手。"),
    MessagesPlaceholder(variable_name="history"),  # 这里插入历史消息列表
    ("human", "{input}"),
])

# 使用示例
history = [
    HumanMessage(content="我叫张伟。"),
    AIMessage(content="你好张伟！有什么可以帮你的？"),
    HumanMessage(content="我家有三口人。"),
    AIMessage(content="好的，我记住了。"),
]
messages = prompt_with_history.format_messages(
    history=history,
    input="我叫什么名字？我家有几口人？",
)
# 现在 messages 包含了 system + 历史4条 + 当前输入


# ====== 3. Few-Shot 模板（少样本提示） ======
# 给模型几个"问题to答案"的范例，引导输出格式
examples = [
    {"input": "晴天 to sunny", "output": '{"chinese": "晴天", "english": "sunny"}'},
    {"input": "下雨 to rain", "output": '{"chinese": "下雨", "english": "rain"}'},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    input_variables=["input"],  # 每个范例中要替换的变量
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "将天气词翻译为 JSON 格式。"),
    few_shot_prompt,
    ("human", "{input}"),
])

print("\n=== Few-Shot 生成的 Prompt ===")
print(final_prompt.format(input="下雪 to snow"))
```

#### Output Parser —— 让 LLM 输出可被代码消费

LLM 输出的是文本，但你的代码需要的是结构化数据。Output Parser 的职责就是在这两者之间架起桥梁。三种 Parser 对应三种不同的场景：`StrOutputParser` 适用于不需要结构的输出（如聊天回复）；`JsonOutputParser` 适用于需要 dict 但不需要强类型校验的场景；`PydanticOutputParser` 是推荐的生产级方案——它利用 Pydantic 的 schema 自动生成 JSON 格式说明、自动校验字段类型、提供代码补全。

PydanticOutputParser 的工作流程很巧妙：它调用 `get_format_instructions()` 生成一段文字说明（"请输出一个 JSON 对象，包含以下字段：title(string), rating(float)...")，这段说明会被注入到 Prompt 中。LLM 看到这段说明后按要求输出 JSON，Parser 再把 JSON 转为 Pydantic 对象。如果 LLM 输出的 JSON 不合法或字段缺失，Pydantic 的校验机制会直接抛出明确的错误。

```python
"""
ex_7_3_output_parser.py: LangChain Output Parser 详解
演示：将 LLM 原始文本自动解析为 Python 数据结构
"""
import json
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,        # 最简单的：直接返回字符串
    JsonOutputParser,       # 解析 JSON to dict
    PydanticOutputParser,   # 解析 JSON to Pydantic 对象（推荐）
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ====== 1. StrOutputParser: 直接取文本 ======
# 什么都不做，就是提取 .content
parser_str = StrOutputParser()
result = parser_str.parse("这是模型回复的文本")
print(f"StrOutputParser: {result}")


# ====== 2. JsonOutputParser: 解析 JSON ======
# 适用场景: 需要结构化输出但不需要强类型校验
parser_json = JsonOutputParser()
text = '{"name": "张三", "age": 25, "skills": ["Python", "Java"]}'
result = parser_json.parse(text)
print(f"\nJsonOutputParser: {result}")
print(f"类型: {type(result)}, name={result['name']}")


# ====== 3. PydanticOutputParser: 解析为 Pydantic 对象（推荐） ======
# 这可以让你在代码中获得完整的类型提示和自动校验

# 第一步: 定义输出结构
class MovieReview(BaseModel):
    """电影评论的结构化输出"""
    title: str = Field(description="电影名称")
    rating: float = Field(description="评分，0-10分")
    summary: str = Field(description="简短评价，50字以内")
    pros: List[str] = Field(description="优点列表，至少2条")
    cons: List[str] = Field(description="缺点列表")
    recommend: bool = Field(description="是否推荐")

# 第二步: 创建解析器
parser = PydanticOutputParser(pydantic_object=MovieReview)

# 第三步: 把格式说明注入 Prompt
# parser.get_format_instructions() 会自动生成 JSON schema 的描述
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业影评人。\n{format_instructions}"),
    ("human", "请评价电影《{movie_name}》"),
])

# 第四步: 构建并执行链
chain = prompt | llm | parser

# 调用
review = chain.invoke({
    "movie_name": "肖申克的救赎",
    "format_instructions": parser.get_format_instructions(),
})

print(f"\n=== 结构化影评 ===")
print(f"电影: {review.title}")
print(f"评分: {review.rating}/10")
print(f"评价: {review.summary}")
print(f"优点: {', '.join(review.pros)}")
print(f"缺点: {', '.join(review.cons)}")
print(f"推荐: {'是' if review.recommend else '否'}")
print(f"\n类型: {type(review)}   -- 是 MovieReview 对象，不是 dict!")

# 你可以利用 Pydantic 的校验和序列化
print(f"\nJSON 序列化: {review.model_dump_json(indent=2)}")
```

### 7.1.4 Retrieval 模块 —— 知识检索的五层架构

RAG 的核心是检索。LangChain 将检索过程解耦为五个独立的组件，每个组件可以在不改变其他组件的情况下替换。理解这五层的职责和衔接关系，是掌握 LangChain 的关键。

**第一层 Document Loader**：从各种来源（txt、pdf、csv、网页、数据库）加载原始文档。不同来源有不同的解析器，但都输出统一的 Document 对象（包含 page_content 和 metadata）。

**第二层 Text Splitter**：将长文档切分为合适大小的 chunk。这是整个 RAG 系统中最被低估的一环——切分质量直接影响检索准确率。切太大：一个 chunk 里信息太杂，LLM 难以提取关键信息。切太小：信息碎片化，LLM 看不到完整上下文。RecursiveCharacterTextSplitter 是目前最推荐的通用方案，它会优先在自然段落边界处切分，尽量避免"拦腰截断"一个句子。

**第三层 Embedding**：用 embedding 模型将文本转为向量。LangChain 内置了 OpenAI、HuggingFace、Cohere 等多种 embedding 模型的适配器。

**第四层 Vector Store**：将向量和原始文本一起存储。LangChain 支持 Chroma、Milvus、Pinecone、Weaviate、FAISS 等主流向量数据库。

**第五层 Retriever**：提供统一的检索接口。这是 LangChain 设计中很重要的一环——通过 Retriever 接口屏蔽了底层向量数据库的差异，你的 RAG 链不需要知道用的是 Chroma 还是 Milvus。

```
Document Loader → Text Splitter → Embedding → Vector Store → Retriever
     |                |              |             |              |
  加载文档        切分文本      向量化文本    存储向量       检索接口
```

下面分别实现每一层。

#### Document Loader

```python
"""
ex_7_4_document_loader.py: LangChain Document Loader
演示：加载 TXT、PDF、Markdown、网页等各类数据源
"""
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
    CSVLoader,
    DirectoryLoader,
)

# ====== 1. 加载纯文本 (.txt) ======
# 最简单、最常用的 loader
loader = TextLoader("./data/readme.txt", encoding="utf-8")
docs = loader.load()  # 返回 List[Document]
print(f"TXT: 加载了 {len(docs)} 个文档")
# 每个 Document 对象包含:
#   - page_content: 文本内容
#   - metadata: 元数据（如 source 文件路径）

# ====== 2. 加载 PDF ======
# 需要安装: pip install pypdf
try:
    loader = PyPDFLoader("./data/report.pdf")
    docs = loader.load()  # 默认每页为一个 Document
    print(f"PDF: 加载了 {len(docs)} 页")
    for i, doc in enumerate(docs[:3]):
        print(f"  第{i+1}页: {len(doc.page_content)} 字符")
except Exception as e:
    print(f"PDF 加载跳过: {e}")

# ====== 3. 加载 Markdown ======
# 需要安装: pip install unstructured markdown
try:
    loader = UnstructuredMarkdownLoader("./data/doc.md")
    docs = loader.load()
    print(f"Markdown: 加载了 {len(docs)} 个文档")
except Exception as e:
    print(f"Markdown 加载跳过: {e}")

# ====== 4. 加载网页 ======
# 需要安装: pip install beautifulsoup4
try:
    loader = WebBaseLoader("https://docs.python.org/3/tutorial/index.html")
    # 网页内容需要清理 HTML 标签，WebBaseLoader 会自动处理
    docs = loader.load()
    print(f"网页: 加载了 {len(docs)} 个文档，总字符数: {len(docs[0].page_content)}")
except Exception as e:
    print(f"网页加载跳过: {e}")

# ====== 5. 加载 CSV ======
loader = CSVLoader("./data/data.csv")  # 每行变为一个 Document
try:
    docs = loader.load()
    print(f"CSV: 加载了 {len(docs)} 行")
except Exception as e:
    print(f"CSV 加载跳过: {e}")

# ====== 6. 批量加载目录 ======
# 加载整个目录中所有匹配的文件
try:
    loader = DirectoryLoader(
        "./docs/",
        glob="**/*.txt",  # 匹配所有 txt 文件
        loader_cls=TextLoader,  # 用什么 loader 处理每个文件
        show_progress=True,  # 显示进度条
    )
    docs = loader.load()
    print(f"目录: 加载了 {len(docs)} 个 txt 文件")
except Exception as e:
    print(f"目录加载跳过: {e}")
```

#### Text Splitter —— 切分策略决定检索质量

Text Splitter 的选择不是随便的事。不同的文档类型、不同的 chunk 大小、不同的重叠量，都会直接影响检索的准确率和召回率。`RecursiveCharacterTextSplitter` 是目前最推荐的通用方案：它先尝试用 `\n\n`（段落间空行）切分，如果某段仍太大就用 `\n`（行间换行），还不够就用 `。`（中文句号），最后才按字符硬切。这种"从粗到细"的递归策略最大限度地保留了自然的语义边界。

`chunk_overlap` 是一个初学者容易忽略但极其重要的参数。假设你设置 chunk_size=200 且 overlap=40，这意味着相邻两个 chunk 之间有 40 个字符是重复的。为什么需要重复？因为如果你搜索的信息恰好落在 chunk 1 的末尾和 chunk 2 的开头之间，没有 overlap 的话这个信息就被"腰斩"了——任一 chunk 单独看都不包含完整信息。overlap 确保了关键信息至少在某一个 chunk 中是完整的。

```python
"""
ex_7_5_text_splitter.py: LangChain Text Splitter 详解
演示：各种切分策略及其对检索质量的影响
"""
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  # 最常用的递归切分器
    CharacterTextSplitter,           # 按固定字符切分
    MarkdownHeaderTextSplitter,      # 按 Markdown 标题层级切分
    TokenTextSplitter,               # 按 token 数量切分
)

# 准备一段测试文本
sample_text = """
# 第一章 Python 基础

Python 是一种解释型、面向对象的高级编程语言。由 Guido van Rossum 于 1991 年首次发布。

## 1.1 变量与类型

Python 是动态类型语言，变量不需要声明类型。常见的数据类型包括：
- int: 整数，如 42
- float: 浮点数，如 3.14
- str: 字符串，如 "Hello"
- list: 列表，如 [1, 2, 3]
- dict: 字典，如 {"key": "value"}

## 1.2 控制流

Python 使用缩进来表示代码块。

if 语句的基本格式：
if condition:
    do_something()
elif other_condition:
    do_other()
else:
    do_default()

循环有 for 和 while 两种。
for item in iterable:
    process(item)
"""


# ====== 1. RecursiveCharacterTextSplitter（推荐） ======
# 原理: 先尝试用双换行("\n\n")切分；若某段仍太大，再用单换行("\n")；还不够用空格(" ")；
#       最终按字符切。这样最大程度保留自然段落边界。
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,     # 每块最多 200 字符
    chunk_overlap=40,   # 块与块之间重叠 40 字符
    separators=["\n\n", "\n", "。", ".", " ", ""],  # 优先级从高到低
    length_function=len,  # 长度计算函数
)
chunks = splitter.create_documents([sample_text])
print(f"RecursiveCharacterTextSplitter: {len(chunks)} 块")
for i, chunk in enumerate(chunks):
    print(f"  [{i}] {chunk.page_content[:80]}... (长度={len(chunk.page_content)})")

# chunk_overlap 的作用: 保证相邻两块之间有重叠内容
# 例如: 块A = "苹果很好吃，香蕉也不"
#       块B = "香蕉也不错，橘子更甜"
# 重叠部分 = "香蕉也不"（出现在 A 和 B 中）
# 目的: 防止关键信息刚好落在边界上被"腰斩"


# ====== 2. CharacterTextSplitter: 按单个字符切分 ======
splitter = CharacterTextSplitter(
    separator="\n",     # 只按换行切
    chunk_size=200,
    chunk_overlap=40,
)
chunks = splitter.create_documents([sample_text])
print(f"\nCharacterTextSplitter: {len(chunks)} 块")


# ====== 3. TokenTextSplitter: 按 token 数切分 ======
# 比按字符数更精确（因为 LLM 是按 token 计费的）
splitter = TokenTextSplitter(
    chunk_size=50,          # 每块 50 tokens
    chunk_overlap=10,
    encoding_name="cl100k_base",  # GPT-4 使用的编码
)
chunks = splitter.create_documents([sample_text])
print(f"TokenTextSplitter: {len(chunks)} 块")


# ====== 4. MarkdownHeaderTextSplitter: 按标题层级切分 ======
# 适用场景: Markdown 文档，希望保持标题-内容的归属关系
try:
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),    # 一级标题
            ("##", "h2"),   # 二级标题
        ],
        strip_headers=False,  # 是否在内容中去掉标题行本身
    )
    chunks = splitter.split_text(sample_text)
    print(f"\nMarkdownHeaderTextSplitter: {len(chunks)} 块")
    for chunk in chunks:
        print(f"  [{chunk.metadata.get('h1', '')} > {chunk.metadata.get('h2', '')}] : "
              f"{chunk.page_content[:60]}...")
except Exception as e:
    print(f"MarkdownHeaderTextSplitter 跳过: {e}")
```

**切分策略选择指南**：

| 文档类型 | 推荐切分器 | chunk_size 建议 | overlap 建议 |
|----------|-----------|----------------|-------------|
| 纯文本/文章 | RecursiveCharacterTextSplitter | 500-1000 字符 | 100-200 |
| 代码 | 按函数/类切分（Language Splitter） | 看函数长度 | 0（代码不需要重叠） |
| Markdown 文档 | MarkdownHeaderTextSplitter | 按章节 | 0 |
| 法律/合同（长段落） | RecursiveCharacterTextSplitter | 1000-2000 字符 | 200-300 |
| FAQ/对话 | 按 Q&A 对切 | 一条 Q&A | 0 |

#### Embedding 与 Vector Store

```python
"""
ex_7_6_embedding_vectorstore.py: Embedding 与向量存储
演示：文本向量化 + Chroma 向量库存储 + 相似度检索
"""
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

# ====== 1. 初始化 Embedding 模型 ======
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 性价比高（比 3-large 便宜 10 倍）
    # model="text-embedding-3-large",  # 更准确但更贵
    # dimensions=1024,  # 可降维（3-small 默认 1536，3-large 默认 3072）
    # 降维可以减少向量存储大小，代价是检索精度轻微下降
)

# 快速测试：将单个文本转为向量
sample_text = "Python 是一种高级编程语言"
vector = embeddings.embed_query(sample_text)
print(f"向量维度: {len(vector)}")
print(f"向量前5个值: {vector[:5]}")
# 每个 embedding 向量是一个高维空间中的点
# 语义相近的文本在向量空间中距离也近


# ====== 2. 加载文档并切分 ======
# 假设有一批产品文档
documents = [
    "Python 是一种解释型语言，适合快速开发。语法简洁，拥有丰富的第三方库。",
    "Java 是一种编译型语言，跨平台能力强。广泛应用于企业级开发。",
    "JavaScript 主要用于 Web 前端开发，也可以在 Node.js 后端运行。",
    "Docker 是一个容器化平台，可以让应用在隔离的环境中运行。",
    "Kubernetes 是容器编排平台，管理大规模的 Docker 容器集群。",
    "Redis 是一个内存数据库，常用于缓存、消息队列和会话管理。",
]

# 将文本转为 Document 对象列表
from langchain_core.documents import Document
docs = [Document(page_content=text) for text in documents]

# 切分（本例文本较短，不做切分也可以，但实际项目应该切分）
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"\n切分后: {len(chunks)} 块")


# ====== 3. 存入 Chroma 向量库 ======
# Chroma 是一个轻量级的向量数据库，数据保存在本地磁盘
persist_dir = "./chroma_db"

# 如果目录已存在，先清空（避免重复数据）
import shutil
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir,  # 持久化到磁盘
    # collection_name="my_knowledge",  # 集合名（同一数据库可以有多个集合）
)
print(f"向量库已创建，保存到: {persist_dir}")

# 下次启动时可以直接加载已有的向量库（不需要重新生成 embedding）
# vectorstore = Chroma(
#     persist_directory=persist_dir,
#     embedding_function=embeddings,
# )


# ====== 4. 相似度检索 ======
query = "哪种语言适合 Web 开发？"

# 方式A: similarity_search 直接返回文档
results = vectorstore.similarity_search(query, k=2)
print(f"\n查询: {query}")
print("检索结果:")
for i, doc in enumerate(results):
    print(f"  [{i}] {doc.page_content[:80]}...")

# 方式B: similarity_search_with_score 返回文档 + 相似度分数
# 分数越低 = 距离越近 = 越相似
results_with_score = vectorstore.similarity_search_with_score(query, k=2)
print("\n带分数的检索结果:")
for i, (doc, score) in enumerate(results_with_score):
    print(f"  [{i}] score={score:.4f} | {doc.page_content[:80]}...")

# 方式C: similarity_search_by_vector 用已有向量检索
# 适用场景: 自己先生成查询向量（可能经过某种变换），再搜
query_vector = embeddings.embed_query(query)
results = vectorstore.similarity_search_by_vector(query_vector, k=2)


# ====== 5. 进阶: MMR (Maximal Marginal Relevance) 检索 ======
# 普通检索只按相似度排序，可能导致 Top-K 结果内容雷同
# MMR 在相似度和多样性之间做平衡
results_mmr = vectorstore.max_marginal_relevance_search(
    query,
    k=3,            # 返回 3 个结果
    fetch_k=10,     # 先从 Top-10 中选
    lambda_mult=0.5, # 多样性权重: 0=完全多样, 1=完全相似
)
print("\nMMR 检索结果 (更注重多样性):")
for i, doc in enumerate(results_mmr):
    print(f"  [{i}] {doc.page_content[:80]}...")
```

#### Retriever —— 统一的检索接口

```python
"""
ex_7_7_retrievers.py: LangChain Retriever 详解
演示：各种 Retriever 的配置和使用
"""
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 准备向量库（继承自 7_6 的 chroma_db）
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 如果已有向量库则加载，否则新建
import os
if os.path.exists("./chroma_db"):
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
else:
    # 没有就跳过这个示例
    print("请先运行 ex_7_6_embedding_vectorstore.py 创建向量库")
    exit()


# ====== 1. 基础 Retriever ======
# 最简单的检索器: 给定 query，返回 k 个最相似文档
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 相似度检索
    search_kwargs={"k": 3},    # 返回前 3 个
)
docs = retriever.invoke("什么是 Python？")
print("基础检索结果:")
for doc in docs:
    print(f"  - {doc.page_content[:60]}...")


# ====== 2. MMR Retriever（多样性检索） ======
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5,
    },
)
docs = retriever_mmr.invoke("编程语言")
print("\nMMR 检索结果:")
for doc in docs:
    print(f"  - {doc.page_content[:60]}...")


# ====== 3. 相似度阈值过滤 ======
# 只返回相似度高于某个阈值的结果
retriever_threshold = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.3,  # 只返回 score > 0.3 的结果
        "k": 5,
    },
)
docs = retriever_threshold.invoke("容器化技术")
print(f"\n阈值过滤检索: 返回 {len(docs)} 个结果")
for doc in docs:
    print(f"  - {doc.page_content[:60]}...")


# ====== 4. 组合多个 Retriever ======
# 从不同数据源分别检索，然后合并结果
# 适用场景: 同时搜知识库 + 产品文档 + FAQ
from langchain.retrievers import EnsembleRetriever

# 创建两个带不同参数的检索器
retriever1 = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever2 = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10}
)

# 组合: 两个检索器各返回 3 个结果，去重后合并
ensemble = EnsembleRetriever(
    retrievers=[retriever1, retriever2],
    weights=[0.7, 0.3],  # retriever1 权重 70%，retriever2 权重 30%
)
docs = ensemble.invoke("编程语言")
print(f"\n组合检索: 返回 {len(docs)} 个结果")


# ====== 5. Self-Query Retriever（自查询检索） ======
# 从用户的自然语言问题中提取查询条件
# 例如用户说 "搜索2024年发布的Python教程"，模型会提取:
#   查询内容="Python教程", 过滤条件="year=2024"
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

# 定义文档的元数据字段
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="文档来源文件",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="文档发布年份",
        type="integer",
    ),
]

# 描述文档内容
document_content_description = "技术教程和文档"

# 理论上这样创建:
# retriever = SelfQueryRetriever.from_llm(
#     llm=ChatOpenAI(model="gpt-4o", temperature=0),
#     vectorstore=vectorstore,
#     document_contents=document_content_description,
#     metadata_field_info=metadata_field_info,
# )
# docs = retriever.invoke("搜索2024年发布的关于Python的文档")
```

### 7.1.5 Chains —— 编排的艺术

Chain 是 LangChain 的"编排层"。从设计哲学上，Chain 将 LLM 应用的构建从"编写过程式代码"转变为"声明式地定义数据流"。你不需要写 `result1 = step1(input); result2 = step2(result1); ...`，而是声明 `chain = step1 | step2 | step3`，然后调用 `chain.invoke(input)`。这种声明式风格让你的代码更接近流程图，更容易理解和维护。

下面的 RouterChain 展示了 Chain 的一个核心价值：**条件分支**。在实际应用中，你不会把所有问题都交给同一个 LLM 处理——不同的问题需要不同的 system prompt、不同的处理逻辑。RouterChain 让 LLM 先对输入做分类，然后根据分类结果自动路由到不同的处理链。

```python
"""
ex_7_8_chains.py: LangChain Chains 详解
演示：LLMChain、SequentialChain、RouterChain
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# ====== 1. 简单 Chain: Prompt + LLM + Parser ======
# 这是最基本的链——三个步骤用管道连接
prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译为{target_lang}: {text}"
)
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"target_lang": "英文", "text": "人工智能正在改变世界"})
print(f"简单 Chain: {result}")


# ====== 2. 复杂的 SequentialChain（多步串联） ======
# 旧版 API 的 LLMChain 和 SequentialChain 已过时，
# 新版推荐用 LCEL（下一节详细介绍）。
# 这里先用 LCEL 展示多步流水线:

# 步骤1: 提取关键词
extract_prompt = ChatPromptTemplate.from_template(
    "从以下文本中提取 3-5 个关键词，用逗号分隔: {text}"
)
extract_chain = extract_prompt | llm | StrOutputParser()

# 步骤2: 根据关键词生成摘要
summarize_prompt = ChatPromptTemplate.from_template(
    "根据以下关键词创作一段简短摘要（100字以内）: {keywords}"
)
summarize_chain = summarize_prompt | llm | StrOutputParser()

# 步骤3: 将摘要翻译为英文
translate_prompt = ChatPromptTemplate.from_template(
    "将以下中文翻译为英文: {summary}"
)
translate_chain = translate_prompt | llm | StrOutputParser()

# 串联: 先提取关键词 to 再生成摘要 to 再翻译
# 使用 RunnableLambda 做中间处理
from langchain_core.runnables import RunnableLambda

def extract_keywords_output(input_dict):
    """提取关键词步骤"""
    result = extract_chain.invoke({"text": input_dict["text"]})
    return {"keywords": result, "text": input_dict["text"]}

def summarize_output(input_dict):
    """生成摘要步骤"""
    result = summarize_chain.invoke({"keywords": input_dict["keywords"]})
    return {"summary": result}

def translate_output(input_dict):
    """翻译步骤"""
    result = translate_chain.invoke({"summary": input_dict["summary"]})
    return {"translation": result, "summary": input_dict["summary"]}

# 用 RunnableLambda 包装纯函数
full_pipeline = (
    RunnableLambda(extract_keywords_output)
    | RunnableLambda(summarize_output)
    | RunnableLambda(translate_output)
)

result = full_pipeline.invoke({
    "text": "Python 是一门简洁高效的编程语言，在数据科学、人工智能和 Web 开发领域广泛应用。"
})
print(f"\n多步流水线结果:")
print(f"  摘要: {result['summary']}")
print(f"  翻译: {result['translation']}")


# ====== 3. RouterChain（路由链） ======
# 根据输入内容的不同，自动路由到不同的处理链
# 适用场景: 客服系统根据问题类型分流

# 定义路由模板
router_prompt = ChatPromptTemplate.from_template(
    """判断以下用户问题的类型，只回答一个词:
- "technical" 如果问题是技术问题（代码、Bug、配置等）
- "billing" 如果问题是账单/付款相关
- "general" 如果是一般性问题

用户问题: {question}

类型: """
)

# 不同类型的处理链
tech_chain = (
    ChatPromptTemplate.from_template(
        "你是一位技术支持工程师。请解决以下问题: {question}"
    )
    | llm
    | StrOutputParser()
)

billing_chain = (
    ChatPromptTemplate.from_template(
        "你是客服部门的账单专员。请处理以下账单问题: {question}"
    )
    | llm
    | StrOutputParser()
)

general_chain = (
    ChatPromptTemplate.from_template(
        "你是一位友好的客服人员。请回答: {question}"
    )
    | llm
    | StrOutputParser()
)


def route(info):
    """根据分类结果路由到不同的链"""
    question = info["question"]

    # 先分类
    category = (router_prompt | llm | StrOutputParser()).invoke({"question": question})
    category = category.strip().lower()
    print(f"  路由到: {category}")

    # 再路由
    if "billing" in category:
        return billing_chain.invoke({"question": question})
    elif "technical" in category:
        return tech_chain.invoke({"question": question})
    else:
        return general_chain.invoke({"question": question})


# 包装为 Runnable
router_chain = RunnableLambda(route)

questions = [
    "我的代码报错 'NameError: name x is not defined'，怎么解决？",
    "为什么这个月扣了我两次费用？",
    "你们的营业时间是几点？",
]

print("\n=== 路由链测试 ===")
for q in questions:
    print(f"\n问题: {q}")
    answer = router_chain.invoke({"question": q})
    print(f"回复: {answer[:100]}...")
```

### 7.1.6 Agents —— 让 LLM 自主决策

Agent 是 LangChain 中从"自动化"到"智能化"的质变。在 Chain 中，每一步做什么是预先写死在代码里的。在 Agent 中，LLM 自主决定用什么工具、按什么顺序、做到什么程度算完成。这个循环是：Agent 收到问题 → 思考（该用什么工具？）→ 行动（调用工具）→ 观察（工具返回的结果）→ 思考（结果够了吗？）→ 如果够了就输出最终答案，不够就继续循环。

`@tool` 装饰器是定义工具的最简单方式。被装饰的函数的 docstring 会作为工具描述发给 LLM，所以写清楚"这个工具是做什么的、参数是什么"非常重要——LLM 需要根据这些描述来决定是否使用这个工具。

```python
"""
ex_7_9_agent.py: LangChain Agent 详解
演示：创建带工具的 Agent，LLM 自主决定调用哪些工具
"""

# 注意：LangChain 的 Agent API 在不同版本间变化很大。
# 以下代码基于 langchain >= 0.3.0, langchain-openai >= 0.3.0

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import math
import datetime

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# ====== 1. 定义工具（Tools） ======
# 工具就是 LLM 可以调用的函数，用 @tool 装饰器声明
# 函数的 docstring 会作为工具的描述发给模型

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持加减乘除、幂运算、三角函数等。
    参数: expression - 数学表达式字符串，如 "2+3*4" 或 "sqrt(16)" """
    try:
        # 安全地计算表达式
        # 注意: eval 有安全风险，生产环境应用更安全的方案
        allowed_names = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e, "abs": abs, "pow": pow,
            "round": round, "int": int, "float": float,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算出错: {e}"


@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前日期和时间。
    参数: format - 时间格式字符串（可选），默认 "%Y-%m-%d %H:%M:%S" """
    return datetime.datetime.now().strftime(format)


@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中搜索技术文档。
    参数: query - 搜索关键词或问题
    返回: 相关的文档片段"""
    # 模拟知识库（实际项目中接入向量库）
    knowledge = {
        "python": "Python 是一种解释型、面向对象的高级编程语言，由 Guido van Rossum 创建。",
        "docker": "Docker 使用容器技术打包应用及其依赖，实现环境一致性。",
        "git": "Git 是一个分布式版本控制系统，用于跟踪代码变更。",
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return "未找到相关信息。"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送电子邮件（模拟）。
    参数:
        to - 收件人邮箱地址
        subject - 邮件主题
        body - 邮件正文"""
    # 模拟发送（实际项目接入 SMTP）
    return f"邮件已发送给 {to}\n主题: {subject}\n正文: {body[:50]}..."


# ====== 2. 创建 Agent ======
# 将工具列表注册给 Agent
tools = [calculator, get_current_time, search_knowledge_base, send_email]

# Agent 的系统提示
system_prompt = """你是一个有用的 AI 助手。你可以使用提供的工具来帮助用户。

使用工具时:
1. 先理解用户的真实需求
2. 选择合适的工具
3. 如果一个问题需要多个工具，按合理顺序调用
4. 用中文回复用户
5. 如果工具返回错误，向用户解释并尝试其他方法

当前时间: {current_time}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # Agent 思考过程
])

# 创建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建 AgentExecutor（负责 Agent 的循环调用和错误处理）
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,         # 打印思考过程
    handle_parsing_errors=True,  # 处理输出解析错误
    max_iterations=10,    # 最大工具调用次数（防止死循环）
    # max_execution_time=30,  # 最大执行时间（秒）
)

# ====== 3. 测试 Agent ======
# Agent 的思考to行动to观察to思考... 循环
# 流程:
#   1. Agent 收到问题
#   2. Thought: 我该用什么工具？
#   3. Action: 调用某个工具
#   4. Observation: 工具返回结果
#   5. Thought: 结果够了吗？还需要更多信息吗？
#   6. 如果够了 to Final Answer; 不够 to 回到步骤 2

test_queries = [
    "现在几点了？",
    "帮我算一下 123 * 456 + 789",
    "告诉我 Python 是什么，然后用中文发一封邮件给 admin@example.com",
    "查一下 Docker 是什么，然后告诉我现在的时间",
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"用户: {query}")
    result = agent_executor.invoke({
        "input": query,
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"Agent: {result['output']}")

# ====== 4. 带对话历史的 Agent ======
# Agent 也能记住之前的对话（通过 chat_history）

from langchain_core.messages import HumanMessage, AIMessage

chat_history = []

print(f"\n=== 多轮对话 Agent ===")
questions = [
    "现在几点了？",
    "刚才你说是几点？",  # 依赖上一轮的上下文
]

for q in questions:
    print(f"\n用户: {q}")
    result = agent_executor.invoke({
        "input": q,
        "chat_history": chat_history,
        "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"Agent: {result['output']}")

    # 更新对话历史
    chat_history.append(HumanMessage(content=q))
    chat_history.append(AIMessage(content=result['output']))
```

---

## 7.2 LCEL —— LangChain 的现代编排语法

### 7.2.1 管道语法与 Runnable 接口的设计理念

LCEL（LangChain Expression Language）是 LangChain 在 2023 年底推出的全新 API 范式。它的核心设计理念受到了 Unix 管道的启发：**每个组件都是独立的黑盒，通过统一的接口连接**。`|` 操作符就是管道——`a | b` 表示 a 的输出作为 b 的输入。

这种设计带来了几个重要的好处。第一，**可组合性**：任何 Runnable 对象都能用 `|` 连接，无论它是 Prompt Template、Chat Model、Output Parser、Retriever 还是自定义函数。第二，**统一接口**：所有 Runnable 都实现了 `invoke()`、`batch()`、`stream()`、`ainvoke()` 等方法，这意味着你可以在同步和异步之间自由切换，还可以开启流式输出。第三，**内置并行**：`RunnableParallel` 让多个分支同时执行，这在需要"同时翻译、摘要、提取关键词"的场景中特别有用——三个独立任务并发出去了，响应时间等于最慢的那个而不是三个之和。

```python
"""
ex_7_10_lcel_basics.py: LCEL 管道语法基础
演示：Runnable 接口的 invoke/batch/stream/invoke_async
"""
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ====== 1. 管道语法基础 ======
# | 操作符: a | b 表示 a 的输出作为 b 的输入

# 最简单的链: prompt to llm to parser
prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
chain = prompt | llm | StrOutputParser()

# 所有 Runnable 对象都实现了统一的接口
# Runnable 接口方法:
#   invoke(input)       to 同步执行，返回结果
#   batch(inputs)       to 批量执行
#   stream(input)       to 流式输出
#   ainvoke(input)      to 异步 invoke
#   abatch(inputs)      to 异步 batch
#   astream(input)      to 异步 stream

# === invoke ===
result = chain.invoke({"topic": "程序员"})
print(f"invoke: {result[:50]}...")

# === batch: 批量处理 ===
# 批量比逐个调用更快（内部可能并行）
results = chain.batch([
    {"topic": "程序员"},
    {"topic": "厨师"},
    {"topic": "医生"},
])
for i, r in enumerate(results):
    print(f"batch[{i}]: {r[:50]}...")


# ====== 2. RunnablePassthrough: 透传 ======
# 在某些链中，你需要把前面的输出原封不动地传给后面

# 场景：先给文本翻译，但需要保留原文做对比
translate_chain = (
    ChatPromptTemplate.from_template("将以下内容翻译为英文: {text}")
    | llm
    | StrOutputParser()
)

# 使用 RunnablePassthrough 保留原始输入
combined_chain = RunnableParallel(
    original=RunnablePassthrough(),
    translated=translate_chain,
)

result = combined_chain.invoke({"text": "人工智能正在改变世界"})
print(f"\n原始: {result['original']}")
print(f"翻译: {result['translated']}")


# ====== 3. RunnableLambda: 插入自定义函数 ======
# 在链的任意位置插入一个普通 Python 函数

def count_chars(text: str) -> str:
    """统计字符数"""
    return f"{text}\n\n(以上共 {len(text)} 个字符)"

def add_timestamp(text: str) -> str:
    """添加时间戳"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    return f"[{ts}]\n{text}"

# 在 LLM 输出后插入自定义函数
chain_with_fn = (
    prompt | llm | StrOutputParser()
    | RunnableLambda(count_chars)      # 统计字符
    | RunnableLambda(add_timestamp)    # 加时间戳
)

result = chain_with_fn.invoke({"topic": "Python"})
print(f"\n带函数处理:\n{result}")


# ====== 4. RunnableParallel: 并行执行 ======
# 同时执行多个子链，合并结果

# 场景：对同一段文本同时做翻译、摘要、关键词提取
text = "Artificial intelligence has revolutionized how we interact with technology. "

translate = (
    ChatPromptTemplate.from_template("翻译为中文: {text}")
    | llm | StrOutputParser()
)
summarize = (
    ChatPromptTemplate.from_template("用一句话总结: {text}")
    | llm | StrOutputParser()
)
keywords = (
    ChatPromptTemplate.from_template("提取3个关键词，逗号分隔: {text}")
    | llm | StrOutputParser()
)

# 三个分支并行执行
parallel_chain = RunnableParallel(
    translation=translate,
    summary=summarize,
    keywords=keywords,
)

result = parallel_chain.invoke({"text": text})
print(f"\n并行处理结果:")
print(f"  翻译: {result['translation']}")
print(f"  摘要: {result['summary']}")
print(f"  关键词: {result['keywords']}")


# ====== 5. stream: 流式输出 ======
# 逐 token 输出，适合聊天场景
print("\n=== 流式输出 ===")
for chunk in chain.stream({"topic": "人工智能"}):
    print(chunk, end="", flush=True)
print()
```

### 7.2.2 用 LCEL 构建完整的 RAG Chain

下面这段代码展示了 LCEL 的优雅之处：整个 RAG 流程（检索 → 格式化 → 注入 Prompt → LLM 生成）被压缩为一条声明式的管道。数据流一目了然——`{"context": retriever | format_docs, "question": RunnablePassthrough()}` 声明了两个并行的数据来源：context 来自检索和格式化，question 来自用户输入的透传。然后这两个变量被注入到 Prompt 模板中，结果送给 LLM。

```python
"""
ex_7_11_lcel_rag.py: 用 LCEL 构建完整的 RAG Chain
演示：文档加载to切分to向量化to检索to生成答案 的完整流水线
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document


def build_rag_chain(
    documents: list[str],
    model_name: str = "gpt-4o-mini",
) -> object:
    """
    构建一个完整的 RAG 链。

    参数:
        documents: 文档字符串列表
        model_name: LLM 模型名

    返回:
        一个可调用的 RAG chain（invoke 传入 {"question": "..."} 即可）

    RAG 的数据流:
        question
          +--to retriever to context（检索到的相关文档）
          +--to question（原始问题，通过 RunnablePassthrough 透传）
               |
        prompt.format(question, context)
               |
        llm.invoke(formatted_prompt)
               |
        StrOutputParser.parse(llm_output)
               |
        answer
    """
    # 步骤1: 创建 Document 对象
    docs = [Document(page_content=text) for text in documents]

    # 步骤2: 切分文本
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(docs)
    print(f"文档切分为 {len(chunks)} 块")

    # 步骤3: 创建向量库和检索器
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        # 使用内存模式（不持久化），适合演示
        # 生产环境应设置 persist_directory
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},  # 取 Top-3 相关片段
    )

    # 步骤4: 定义 Prompt 模板
    # 这个模板把检索到的上下文和用户问题组合在一起
    template = """你是一位知识丰富的助手。请根据以下参考资料回答用户问题。

参考资料:
{context}

用户问题: {question}

回答要求:
1. 优先使用参考资料中的信息
2. 如果参考资料不够，可以结合你的知识补充，但要明确说明
3. 用中文回答，语言简洁清晰
4. 如果无法回答，直接说"根据现有资料，我无法回答这个问题。"

回答: """

    prompt = ChatPromptTemplate.from_template(template)

    # 步骤5: 构建 RAG Chain
    llm = ChatOpenAI(model=model_name, temperature=0.3)

    # 关键: 定义一个函数将检索到的文档拼接成上下文字符串
    def format_docs(docs: list[Document]) -> str:
        """将检索到的文档列表拼接为一段上下文字符串"""
        return "\n\n---\n\n".join(
            f"[来源 {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

    # 构建链:
    # {
    #   context: retriever to format_docs    (检索 + 格式化)
    #   question: RunnablePassthrough()     (透传原始问题)
    # }
    # to prompt to llm to StrOutputParser
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ====== 测试 RAG Chain ======
if __name__ == "__main__":
    # 准备知识库文档
    knowledge_base = [
        """Python 是一种解释型、面向对象的高级编程语言。
        Python 的设计哲学强调代码的可读性和简洁的语法。
        它支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
        Python 拥有一个庞大的标准库，被称为"自带电池(Batteries Included)"。""",

        """Docker 是一个开源的应用容器引擎，让开发者可以打包他们的应用以及依赖包
        到一个可移植的容器中，然后发布到任何流行的 Linux 机器上。
        容器是完全沙箱机制，相互之间不会有任何接口。
        Docker 使用客户端-服务器架构，Docker 守护进程负责构建、运行和分发容器。""",

        """Git 是 Linus Torvalds 开发的分布式版本控制系统。
        与 CVS、Subversion 等集中式版本控制系统不同，Git 每个工作目录
        都是一个完整的仓库，包含完整的版本历史。
        Git 的核心概念包括: 工作区、暂存区、本地仓库、远程仓库。""",

        """RESTful API 是一种符合 REST (Representational State Transfer)
        架构风格的 Web API 设计规范。它使用 HTTP 方法(GET, POST, PUT, DELETE)
        来操作资源，资源通过 URL 来标识。RESTful API 是无状态的，
        每个请求都包含完整的信息。""",
    ]

    # 构建 RAG 链
    rag = build_rag_chain(knowledge_base)

    # 测试查询
    questions = [
        "Python 的设计哲学是什么？",
        "Docker 和 Git 有什么区别？",
        "什么是无状态？",
    ]

    print("\n" + "=" * 60)
    for q in questions:
        print(f"\n问题: {q}")
        answer = rag.invoke(q)  # 注意: 直接传字符串，不是 dict
        print(f"回答: {answer}")
        print("-" * 40)
```

### 7.2.3 LCEL vs 旧 Chain API

| 特性 | 旧 Chain API (LLMChain 等) | LCEL |
|------|---------------------------|------|
| 语法 | 面向对象，需继承类 | 函数式，管道操作符 `|` |
| 组合 | 嵌套复杂 | 直观的线性管道 |
| 并行 | 需手动管理 | `RunnableParallel` 一行搞定 |
| 流式 | 部分支持 | 原生 `stream()` 支持 |
| 异步 | 部分支持 | 原生 `ainvoke()/astream()` |
| 调试 | 困难 | 每个步骤可以单独 debug |
| 类型 | 弱类型 | 运行时可检查输入输出 schema |
| 推荐度 | **已过时，不推荐新项目使用** | **官方推荐，稳定 API** |

---

## 7.3 LlamaIndex 核心架构

### 7.3.1 两种设计哲学的对比

如果 LangChain 是一把瑞士军刀（什么都能做），LlamaIndex 就是一座专业的图书馆（专注于数据索引和检索）。这个比喻不是为了褒贬谁，而是揭示它们不同的设计哲学。

**LangChain 从"流程"出发**：它的核心问题是"LLM 应用的各个步骤如何编排？"所以它最核心的抽象是 Chain 和 Agent——定义数据如何在这一步和下一步之间流动。LangChain 的宇宙里，LLM 是"大脑"，一切组件都是可以被大脑调用的工具或数据源。

**LlamaIndex 从"数据"出发**：它的核心问题是"如何把一堆数据组织成 LLM 能高效理解和检索的结构？"所以它最核心的抽象是 Index 和 Query Engine——定义数据如何被索引、查询如何被路由到最相关的那部分数据。在 LlamaIndex 的宇宙里，数据是"地基"，LLM 是建在地基上的查询和生成层。

这种哲学差异体现在 API 设计上。在 LangChain 做一个 RAG，你需要手动组合 Loader → Splitter → Embedding → VectorStore → Retriever 五个组件。在 LlamaIndex，你只需 `VectorStoreIndex.from_documents(docs)` 一行——它认为"数据索引"是一个原子操作，内部的切分、embedding、存储都应该自动完成。

**两种哲学不是互斥的，而是互补的**。在前面章节中我们说过，"先框架还是先 SDK"的决策取决于复杂度。LangChain vs LlamaIndex 的决策则取决于你的主要开销在"编排"还是"检索"：如果 Agent 和工具调用是你系统的核心，LangChain 是更好的骨架；如果知识库检索是你系统的核心，LlamaIndex 能让你更快地做出高质量的原型。最佳实践通常是**组合使用**——用 LlamaIndex 构建索引和检索层，用 LangChain 构建 Agent 和业务编排层。

```bash
pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
```

### 7.3.2 LlamaIndex 基础入门 —— 从文档到问答的最短路径

LlamaIndex 的使用流程极具特色：`Settings` 全局配置 → 创建 Document → `from_documents` 自动索引 → `as_query_engine()` 获取查询引擎 → `query()` 提问。五步完成一个可工作的 RAG 系统，不需要手动管理切分、embedding、存储。这个体验比 LangChain 简洁得多，但代价是灵活性降低——如果你需要自定义切分策略或 embedding 模型，仍然需要深入了解 LlamaIndex 的内部配置。

```python
"""
ex_7_12_llamaindex_basics.py: LlamaIndex 基础入门
演示：从文档到索引到查询的完整流程
"""
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Document

# ====== 1. 全局配置 ======
# LlamaIndex 使用全局 Settings 对象管理默认配置
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Settings.chunk_size = 512       # 默认切分大小
# Settings.chunk_overlap = 100    # 默认重叠大小

# ====== 2. 创建 Document 对象 ======
# LlamaIndex 的 Document 与 LangChain 的 Document 类似
documents = [
    Document(
        text="Python 是一种解释型、面向对象的高级编程语言，由 Guido van Rossum 创建。"
             "Python 的设计哲学强调代码可读性，使用缩进来定义代码块。",
        metadata={"source": "python_intro", "category": "programming"},
    ),
    Document(
        text="Docker 是一个容器化平台，允许将应用和依赖打包在一起。"
             "容器是轻量级的、可移植的，可以在任何支持 Docker 的平台上运行。",
        metadata={"source": "docker_intro", "category": "devops"},
    ),
    Document(
        text="机器学习是人工智能的一个分支，让计算机从数据中学习模式。"
             "常见的算法包括线性回归、决策树、支持向量机和神经网络。",
        metadata={"source": "ml_intro", "category": "ai"},
    ),
    Document(
        text="FastAPI 是一个现代、快速的 Python Web 框架，用于构建 API。"
             "它基于 Python 类型提示，支持自动生成 OpenAPI 文档。",
        metadata={"source": "fastapi_intro", "category": "web"},
    ),
]

# ====== 3. 创建索引 ======
# VectorStoreIndex 是 LlamaIndex 最常用的索引类型
# 内部流程: 切分文档 to 生成 embedding to 存储到向量库
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True,  # 显示处理进度
)
print(f"索引创建完成，包含 {len(documents)} 篇文档")

# ====== 4. 保存和加载索引 ======
# 持久化到磁盘（避免每次重新生成昂贵的 embedding）
index.storage_context.persist(persist_dir="./llama_index_storage")
print("索引已保存到 ./llama_index_storage")

# 下次加载:
# storage_context = StorageContext.from_defaults(
#     persist_dir="./llama_index_storage"
# )
# index = load_index_from_storage(storage_context)


# ====== 5. 查询索引 ======
# 创建查询引擎
query_engine = index.as_query_engine(
    similarity_top_k=2,  # 检索 Top-2 相关片段
    # response_mode="compact",  # 将检索到的片段压缩后给 LLM
    # streaming=False,
)

# 执行查询
questions = [
    "Python 的设计哲学是什么？",
    "Docker 有什么优点？",
    "机器学习有哪些常见算法？",
]

print("\n=== 查询结果 ===")
for q in questions:
    response = query_engine.query(q)
    print(f"\nQ: {q}")
    print(f"A: {response}")
    # response 对象包含更多信息:
    # print(f"  来源节点: {response.source_nodes}")
    # print(f"  元数据: {response.metadata}")
```

### 7.3.3 数据连接器与索引类型

LlamaIndex 的数据连接器覆盖了 PDF、网页、数据库、Notion、GitHub 等常见数据源。与 LangChain 的 Loader 概念相似，但 LlamaIndex 更进一步——它内置了元数据提取和管理能力。元数据在检索时可以用作过滤条件，这是 LangChain 需要额外配置才有的功能。

```python
"""
ex_7_13_llamaindex_connectors.py: LlamaIndex 数据连接器
演示：从各种数据源加载数据
"""
from llama_index.core import SimpleDirectoryReader, Document

# ====== 1. SimpleDirectoryReader：加载目录中的文件 ======
# 自动检测文件类型并选择合适的解析器
try:
    reader = SimpleDirectoryReader(
        input_dir="./data/",           # 源目录
        recursive=True,               # 递归子目录
        required_exts=[".txt", ".md"],  # 只加载特定扩展名
        # exclude_hidden=True,        # 排除隐藏文件
        # encoding="utf-8",           # 文件编码
    )
    documents = reader.load_data()
    print(f"从目录加载了 {len(documents)} 个文档")
    for doc in documents[:3]:
        print(f"  源: {doc.metadata.get('file_name', 'unknown')}, "
              f"长度: {len(doc.text)} 字符")
except Exception as e:
    print(f"目录加载跳过: {e}")


# ====== 2. 从各种数据源加载 ======
# LlamaIndex 提供了丰富的 Connector（需要额外安装依赖）

# PDF: pip install llama-index-readers-file
# from llama_index.readers.file import PDFReader
# reader = PDFReader()
# docs = reader.load_data("report.pdf")

# 网页: pip install llama-index-readers-web
# from llama_index.readers.web import SimpleWebPageReader
# reader = SimpleWebPageReader()
# docs = reader.load_data(["https://example.com/page1", "https://example.com/page2"])

# 数据库: pip install llama-index-readers-database
# from llama_index.readers.database import DatabaseReader
# reader = DatabaseReader(
#     scheme="postgresql",
#     host="localhost",
#     ...
# )
# docs = reader.load_data("SELECT * FROM articles")

# Notion: pip install llama-index-readers-notion
# from llama_index.readers.notion import NotionPageReader
# reader = NotionPageReader(integration_token="...")
# docs = reader.load_data(page_ids=["page_id_1", "page_id_2"])

# GitHub: pip install llama-index-readers-github
# from llama_index.readers.github import GithubRepositoryReader
# reader = GithubRepositoryReader(github_token="...", owner="...", repo="...")
# docs = reader.load_data()


# ====== 3. 手动创建 Document ======
# 也可以直接编程式创建文档
custom_docs = [
    Document(
        text="这是一篇手动创建的技术文档，内容是关于...",
        metadata={
            "title": "技术文档1",
            "author": "张三",
            "created_at": "2025-01-15",
            "tags": ["python", "tutorial"],
        },
        # metadata 中的字段可以用于后续的元数据过滤
        # excluded_llm_metadata_keys=["author"],  # 某些字段不发给 LLM
        # excluded_embed_metadata_keys=["created_at"],  # 某些字段不参与 embedding
    )
]
print(f"\n手动创建了 {len(custom_docs)} 个文档")
```

LlamaIndex 的四种索引类型对应了不同的检索需求。VectorStoreIndex 做语义搜索，SummaryIndex 做文档总结，TreeIndex 做层级式问答，KeywordTableIndex 做精确关键词匹配。大多数场景下 VectorStoreIndex 就够用了，其他索引类型解决的是特定的"非典型"需求。

```python
"""
ex_7_14_llamaindex_index_types.py: LlamaIndex 索引类型对比
演示：VectorStoreIndex、SummaryIndex、TreeIndex 的使用场景
"""
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    TreeIndex,
    KeywordTableIndex,
    Settings,
    Document,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 准备文档
documents = [
    Document(text=f"这是第{i}篇文档，内容是关于主题{i}的详细信息。" * 3)
    for i in range(20)
]

# ====== 1. VectorStoreIndex（向量索引）【最常用】 ======
# 原理: 将文档切片转为向量，查询时做相似度检索 + LLM 合成
# 适用: 大部分场景，尤其是语义搜索
index_vec = VectorStoreIndex.from_documents(documents)
query_engine = index_vec.as_query_engine(similarity_top_k=2)
result = query_engine.query("关于主题5的信息")
print(f"VectorStoreIndex: {result}")


# ====== 2. SummaryIndex（摘要索引） ======
# 原理: 将每个文档生成摘要，查询时基于摘要做匹配
# 适用: 需要对大段内容做概括的场景
index_sum = SummaryIndex.from_documents(documents)
query_engine = index_sum.as_query_engine(
    response_mode="tree_summarize",  # 递归摘要模式
)
result = query_engine.query("总结所有文档的主要内容")
print(f"\nSummaryIndex: {result}")


# ====== 3. TreeIndex（树索引） ======
# 原理: 自底向上构建一棵总结树，每个父节点是子节点的摘要
# 适用: 需要回答总结性/概括性问题的场景
# 注意: TreeIndex 构建较慢（需要多次 LLM 调用生成摘要）
# index_tree = TreeIndex.from_documents(documents)
# query_engine = index_tree.as_query_engine()
# result = query_engine.query("总结这些文档的核心观点")

# ====== 4. KeywordTableIndex（关键词表索引） ======
# 原理: 为每个文档提取关键词，建立关键词to文档的映射
# 适用: 关键词匹配精确度要求高的场景
# index_kw = KeywordTableIndex.from_documents(documents)
# query_engine = index_kw.as_query_engine()
# result = query_engine.query("主题5")
```

**索引选择指南**：

| 索引类型 | 适用场景 | 构建速度 | 查询质量 | 成本 |
|----------|---------|---------|---------|------|
| VectorStoreIndex | 通用语义搜索（推荐首选） | 快 | 高 | 低 |
| SummaryIndex | 长文档摘要 | 中 | 中-高 | 中 |
| TreeIndex | 总结性问答 | 慢 | 高（总结） | 高 |
| KeywordTableIndex | 精确关键词匹配 | 快 | 中（关键词） | 低 |
| KnowledgeGraphIndex | 实体关系查询 | 慢 | 高（关系） | 高 |

### 7.3.4 查询引擎与对话引擎

LlamaIndex 的查询引擎支持两个 LangChain 没有的能力：元数据过滤和子问题查询。元数据过滤让你在检索时加上精确的条件（如 year=2024），避免了纯语义搜索可能带回的不相关内容。子问题查询（SubQuestionQueryEngine）则会把一个复杂问题自动拆解为多个子问题，分别去对应领域的索引中查询，最后汇总——这在"多领域知识库"场景中非常有用。

```python
"""
ex_7_15_llamaindex_query_engine.py: LlamaIndex 查询引擎高级用法
演示：自定义 Prompt、子问题查询、元数据过滤
"""
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 准备带丰富元数据的文档
documents = [
    Document(
        text="Python 3.12 引入了新的 f-string 语法，支持在表达式内使用双引号。"
             "性能方面，Python 3.12 比 3.11 提升了约 5%。",
        metadata={"version": "3.12", "year": 2024, "topic": "python"},
    ),
    Document(
        text="Python 3.11 的主要改进是错误消息更加友好，能够精确定位错误位置。"
             "此外还引入了 ExceptionGroup 和 except* 语法。",
        metadata={"version": "3.11", "year": 2023, "topic": "python"},
    ),
    Document(
        text="FastAPI 使用 Pydantic v2 进行数据校验，支持异步路由处理。"
             "它的性能与 Node.js 和 Go 框架相当。",
        metadata={"version": "0.100", "year": 2024, "topic": "fastapi"},
    ),
    Document(
        text="Docker Compose v2 使用 'docker compose' 命令（无连字符）。"
             "它支持 GPU 设备映射和服务配置文件。",
        metadata={"version": "v2", "year": 2023, "topic": "docker"},
    ),
]

index = VectorStoreIndex.from_documents(documents)


# ====== 1. 元数据过滤 ======
# 只在与 year=2024 的文档中搜索
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter

query_engine = index.as_query_engine(
    similarity_top_k=3,
    filters=MetadataFilters(
        filters=[
            MetadataFilter(key="year", value=2024),
        ],
        condition="and",  # 所有条件都满足
    ),
)

result = query_engine.query("有哪些新特性？")
print(f"元数据过滤 (year=2024): {result}")


# ====== 2. 自定义 Prompt 模板 ======
from llama_index.core.prompts import PromptTemplate

# 自定义 QA Prompt
qa_prompt = PromptTemplate(
    "你是一位技术文档助手。根据以下信息回答问题。\n"
    "参考资料:\n"
    "{context_str}\n"
    "问题: {query_str}\n"
    "请用中文回答，以'根据文档记载，'开头。"
    "如果文档中没有相关信息，说'文档中未找到相关信息'。\n"
    "回答: "
)

query_engine = index.as_query_engine(
    text_qa_template=qa_prompt,
    similarity_top_k=3,
)

result = query_engine.query("Python 3.12 有什么新特性？")
print(f"\n自定义 Prompt: {result}")


# ====== 3. 子问题查询（SubQuestionQueryEngine） ======
# 将复杂问题拆解为多个子问题，分别查询后汇总
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

# 为不同 topic 创建不同的查询引擎
python_index = VectorStoreIndex.from_documents(
    [d for d in documents if d.metadata["topic"] == "python"]
)
fastapi_index = VectorStoreIndex.from_documents(
    [d for d in documents if d.metadata["topic"] == "fastapi"]
)
docker_index = VectorStoreIndex.from_documents(
    [d for d in documents if d.metadata["topic"] == "docker"]
)

# 为每个索引创建查询工具
python_tool = QueryEngineTool(
    query_engine=python_index.as_query_engine(),
    metadata=ToolMetadata(
        name="python_docs",
        description="Python 编程语言的版本更新和新特性文档",
    ),
)
fastapi_tool = QueryEngineTool(
    query_engine=fastapi_index.as_query_engine(),
    metadata=ToolMetadata(
        name="fastapi_docs",
        description="FastAPI Web 框架文档",
    ),
)
docker_tool = QueryEngineTool(
    query_engine=docker_index.as_query_engine(),
    metadata=ToolMetadata(
        name="docker_docs",
        description="Docker 容器技术的文档",
    ),
)

# 创建子问题查询引擎
sq_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[python_tool, fastapi_tool, docker_tool],
    llm=Settings.llm,
)

# 这个引擎会自动把问题拆解为子问题，分别查不同索引，最后汇总
result = sq_engine.query("Python 3.12 和 FastAPI 最近有什么更新？")
print(f"\n子问题查询: {result}")
```

### 7.3.5 Chat Engine —— 带记忆的对话式查询

```python
"""
ex_7_16_llamaindex_chat_engine.py: LlamaIndex Chat Engine
演示：带记忆的多轮对话式查询
"""
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 准备文档
documents = [
    Document(text="公司2025年第一季度的收入为 1500 万元，同比增长 20%。"),
    Document(text="公司2025年第二季度的收入为 1800 万元，环比增长 20%。"),
    Document(text="公司员工总数为 500 人，其中研发人员占 60%。"),
    Document(text="公司总部位于北京，在上海、深圳设有分公司。"),
]

index = VectorStoreIndex.from_documents(documents)

# ====== 1. 创建 Chat Engine（有记忆的对话） ======
# 三种模式:
#   "condense_question": 将对话历史 + 新问题浓缩为一个查询（默认）
#   "context": 直接用对话历史+检索内容构建上下文
#   "react": Agent 模式，可使用工具

chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    memory=ChatMemoryBuffer.from_defaults(token_limit=3000),  # 保留最近 3000 token 的对话
    verbose=True,  # 打印内部处理过程
)

# ====== 2. 多轮对话 ======
print("=== 第1轮 ===")
response = chat_engine.chat("公司第一季度的收入是多少？")
print(f"回答: {response}")

print("\n=== 第2轮 ===")
# 使用指代（"它"、"那个"），依赖上下文
response = chat_engine.chat("第二季度相比有什么变化？")
print(f"回答: {response}")

print("\n=== 第3轮 ===")
response = chat_engine.chat("研发人员有多少人？")
print(f"回答: {response}")

print("\n=== 第4轮 ===")
# 结合多轮信息
response = chat_engine.chat("总结一下公司的整体情况。")
print(f"回答: {response}")

# ====== 3. 重置对话 ======
chat_engine.reset()
print("\n=== 重置后 ===")
response = chat_engine.chat("我刚才问了什么？")
print(f"回答: {response}")


# ====== 4. 流式 Chat Engine ======
print("\n=== 流式对话 ===")
streaming_chat = index.as_chat_engine(chat_mode="condense_question", streaming=True)
response = streaming_chat.stream_chat("简单介绍一下公司")
for token in response.response_gen:
    print(token, end="", flush=True)
print()
```

---

## 7.4 LangChain vs LlamaIndex 选型指南

### 7.4.1 对比总表

| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| **核心定位** | 通用 LLM 应用框架 | 数据索引与检索框架 |
| **优势** | Agent、工具调用、多步链编排 | 文档索引、多索引类型、元数据管理 |
| **检索能力** | 基础的向量检索 | 丰富的索引类型 + 检索策略 |
| **Agent** | 成熟的 Agent 框架 + 工具生态 | 基础的 Agent 支持 |
| **Prompt 管理** | Prompt Template + Hub | Query Prompt 自定义 |
| **流式输出** | 原生 stream() + astream() | streaming=True 参数 |
| **学习曲线** | 较陡（概念多、API 变化快） | 较平缓（接口简洁） |
| **最佳场景** | 复杂 Agent 工作流、多工具编排 | 知识库问答、文档分析 |

### 7.4.2 选型决策框架 —— 何时用框架、何时用 SDK、何时组合

做框架选型不是二选一的选择题，而是一个分层决策：

**第一层：需要框架吗？** 如果你的应用只是简单的"用户提问 → LLM 回复"（带或不带一个固定的 system prompt），直接用 OpenAI SDK 更简洁。引入 LangChain 或 LlamaIndex 的收益从以下特征开始显著：(a) 需要从多个数据源检索并组合结果；(b) 需要 LLM 调用工具或 API；(c) 需要管理对话历史和记忆；(d) 需要流式输出的中间处理或结构化解析。

**第二层：主要开销在"检索"还是"编排"？** 如果你的核心痛点是把大量文档组织成一个可查询的知识库，LlamaIndex 是最快路径——`from_documents` 一行完成索引，`as_query_engine` 一行完成查询。如果你的核心痛点是让 LLM 自主判断"先查知识库还是先调用计算器还是先发邮件"，LangChain Agent 是更成熟的选择。

**第三层：是否两者结合？** 大多数生产级应用会同时需要"好的检索"和"好的编排"。此时的最佳架构是：用 LlamaIndex 做索引和检索层，包装为 LangChain Tool；用 LangChain Agent 做编排层，调用这些 Tool 和其他工具。下面的混合使用示例展示了这种架构。

**用 LangChain 的场景**:
- 需要 LLM + 多工具调用（搜索、计算、API 调用等）
- 需要复杂的多步链（先A后B再C，根据结果决定分支）
- 需要自定义 Agent 行为（记忆、规划、反思等）
- 项目已有 LangChain 生态（LangSmith 监控、LangServe 部署等）

**用 LlamaIndex 的场景**:
- 核心需求是"对我的文档集提问"
- 需要多种索引策略（向量检索 + 关键词检索 + 树形总结）
- 需要丰富的元数据过滤和管理
- 需要快速构建可工作的 RAG 原型

**两者结合的场景（推荐）**:
- 用 **LlamaIndex** 构建索引和检索层（它在这块更专业）
- 用 **LangChain** 构建 Agent 和应用编排层（它在这块更灵活）
- 两者可以通过 `LlamaIndexTool` 互操作

### 7.4.3 混合使用示例 —— LlamaIndex 做索引，LangChain 做编排

```python
"""
ex_7_17_hybrid.py: LangChain + LlamaIndex 混合使用
演示：LlamaIndex 负责索引检索，LangChain 负责 Agent 编排
"""
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding as LlamaEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import datetime

# ====== 1. LlamaIndex: 构建知识库索引 ======
Settings.llm = LlamaOpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = LlamaEmbedding(model="text-embedding-3-small")

# 产品知识库
product_docs = [
    Document(text="ProductX 是一款企业级项目管理工具，支持甘特图、看板和敏捷看板。价格从 $29/月 起。"),
    Document(text="ProductX 的退款政策: 购买后 30 天内可无理由全额退款。年付用户享受 20% 折扣。"),
    Document(text="ProductX 支持集成 Slack、Jira、GitHub、GitLab 等第三方工具。API 文档在 docs.productx.com。"),
]

# 技术文档
tech_docs = [
    Document(text="ProductX 的 API 使用 RESTful 风格，认证方式为 Bearer Token。API Base URL: https://api.productx.com/v1"),
    Document(text="创建项目: POST /projects, 获取项目列表: GET /projects, 删除项目: DELETE /projects/{id}"),
]

# 分别建索引
product_index = VectorStoreIndex.from_documents(product_docs)
tech_index = VectorStoreIndex.from_documents(tech_docs)

# 包装为 LlamaIndex QueryEngineTool
product_tool = QueryEngineTool.from_defaults(
    query_engine=product_index.as_query_engine(),
    name="product_knowledge",
    description="搜索产品信息、功能、价格、退款政策等。",
)

tech_tool = QueryEngineTool.from_defaults(
    query_engine=tech_index.as_query_engine(),
    name="technical_docs",
    description="搜索 API 文档、技术集成细节。",
)

# ====== 2. LangChain: 构建 Agent ======
# 将 LlamaIndex 工具转换为 LangChain 工具
@tool
def query_product_knowledge(query: str) -> str:
    """搜索产品知识库：产品信息、功能、价格、退款政策。"""
    return str(product_tool.query_engine.query(query))

@tool
def query_technical_docs(query: str) -> str:
    """搜索技术文档：API 使用方法、集成方案。"""
    return str(tech_tool.query_engine.query(query))

@tool
def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """获取当前时间。"""
    return datetime.datetime.now().strftime(format)

# 创建 Agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tools = [query_product_knowledge, query_technical_docs, get_current_time]

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是 ProductX 的智能客服。用产品知识库和技术文档回答问题。"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
    ("human", "{input}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 测试
result = executor.invoke({
    "input": "ProductX 多少钱？支持退款吗？另外我可以用 API 创建项目吗？"
})
print(f"\n最终回复: {result['output']}")
```

---

## 基础练习

### 练习 1: 简单 RAG 问答系统

用 LangChain + LCEL 构建一个问答系统：加载本地 TXT 文件 to 切分 to 向量化 to 检索 to 生成答案。

**练习文件**: `exercise/ai-application/ch07_langchain_llamaindex/ex_simple_rag.py`

### 练习 2: LlamaIndex 文档索引查询

用 LlamaIndex 加载一个目录中的 Markdown 文档建索引，实现关键词+语义的双路检索，并支持按文档来源过滤。

**练习文件**: `exercise/ai-application/ch07_langchain_llamaindex/ex_doc_index.py`

### 练习 3: 带工具的 Agent

使用 LangChain 创建一个 Agent，配备计算器、日期查询、单位换算三个工具。测试多步推理问题。

**练习文件**: `exercise/ai-application/ch07_langchain_llamaindex/ex_agent_tools.py`

---

## 进阶练习

### 练习 4: 多源 RAG（文档+网页+数据库）

构建一个从多个数据源（本地文件、指定 URL 网页、SQLite 数据库）检索的 RAG 系统，用 LangChain 的 EnsembleRetriever 合并不同检索器的结果。

**练习文件**: `exercise/ai-application/ch07_langchain_llamaindex/ex_multi_source_rag.py`

### 练习 5: Agentic RAG（Agent 驱动的 RAG）

结合 LangChain Agent + LlamaIndex 索引，实现一个"智能研究助手"：用户给出一个研究主题 to Agent 自动分解子问题 to 查知识库 to 查外部搜索 to 汇总生成研究报告。

**练习文件**: `exercise/ai-application/ch07_langchain_llamaindex/ex_research_assistant.py`

---

## 常见错误

### 错误 1: 混淆 LangChain 和 OpenAI 原生的消息格式

```python
# 错误: 直接传 dict 给 ChatOpenAI
llm.invoke([{"role": "user", "content": "Hello"}])
# to AttributeError

# 正确: 使用 LangChain 的消息对象
from langchain_core.messages import HumanMessage
llm.invoke([HumanMessage(content="Hello")])
```

### 错误 2: LCEL 中字典 key 不匹配

```python
# 错误: Prompt 模板要求 {topic}，但 invoke 时传入 {subject}
prompt = ChatPromptTemplate.from_template("讲讲{topic}")
chain = prompt | llm
chain.invoke({"subject": "Python"})  # to KeyError: 'topic'

# 正确: 保持 key 一致
chain.invoke({"topic": "Python"})
```

### 错误 3: 向量库重复插入数据

```python
# 错误: 每次启动都运行 from_documents，导致重复数据
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./db")
# 第2次运行时，数据库里有两份相同数据 to 检索结果是两份重复的

# 正确: 检查是否已有数据，有则加载，无则创建
import os
if os.path.exists("./db") and os.listdir("./db"):
    vectorstore = Chroma(persist_directory="./db", embedding_function=embeddings)
else:
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./db")
```

### 错误 4: Agent 陷入无限循环

```python
# 错误: 工具返回的结果不符合 LLM 预期，导致 Agent 反复调用同一个工具
# 症状: verbose=True 时看到不断重复 Thought to Action to Observation to Thought...

# 解决:
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,    # 限制最大迭代次数
    max_execution_time=30, # 限制最大执行时间
    return_intermediate_steps=True,  # 返回中间步骤便于调试
    handle_parsing_errors=True,      # 处理 LLM 输出格式错误
)
```

### 错误 5: LlamaIndex 的 Document 与 LangChain 的 Document 混淆

```python
# 错误: 用 LangChain 的 Document 创建 LlamaIndex 的索引
from langchain_core.documents import Document as LCDocument
from llama_index.core import VectorStoreIndex
docs = [LCDocument(page_content="...")]
index = VectorStoreIndex.from_documents(docs)  # to 类型错误

# 正确: 使用对应框架的 Document
from llama_index.core import Document as LIDocument
docs = [LIDocument(text="...")]
index = VectorStoreIndex.from_documents(docs)
```

### 错误 6: 忘记设置 LLM 和 Embedding

```python
# 错误: LlamaIndex 默认使用 OpenAI 的模型，但如果没有 API Key 会报错
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_documents(docs)
# to OpenAI API key not found

# 正确: 在代码开始处设置
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
```

---

## 本章小结

本章系统性地介绍了 LLM 应用开发的两个核心框架：

1. **LangChain** 是一套通用的 LLM 编排工具，核心理念是"一切皆 Runnable，一切皆可管道连接"。四大核心模块包括：
   - **Model I/O**: Prompt Template、Chat Model、Output Parser，处理模型输入输出的标准化
   - **Retrieval**: Document Loader to Text Splitter to Embedding to Vector Store to Retriever 五层架构
   - **Chains**: 将多个步骤串联成流水线，LCEL 管道语法实现声明式编排
   - **Agents**: 让 LLM 自主选择和使用工具，实现复杂任务的自动规划与执行（Thought→Action→Observation 循环）

2. **LCEL** 是 LangChain 的现代 API，通过 `|` 管道符和 `Runnable` 接口提供：
   - 统一的 `invoke()/batch()/stream()` 调用模式
   - `RunnableParallel` 并行执行
   - `RunnablePassthrough` 透传和 `RunnableLambda` 自定义函数注入

3. **LlamaIndex** 专注于数据索引与检索，设计哲学是"数据优先"，提供：
   - 多种索引类型（向量、摘要、树形、关键词表）
   - 丰富的查询引擎和 Chat Engine
   - 子问题拆解和多索引路由

4. **选型建议**（三层决策框架）:
   - 第一层：应用复杂度低（单一问答）→ 直接用 OpenAI SDK
   - 第二层：主要开销在"检索" → LlamaIndex；主要开销在"编排" → LangChain
   - 第三层：两者结合（LlamaIndex 做索引层，LangChain 做编排层）是大多数生产级应用的最佳架构

---

**下一章**: 第08章 AI 应用 Web 化——将 AI 能力部署为 Web 服务。
