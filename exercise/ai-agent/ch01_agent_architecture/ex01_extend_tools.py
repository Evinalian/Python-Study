"""
练习 1-1: 扩展工具集

为第 1 章中的最简 Agent 添加一个新的工具：translate（翻译工具）。

要求：
1. 工具接受两个参数：text（待翻译文本）和 target_language（目标语言）
2. Schema 描述清晰，让 LLM 知道什么情况下用翻译工具
3. 翻译功能可以用一个简单的字典映射来模拟（中英互译即可，至少支持 5 种语言）
4. 将新工具注册到 TOOLS_SCHEMA 和 TOOL_FUNCTIONS 中
5. 测试：让 Agent 翻译 "Hello, how are you?" 到中文

TODO:
1. 在 TOOLS_SCHEMA 列表中添加 translate 工具的 JSON Schema
   - name: "translate"
   - description 要写清楚何时使用、何时不使用
   - parameters 包含 text (string) 和 target_language (string, 用 enum 限定支持的语言)
2. 实现 translate 函数
   - 创建一个中英日韩法互译的字典映射（至少 20 个词条）
   - 对于字典中没有的词，返回原文并说明"部分词汇未找到翻译"
   - 返回格式: {"success": True/False, "original": ..., "translated": ..., "target_language": ...}
3. 将 translate 函数添加到 TOOL_FUNCTIONS 字典
4. 运行 Agent 测试翻译功能
"""


# 请在此处编写你的代码
# 可以参考第 1 章中的完整代码结构


if __name__ == "__main__":
    # TODO: 测试你的 translate 工具
    # 1. 测试直接翻译: "Hello, how are you?" → 中文
    # 2. 测试 Agent 自主判断何时使用翻译工具
    # 3. 测试翻译不存在的词时的错误处理
    pass
