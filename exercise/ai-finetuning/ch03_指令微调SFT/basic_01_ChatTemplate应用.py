"""
练习 1: Chat Template 对比 —— 理解不同模型的模板差异

场景:
    不同模型使用不同的特殊 token 和格式。如果训练和推理使用了不一致的模板，
    模型行为会非常混乱。你需要深入理解几种主流模型的 Chat Template 差异。

要求:
    1. 加载三种不同模型的 tokenizer（LLaMA-3, Qwen2.5, ChatGLM-4）
    2. 用同一组 messages 在三者上调用 apply_chat_template
    3. 对比输出的字符串格式
    4. 对比 tokenize 后的 token ID

TODO:
    1. 实现 load_tokenizers() 函数:
       - 加载至少两种不同模型的 tokenizer
       - 返回 {model_name: tokenizer} 字典

    2. 实现 compare_templates(tokenizers, messages_sets) 函数:
       - messages_sets 包含多组不同的 messages:
         * 简单单轮问答（system + user + assistant）
         * 多轮对话（system + user + assistant + user + assistant）
         * 无 system prompt 的单轮问答
         * 只有 user 的输入（模拟推理时的输入）
       - 对每组 messages，对比各 tokenizer 的输出
       - 打印差异分析

    3. 实现 analyze_special_tokens(tokenizers) 函数:
       - 打印每个 tokenizer 的特殊 token 及其 ID:
         * bos_token, eos_token, pad_token, unk_token
         * 模型的 Chat Template 中使用的特殊 token（如 <|im_start|>, [INST] 等）
       - 分析哪些 token 在不同模型间含义相同但 ID 不同

    4. 实现 detect_template_mismatch(training_template, inference_template) 函数:
       - 输入训练时和推理时使用的模板文本
       - 对比差异并报告:
         * 特殊 token 是否一致
         * 空格和换行是否一致
         * 角色标记是否一致

    5. 思考题（注释回答）:
       - 如果训练时使用了 Qwen 的模板但推理时手动拼接了 LLaMA 格式，会发生什么？
       - 为什么不能用简单的 f-string 来替代 apply_chat_template？
"""
from transformers import AutoTokenizer


# ============================================================
# TODO 1: 加载 tokenizers
# ============================================================
def load_tokenizers() -> dict:
    """
    加载多种模型的 tokenizer。

    返回:
        {"Qwen2.5-0.5B": tokenizer, "LLaMA-3.2-1B": tokenizer, ...}
    """
    # TODO: 定义模型列表（至少两个不同的架构）
    # TODO: 对每个模型加载 tokenizer
    # TODO: 返回 {model_name: tokenizer} 字典
    pass


# ============================================================
# TODO 2: 对比 Chat Templates
# ============================================================
def compare_templates(tokenizers: dict, messages_sets: dict):
    """
    对比不同 tokenizer 对同一组 messages 的模板应用结果。

    messages_sets 格式:
    {
        "单轮问答": [{"role": "system", "content": "..."}, ...],
        "多轮对话": [...],
        "无system": [...],
        "仅user(推理)": [...],
    }

    对每组 messages:
    1. 用每个 tokenizer 的 apply_chat_template 生成字符串
    2. 打印字符串（标注特殊 token）
    3. Tokenize 并打印 token 数量
    4. 分析差异
    """
    # TODO: 遍历 messages_sets
    # TODO: 对每组 messages，遍历 tokenizers
    # TODO: 调用 tokenizer.apply_chat_template(messages, tokenize=False)
    # TODO: 打印对比输出（用分隔线标注不同 tokenizer）
    # TODO: 调用 tokenizer(messages, tokenize=True) 获取 token IDs
    # TODO: 比较 token 数量差异
    pass


# ============================================================
# TODO 3: 分析特殊 token
# ============================================================
def analyze_special_tokens(tokenizers: dict):
    """
    分析每个 tokenizer 的特殊 token 配置。

    打印:
    - bos_token / bos_token_id
    - eos_token / eos_token_id
    - pad_token / pad_token_id
    - unk_token / unk_token_id
    - Chat Template 字符串（如果可用）
    - 所有 additional_special_tokens
    """
    # TODO: 遍历 tokenizers
    # TODO: 打印每个 tokenizer 的特殊 token 信息
    # TODO: 尝试获取 chat_template 属性（tokenizer.chat_template）
    # TODO: 解析 chat_template 中使用了哪些特殊 token
    pass


# ============================================================
# TODO 4: 检测模板不匹配
# ============================================================
def detect_template_mismatch(training_template: str, inference_template: str) -> list[str]:
    """
    检测训练和推理模板之间的差异。

    返回: 差异描述列表
    """
    issues = []
    # TODO: 比较两个字符串的差异
    # TODO: 检查特殊 token 是否一致
    #   提示: 搜索 <|im_start|>, <|im_end|>, [INST], [/INST], <s>, </s> 等
    # TODO: 检查空格模式是否一致
    # TODO: 检查角色标记顺序是否一致
    return issues


# ============================================================
# TODO 5: 思考题
# ============================================================
"""
Q1: 如果训练时使用了 Qwen 的模板但推理时手动拼接了 LLaMA 格式，会发生什么？
A1: TODO
    提示: 思考特殊 token 的含义——模型在训练时学到 <|im_start|> 表示角色切换，
    但推理时看到的是 [INST]——模型会如何"解读"这个从未见过的 token？

Q2: 为什么不能用简单的 f-string 来替代 apply_chat_template？
A2: TODO
    提示: 除了格式字符串，apply_chat_template 还做了什么？
    比如 add_generation_prompt 参数的处理，以及不同模型的 system prompt 位置差异。

Q3: 如果你拿到的开源微调数据集没有标注是用什么 Chat Template 生成的，如何判断？
A3: TODO
"""


if __name__ == "__main__":
    print("=" * 50)
    print("  Chat Template 对比分析")
    print("=" * 50)

    # 示例 messages
    test_messages_sets = {
        "单轮问答（含system）": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "什么是机器学习？"},
            {"role": "assistant", "content": "机器学习是人工智能的一个分支..."},
        ],
        "多轮对话": [
            {"role": "system", "content": "你是一个编程助手。"},
            {"role": "user", "content": "如何读取文件？"},
            {"role": "assistant", "content": "使用 open() 函数..."},
            {"role": "user", "content": "那二进制文件呢？"},
            {"role": "assistant", "content": "使用 'rb' 模式..."},
        ],
        "仅user（推理模式）": [
            {"role": "system", "content": "你是一个有帮助的AI助手。"},
            {"role": "user", "content": "介绍一下深度学习"},
            # 注意: 推理时没有 assistant 回复，需要设置 add_generation_prompt=True
        ],
    }

    # TODO: 取消注释完成实验
    # tokenizers = load_tokenizers()
    # analyze_special_tokens(tokenizers)
    # compare_templates(tokenizers, test_messages_sets)

    print("\n请完成以上 TODO 后取消注释运行。")
