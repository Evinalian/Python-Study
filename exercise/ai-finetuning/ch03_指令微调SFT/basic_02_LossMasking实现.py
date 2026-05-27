"""
练习 2: Loss Masking 手动实现与验证

场景:
    你已经理解了 Loss Masking 的原理——只对 assistant 回复计算 loss，
    忽略 system 和 user 部分。现在需要你手动实现这个逻辑并验证其正确性。

要求:
    1. 独立实现 tokenize_with_loss_mask 函数（不复制教程中的代码）
    2. 为单轮对话和多轮对话分别实现正确的 masking
    3. 编写测试用例验证实现的正确性

TODO:
    1. 实现 find_assistant_spans(messages, tokenizer) 函数:
       - messages 是标准的 messages 列表
       - 利用 apply_chat_template + add_generation_prompt 的方法
       - 找到每个 assistant 回复在完整序列中的 token 位置
       - 返回 [(start1, end1), (start2, end2), ...]

    2. 实现 create_loss_mask(input_ids, assistant_spans, ignore_index=-100):
       - input_ids 是完整序列的 token IDs
       - assistant_spans 是 assistant 的位置列表
       - 返回 labels 列表: assistant 位置保留原 token ID，其余为 ignore_index

    3. 实现 tokenize_with_loss_mask(messages, tokenizer, max_length=2048):
       - 组合上述两个函数
       - Tokenize 完整序列
       - 创建 loss mask
       - 处理截断（确保截断时 assistant 部分优先保留）

    4. 实现 validate_masking(messages, tokenizer, labels, input_ids) 函数:
       - 验证 system 和 user 部分的 labels 都为 -100
       - 验证 assistant 部分的 labels 都不是 -100
       - 验证 labels 和 input_ids 在 assistant 部分的值相同
       - 打印验证报告（通过/失败 + 详细统计）

    5. 准备至少 4 组测试 messages:
       - 单轮对话（system + user + assistant）
       - 多轮对话（system + 2轮 user/assistant）
       - 无 system 的单轮
       - 空 assistant 回复（边界情况）

    6. 对每组测试 messages 运行验证并打印结果

    7. 思考题（注释回答）:
       - 为什么 labels 中使用的 ignore_index 是 -100？
       - 如果截断发生在 assistant 回复中间怎么办？
"""
from transformers import AutoTokenizer


# ============================================================
# TODO 1: 定位 assistant 部分的 token 位置
# ============================================================
def find_assistant_spans(messages: list[dict], tokenizer) -> list[tuple[int, int]]:
    """
    找到 messages 中每个 assistant 回复在 tokenize 后的位置。

    策略:
    1. 构建 prompt（所有 system + user，加上 generation prompt）
    2. 构建完整文本（包含 assistant 回复）
    3. prompt 的 token 长度就是第一个 assistant 的起始位置
    4. 对于多轮对话，需要逐轮构建来定位每个 assistant

    返回: [(assist_start_1, assist_end_1), (assist_start_2, assist_end_2), ...]
    """
    # TODO: 实现多轮 assistant 位置定位
    #   提示: 对于多轮对话，可以逐轮处理——
    #   第 i 轮的 prompt 包含第 1 到 i 轮的全部内容 + 第 i+1 轮的 user
    #   第 i 轮的 assistant 就从 prompt_len_i 开始到 prompt_len_i+1 的 assistant 前结束
    pass


# ============================================================
# TODO 2: 创建 Loss Mask
# ============================================================
def create_loss_mask(
    input_ids: list[int],
    assistant_spans: list[tuple[int, int]],
    ignore_index: int = -100,
) -> list[int]:
    """
    创建带 masking 的 labels。

    参数:
        input_ids: 完整序列的 token ID 列表
        assistant_spans: assistant 部分的 (start, end) 列表
        ignore_index: mask 填充值

    返回:
        labels 列表，长度与 input_ids 相同
    """
    # TODO: 初始化 labels 列表，全部为 ignore_index
    # TODO: 对每个 assistant span，将 labels[start:end] 设为 input_ids[start:end]
    pass


# ============================================================
# TODO 3: 完整的 tokenize + loss mask 函数
# ============================================================
def tokenize_with_loss_mask(
    messages: list[dict],
    tokenizer,
    max_length: int = 2048,
    ignore_index: int = -100,
) -> dict:
    """
    将 messages 序列 tokenize 并创建 loss mask。

    返回: {"input_ids": [...], "attention_mask": [...], "labels": [...]}
    """
    # TODO: 应用 Chat Template 得到完整文本
    # TODO: Tokenize 完整文本
    # TODO: 调用 find_assistant_spans 定位 assistant 部分
    # TODO: 调用 create_loss_mask 创建 labels
    # TODO: 处理 max_length 截断（确保 assistant 部分优先保留）
    # TODO: 创建 attention_mask（全 1，因为截断后没有 padding）
    pass


# ============================================================
# TODO 4: 验证 Loss Mask 的正确性
# ============================================================
def validate_masking(
    messages: list[dict],
    tokenizer,
    labels: list[int],
    input_ids: list[int],
    ignore_index: int = -100,
) -> dict:
    """
    验证 loss mask 是否正确。

    检查:
    1. labels 长度 == input_ids 长度
    2. assistant 位置的 labels != ignore_index
    3. non-assistant 位置的 labels == ignore_index
    4. assistant 位置的 labels == input_ids

    返回:
        {
            "passed": bool,          # 全部检查通过
            "total_tokens": int,     # 总 token 数
            "assistant_tokens": int, # assistant 部分的 token 数
            "masked_tokens": int,    # 被 mask 的 token 数
            "mask_ratio": float,     # mask 比例
            "errors": [str, ...],    # 错误描述列表
        }
    """
    # TODO: 检查长度一致性
    # TODO: 统计 assistant 部分的 token 数
    # TODO: 验证 assistant 位置的 labels == input_ids
    # TODO: 验证非 assistant 位置的 labels == ignore_index
    # TODO: 收集所有错误
    # TODO: 返回验证结果
    pass


# ============================================================
# TODO 5: 准备测试用例
# ============================================================
def get_test_cases() -> list[dict]:
    """返回多组用于测试的 messages。"""
    test_cases = [
        # Case 1: 标准单轮对话
        {
            "name": "单轮对话（含system）",
            "messages": [
                {"role": "system", "content": "你是一个有帮助的AI助手。"},
                {"role": "user", "content": "Python是什么？"},
                {"role": "assistant", "content": "Python是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。"},
            ],
        },
        # TODO: Case 2: 多轮对话
        # TODO: Case 3: 无 system prompt
        # TODO: Case 4: 空 assistant 回复（边界情况）
        # TODO: Case 5: 超长 assistant 回复（测试截断行为）
    ]
    return test_cases


# ============================================================
# TODO 6: 运行测试
# ============================================================
def run_all_tests(tokenizer):
    """对所有测试用例运行验证。"""
    test_cases = get_test_cases()
    all_passed = True

    for case in test_cases:
        print(f"\n{'='*40}")
        print(f"测试: {case['name']}")
        print(f"{'='*40}")

        # TODO: 调用 tokenize_with_loss_mask
        # TODO: 调用 validate_masking
        # TODO: 打印验证结果

        # if not result["passed"]:
        #     all_passed = False

    return all_passed


# ============================================================
# TODO 7: 思考题
# ============================================================
"""
Q1: 为什么 labels 中使用的 ignore_index 是 -100？
A1: TODO
    提示: 查看 PyTorch 的 CrossEntropyLoss 文档中的 ignore_index 参数

Q2: 如果截断发生在 assistant 回复中间怎么办？
A2: TODO
    提示: 思考截断策略——从左边截断（保留最近的上下文）还是从右边截断？
    SFT 中通常应该保留 assistant 回复的完整性。

Q3: 多轮对话的第一个 assistant 回复如果不计算 loss 会怎样？
A3: TODO
"""


if __name__ == "__main__":
    print("=" * 50)
    print("  Loss Masking 测试套件")
    print("=" * 50)

    # 加载 tokenizer（使用小模型）
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TODO: 取消注释运行测试
    # all_passed = run_all_tests(tokenizer)
    # print(f"\n{'='*50}")
    # print(f"全部测试{'通过' if all_passed else '未通过'}")
    # print(f"{'='*50}")

    print("\n请完成以上 TODO 后取消注释运行。")
