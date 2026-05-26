"""
练习 3: System Prompt 角色设计

场景:
    设计一个"代码审查员"或"面试模拟官"的 System Prompt，
    要求覆盖角色设定的六个关键要素。

角色设定六要素:
    1. 身份定义: 你是谁，职责是什么
    2. 能力边界: 你可以做什么 + 你不可以做什么
    3. 知识范围: 专长领域、知识截止时间
    4. 输出风格: 语气、格式、长度要求
    5. 行为约束: 什么情况下做什么、不做什么
    6. 交互协议: 用户如何与你交互，你如何回应

TODO:
    1. 选择角色（代码审查员 或 面试模拟官）

    2. 用 build_system_prompt() 函数构建 System Prompt:
       - 实现一个函数接收各个要素作为参数
       - 拼装成结构化的 System Prompt 文本
       - 参考教程中的 SYSTEM_PROMPT_TEMPLATE 模板

    3. 实现 converse(user_message, conversation_history) 函数:
       - 维护对话历史（messages 列表）
       - 每次调用追加 user message 并获取 assistant 回复
       - 支持多轮对话

    4. 测试场景（4-5 轮对话）:
       - 场景A（代码审查员）: 提交一段有问题的代码 → 看审查意见 → 追问 → 提交修正版
       - 场景B（面试模拟官）: 开始面试 → 回答问题 → 收到追问 → 结束评价

    5. 思考: 你的 System Prompt 中哪个要素最重要？为什么？
"""

import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# TODO 1: 选择角色（二选一，把另一个注释掉）
# ============================================================
ROLE_CHOICE = "code_reviewer"  # 或 "interviewer"


# ============================================================
# TODO 2: 实现 build_system_prompt()
# ============================================================
def build_system_prompt(
    role_name: str,
    role_description: str,
    capabilities: list[str],
    limitations: list[str],
    expertise: str,
    tone: str,
    output_format: str,
    constraints: list[str],
    protocol: str,
) -> str:
    """
    根据六要素构建 System Prompt。

    参数:
        role_name: 角色名称（如"高级代码审查员"）
        role_description: 角色描述（一句话说明职责）
        capabilities: 能力列表 ["可以做的事1", "可以做的事2"]
        limitations: 限制列表 ["不能做的事1"]
        expertise: 专长领域说明
        tone: 输出语气（专业/友好/严格/幽默）
        output_format: 输出格式要求（如"Markdown + 代码块"）
        constraints: 行为约束列表
        protocol: 交互协议说明

    返回:
        完整的 system prompt 字符串
    """
    # TODO: 拼装模板。可参考以下结构：
    # # 身份定义
    # 你是{role_name}，{role_description}。
    # # 能力范围
    # 你可以: ...
    # 你不可以: ...
    # # 知识领域
    # 专长: {expertise}
    # # 输出规范
    # 语气: {tone} | 格式: {output_format}
    # # 行为约束
    # - ...
    # # 交互协议
    # {protocol}
    pass


# ============================================================
# TODO 2b: 定义角色的具体参数
# ============================================================
def get_role_params() -> dict:
    """
    返回 build_system_prompt() 所需的所有参数。

    代码审查员参考:
        role_name = "高级代码审查员"
        role_description = "审查代码质量、安全性和可维护性"
        capabilities = ["发现bug", "安全漏洞检测", "性能优化建议", "代码风格审查"]
        limitations = ["不执行代码", "不修改代码仓库", "不处理与代码无关的问题"]
        expertise = "Python, JavaScript, Java, SQL, 常见Web框架"
        tone = "专业但友好"
        output_format = "Markdown，问题按严重程度排列 (CRITICAL > WARNING > INFO)"
        constraints = ["必须给出修复前后的对比代码", "安全漏洞标为CRITICAL"]
        protocol = "用户发代码 → 逐行审查 → 分行标记问题 → 总结"

    面试模拟官参考:
        role_name = "高级技术面试官"
        role_description = "模拟真实技术面试，考察候选人的综合能力"
        capabilities = ["出算法题", "追问深挖", "评估回答质量", "给出改进建议"]
        limitations = ["不给直接答案", "不透露真实面试题", "不泄露公司信息"]
        expertise = "数据结构、算法、系统设计、项目经验深挖"
        tone = "专业但不压迫，鼓励思考"
        output_format = "自然对话，关键问题加粗"
        constraints = ["根据回答水平动态调整难度", "最终给出评分和改进建议"]
        protocol = "开场(1轮) → 基础(2轮) → 算法(2轮) → 反问(1轮) → 评价"
    """
    # TODO: 根据 ROLE_CHOICE 返回合适的参数
    pass


# ============================================================
# TODO 3: 实现对话管理
# ============================================================
class RoleAgent:
    """
    角色化对话 Agent。

    职责:
    - 维护对话历史
    - 封装 API 调用
    - 支持多轮对话
    """

    def __init__(self, system_prompt: str):
        """
        初始化 Agent。

        参数:
            system_prompt: 角色的 System Prompt
        """
        self.system_prompt = system_prompt
        self.history: list[dict] = []  # 对话历史
        # TODO: 将 system_prompt 作为第一条 system message 加入 history

    def chat(self, user_message: str) -> str:
        """
        发送一条用户消息，获取助手回复。

        参数:
            user_message: 用户输入

        返回:
            助手的回复文本
        """
        # TODO:
        # 1. 将 user_message 加入 history
        # 2. 调用 client.chat.completions.create()
        #    - model="gpt-4o"
        #    - messages=self.history（包含 system + 所有历史）
        #    - temperature=0.7（角色扮演需要一定创造性）
        # 3. 将 assistant 回复加入 history
        # 4. 返回回复文本
        pass

    def get_history(self) -> list[dict]:
        """返回完整对话历史"""
        return self.history.copy()

    def reset(self):
        """重置对话（保留 system prompt）"""
        # TODO: 只保留 system message，清除所有 user/assistant 消息
        pass


# ============================================================
# TODO 4: 测试场景
# ============================================================
def test_code_reviewer():
    """
    测试代码审查员角色。

    模拟场景:
    1. 提交一段有 SQL 注入漏洞的代码
    2. 看审查意见
    3. 追问某个具体问题
    4. 提交修正后的代码
    5. 看二次审查意见
    """
    # TODO:
    # 1. 构建 system_prompt
    # 2. 创建 RoleAgent
    # 3. 模拟多轮对话
    # 4. 打印每轮对话
    pass


def test_interviewer():
    """
    测试面试模拟官角色。

    模拟场景:
    1. 开始面试
    2. 回答第一个基础问题
    3. 回答第二个数据结构问题
    4. 回答追问
    5. 收到评价
    """
    # TODO: 类似 test_code_reviewer()
    pass


if __name__ == "__main__":
    if ROLE_CHOICE == "code_reviewer":
        test_code_reviewer()
    else:
        test_interviewer()

    # TODO 5: 思考题
    """
    Q: 你的 System Prompt 中哪个要素最重要？为什么？
    A: TODO
    """
