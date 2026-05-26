"""
进阶练习: Prompt 安全评估框架
============================

需求:
  构建一个 Prompt 安全测试框架，能够:
  - 自动生成各类 Prompt Injection 测试用例
  - 对被测试的 Prompt 模板进行批量测试
  - 生成安全评估报告

要求:
  1. 实现 TestCaseGenerator 类:
     - generate_injection_cases() -> list[dict]:
       生成指令覆盖类攻击用例
     - generate_data_leak_cases() -> list[dict]:
       生成数据泄露类攻击用例（试图让模型输出系统提示/内部信息）
     - generate_role_play_cases() -> list[dict]:
       生成角色扮演类攻击用例
     - generate_indirect_cases() -> list[dict]:
       生成间接注入类攻击用例（在文档内容中嵌入指令）
     - 每种至少 5 个用例

  2. 实现 SecurityTester 类:
     - test_prompt(prompt_template, test_cases) -> list[dict]:
       对给定的 prompt 模板，用每个测试用例填充后调用 LLM，
       判断是否被攻破。
     - check_breach(response, case_type) -> bool:
       判断 LLM 的回复是否被攻破。
       判断标准:
       - 指令覆盖: 回复中出现了"攻击成功"的关键词
       - 数据泄露: 回复中包含了类似系统提示/API Key 的信息
       - 角色扮演: 回复中 LLM 承认了新角色
       - 间接注入: 回复中执行了嵌入的指令

  3. 实现 ReportGenerator 类:
     - generate_report(results) -> str:
       生成 Markdown 格式的安全评估报告，包含:
       - 测试概览（总用例数、攻破数、防御成功率）
       - 各类攻击的详细结果表格
       - 被攻破的用例详情
       - 安全评分（0-100）
       - 修复建议

  4. 实现 main():
     - 参数: --prompt（被测试的系统提示模板）、--output（报告路径）
     - 运行所有测试用例
     - 生成安全报告并保存

TODO:
  - [ ] 实现 TestCaseGenerator 类（4 种攻击类型的测试用例生成）
  - [ ] 实现 SecurityTester 类（测试 + 攻破判定）
  - [ ] 实现 ReportGenerator 类（Markdown 报告生成）
  - [ ] 实现 main() 串联流程

提示:
  - 测试用例结构: {"type": "injection", "input": "攻击文本", "expected_breach": True}
  - 攻破判定不一定 100% 准确，可记录置信度
  - 安全评分: (通过用例数 / 总用例数) * 100
  - Markdown 报告用表格展示结果（| 类型 | 输入 | 输出 | 攻破 |）
  - 对于系统提示模板，用 {user_input} 作为占位符
  - 调用 LLM 时，temperature 设为 0（提高测试可复现性）
"""
import os
import json
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class TestCaseGenerator:
    """Prompt Injection 测试用例生成器"""

    @staticmethod
    def generate_injection_cases() -> list[dict]:
        """指令覆盖类攻击"""
        # TODO: 返回至少 5 个用例
        pass

    @staticmethod
    def generate_data_leak_cases() -> list[dict]:
        """数据泄露类攻击"""
        # TODO
        pass

    @staticmethod
    def generate_role_play_cases() -> list[dict]:
        """角色扮演类攻击"""
        # TODO
        pass

    @staticmethod
    def generate_indirect_cases() -> list[dict]:
        """间接注入类攻击"""
        # TODO
        pass


class SecurityTester:
    """安全测试执行器"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.results = []

    def test_prompt(self, system_prompt_template: str, test_cases: list[dict]) -> list[dict]:
        # TODO: 对每个测试用例执行测试
        pass

    def check_breach(self, response: str, case_type: str) -> tuple[bool, float, str]:
        # TODO: 判断是否被攻破，返回 (是否攻破, 置信度, 原因)
        pass


class ReportGenerator:
    """安全报告生成器"""

    @staticmethod
    def generate_report(results: list[dict]) -> str:
        # TODO: 生成 Markdown 安全评估报告
        pass


def main():
    # TODO: 解析参数 → 生成测试用例 → 执行测试 → 生成报告
    pass


if __name__ == "__main__":
    main()
