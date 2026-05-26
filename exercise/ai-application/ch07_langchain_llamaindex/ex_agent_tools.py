"""
练习: 带工具的 LangChain Agent
=============================

需求:
  创建一个 LangChain Agent，配备计算器、日期时间查询、单位换算三个工具，
  测试 Agent 在需要多步推理和工具组合时自主决策的能力。

要求:
  1. 实现工具定义（使用 @tool 装饰器）:
     - calculator(expression: str) -> str:
       计算数学表达式。使用安全的 eval（限制可用函数）。
       支持: +, -, *, /, %, **, sqrt, sin, cos, log, abs, round
     - get_datetime_info(query: str) -> str:
       处理日期时间相关查询。支持:
       - "now" / "当前时间" → 返回完整时间
       - "today" / "今天日期" → 返回日期
       - "weekday" / "星期几" → 返回星期
       - "timestamp" / "时间戳" → 返回 Unix 时间戳
     - unit_converter(value: float, from_unit: str, to_unit: str) -> str:
       单位换算。至少支持:
       - 长度: km, m, cm, mm, mile, yard, foot, inch
       - 重量: kg, g, mg, lb, oz
       - 温度: C, F, K (需要公式转换)
       - 面积: sqm, sqft, acre

  2. 实现 create_agent(tools):
     - 使用 create_tool_calling_agent + AgentExecutor
     - 设置 max_iterations=15
     - 设置详细的系统提示，告知 Agent 如何使用这些工具
     - 使用 ChatPromptTemplate + MessagesPlaceholder("agent_scratchpad")

  3. 实现 test_agent(executor, test_cases):
     - 测试用例至少 5 个，覆盖:
       a. 单个工具调用（如"算一下 2的10次方"）
       b. 两步推理（如"100英里是多少米？" → 需要先换算 mile→m 再计算结果）
       c. 多工具组合（如"我现在体重70kg，如果减掉15磅，还剩多少kg？"）
       d. 含日期的问题（如"今天是星期几？距离下周一还有几天？"）
       e. 复杂推理（如"一个长方形房间长5米宽4米，面积是多少平方英尺？"）

  4. 实现 main():
     - 构建 Agent
     - 运行测试用例并打印每个测试的: 问题、思考过程、最终答案
     - 统计成功/失败/工具调用次数

TODO:
  - [ ] 实现 calculator(expression) 工具
  - [ ] 实现 get_datetime_info(query) 工具
  - [ ] 实现 unit_converter(value, from_unit, to_unit) 工具
  - [ ] 实现 create_agent(tools) 创建 AgentExecutor
  - [ ] 实现 run_tests(executor, test_cases) 执行测试并记录结果
  - [ ] 实现 main()

提示:
  - 安全的 eval 应使用受限的 globals 字典
  - 温度换算: C→F: F=C*9/5+32, F→C: C=(F-32)*5/9, K=C+273.15
  - 单位换算可以用字典定义换算比率，统一换算到基准单位再换算到目标单位
  - AgentExecutor 设置 verbose=True 可以看到思考过程
  - 统计工具调用次数可以通过 return_intermediate_steps=True 来实现
"""
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

load_dotenv()


@tool
def calculator(expression: str) -> str:
    """安全的数学表达式计算器。
    支持基本运算(+, -, *, /, **, %)和数学函数(sqrt, sin, cos, log, abs, round, pi, e)。
    参数: expression - 数学表达式，如 "2**10" 或 "sqrt(144)" """
    # TODO: 实现安全的 eval 计算
    pass


@tool
def get_datetime_info(query: str) -> str:
    """获取日期时间相关信息。
    参数 query 可以是: "now"(当前完整时间), "today"(日期), "weekday"(星期几), "timestamp"(时间戳)。
    示例: get_datetime_info("now") 返回类似 '2025-01-15 14:30:00' """
    # TODO: 实现日期时间查询
    pass


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算工具。
    支持长度(km,m,cm,mm,mile,yard,foot,inch)、
    重量(kg,g,mg,lb,oz)、
    温度(C,F,K)、
    面积(sqm,sqft,acre)。
    参数: value=数值, from_unit=源单位, to_unit=目标单位。
    示例: unit_converter(100, "mile", "km") """
    # TODO: 实现单位换算
    pass


def create_agent(tools: list):
    # TODO: 创建 Agent + AgentExecutor
    pass


def run_tests(executor: AgentExecutor, test_cases: list[dict]) -> list[dict]:
    # TODO: 运行测试用例，收集结果
    pass


def main():
    # TODO: 组装 tools、创建 agent、运行测试、打印统计
    pass


if __name__ == "__main__":
    main()
