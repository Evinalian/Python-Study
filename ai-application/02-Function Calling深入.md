# 第02章 Function Calling 深入

## 学习目标

完成本章后，你将能够：

1. 理解 Function Calling 的完整调用链路：定义 → 触发 → 执行 → 反馈
2. 掌握 Tool Schema 的设计规范，写出高质量的 function description 和 parameters 定义
3. 理解单工具和多工具场景下的 tool_choice 策略
4. 处理并行工具调用，正确匹配 tool_call_id
5. 实现工具执行的错误处理和重试逻辑
6. 应用三种工具设计模式：查询类、操作类、验证类
7. 构建多工具协作的复杂 Agent

## 前置知识

- OpenAI SDK 基本用法（python-core 第23章水平）
- 理解 messages 的 role 体系：system / user / assistant / tool
- 了解 JSON Schema 的基本语法（type, properties, required, enum）
- 对 Python 的类型注解和异常处理有一定基础

---

## 1. Function Calling 基础回顾

### 1.1 什么是 Function Calling

Function Calling 不是模型真的"执行"了你的函数，而是一个**协商机制**：

1. 你告诉模型："我有一组工具，你可以调用它们"
2. 模型判断当前对话是否需要调用工具
3. 如果需要，模型输出一个函数调用请求（函数名 + 参数）
4. 你的代码执行这个函数
5. 你把执行结果返回给模型
6. 模型基于结果生成最终回复

```
User: "北京今天天气怎么样？"
  ↓
Model: (判断需要调用 get_weather) → 输出 tool_call: get_weather(city="北京")
  ↓
Your Code: 执行 get_weather(city="北京") → "晴，25°C"
  ↓
Model: (基于结果生成) → "北京今天晴朗，气温25°C"
```

### 1.2 最小可运行示例

```python
"""
Function Calling 最小可运行示例 —— 天气查询。
展示完整的调用链路。
"""
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# 步骤1: 定义真实的函数
# ============================================================
def get_weather(city: str, date: str = "今天") -> dict:
    """
    模拟天气查询。实际项目中应调用真实的天气 API。
    """
    # 模拟数据
    weather_data = {
        "北京": {"今天": "晴，25°C，湿度40%", "明天": "多云，22°C，湿度55%"},
        "上海": {"今天": "小雨，20°C，湿度80%", "明天": "阴，23°C，湿度70%"},
        "深圳": {"今天": "晴，30°C，湿度65%", "明天": "雷阵雨，28°C，湿度85%"},
    }
    return {
        "city": city,
        "date": date,
        "weather": weather_data.get(city, {}).get(date, "暂无数据"),
        "query_time": "2024-06-15T10:00:00",
    }


# ============================================================
# 步骤2: 定义 Tool Schema（告诉模型这个函数怎么用）
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市在指定日期的天气情况。返回天气描述、温度和湿度信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'、'上海'。支持中国主要城市。",
                    },
                    "date": {
                        "type": "string",
                        "description": "查询日期。可以是'今天'、'明天'或具体日期如'2024-06-15'。默认'今天'。",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

# 步骤3: 建立函数名到实际函数的映射
AVAILABLE_FUNCTIONS = {"get_weather": get_weather}

# ============================================================
# 步骤4: 完整的对话处理流程
# ============================================================
def run_conversation(user_query: str) -> str:
    """处理一次对话，自动判断是否需要调用工具"""

    # 初始化 messages
    messages = [
        {"role": "system", "content": "你是一个助手，可以查询天气信息。当用户询问天气时，请调用 get_weather 函数。"},
        {"role": "user", "content": user_query},
    ]

    # 第一次调用：模型决定是否调用工具
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",  # 让模型自己决定
        temperature=0.0,
    )

    # 获取模型的响应消息
    response_message = response.choices[0].message

    # 检查是否有 tool_calls
    if response_message.tool_calls:
        # --- 模型要求调用工具 ---
        # 把模型的响应（含 tool_calls）加入对话
        messages.append(response_message)

        # 逐个处理每个 tool_call
        for tool_call in response_message.tool_calls:
            # 解析函数名和参数
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"[Tool Call] 调用 {function_name}({function_args})")

            # 执行真实的函数
            if function_name in AVAILABLE_FUNCTIONS:
                function_result = AVAILABLE_FUNCTIONS[function_name](**function_args)
            else:
                function_result = {"error": f"未知函数: {function_name}"}

            # 将执行结果以 tool role 消息加入对话
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,  # 必须匹配！
                    "name": function_name,
                    "content": json.dumps(function_result, ensure_ascii=False),
                }
            )

        # 第二次调用：模型基于工具结果生成最终回答
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
        )
        return final_response.choices[0].message.content

    else:
        # --- 模型直接回答，无需调用工具 ---
        return response_message.content


if __name__ == "__main__":
    # 测试：需要调用工具的问题
    print("用户: 北京今天天气怎么样？")
    print("助手:", run_conversation("北京今天天气怎么样？"))
    print()

    # 测试：不需要调用工具的问题
    print("用户: 你好，你能做什么？")
    print("助手:", run_conversation("你好，你能做什么？"))
```

### 1.3 tool_call 的数据结构

```python
"""
tool_call 对象的内部结构详解。
"""
import json


def inspect_tool_call():
    """
    模拟一个 tool_call 对象的结构，帮助理解每个字段的含义。
    实际返回的是 OpenAI SDK 的 ChatCompletionMessageToolCall 对象。
    """
    # 当模型决定调用工具时，response.choices[0].message 包含:
    # message.content = None  (调用工具时不生成文本)
    # message.tool_calls = [tool_call, ...]

    print("=== tool_call 对象结构 ===")
    print("""
    ChatCompletionMessageToolCall:
        .id          = "call_abc123..."     # 唯一标识，用于 tool 消息的 tool_call_id 匹配
        .type        = "function"           # 目前只有 function
        .function.name       = "get_weather"     # 要调用的函数名
        .function.arguments  = '{"city":"北京"}'  # JSON 字符串，需要 json.loads()
    """)

    print("=== 对应的 tool 消息结构 ===")
    print("""
    {
        "role": "tool",
        "tool_call_id": "call_abc123...",  # 必须和上面的 .id 一致
        "name": "get_weather",             # 可选，建议填写
        "content": "{\\"weather\\": \\"晴，25°C\\"}"  # 字符串，通常是 JSON
    }
    """)

    print("=== 完整的 messages 流程 ===")
    print("""
    [1] {"role": "system", "content": "..."}
    [2] {"role": "user", "content": "北京天气怎么样？"}
    [3] {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", ...}]}
    [4] {"role": "tool", "tool_call_id": "call_1", "content": "{\\"weather\\":\\"晴\\"}"}
    [5] {"role": "assistant", "content": "北京今天晴朗，气温25°C"}
    """)


if __name__ == "__main__":
    inspect_tool_call()
```

---

## 2. Tool Schema 设计指南

### 2.1 description 的写法技巧

Tool Schema 中的 `description` 是**模型判断是否调用该工具的唯一依据**。写得好，模型能在合适的时候精准调用；写得差，模型要么不调，要么乱调。

```python
"""
演示 description 写法对模型调用行为的影响。
"""
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# 模拟函数
def send_email(to: str, subject: str, body: str) -> dict:
    return {"status": "sent", "to": to, "subject": subject}


# ============================================================
# 三种不同质量的 description 对比
# ============================================================

# 差: 描述太模糊，模型不知道什么时候该调用
BAD_DESC_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "发送邮件。",  # 太模糊！
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "收件人"},
                "subject": {"type": "string", "description": "主题"},
                "body": {"type": "string", "description": "正文"},
            },
            "required": ["to", "subject", "body"],
        },
    },
}

# 中等: 描述了功能，但缺少触发条件
MEDIUM_DESC_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "通过SMTP发送电子邮件。",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "收件人邮箱地址，如 user@example.com",
                },
                "subject": {
                    "type": "string",
                    "description": "邮件主题，简洁描述邮件内容",
                },
                "body": {
                    "type": "string",
                    "description": "邮件正文，支持纯文本",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
}

# 好: 描述了功能、触发场景、输入约束、输出行为
GOOD_DESC_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": """\
发送电子邮件给指定收件人。

触发场景（当用户表达以下意图时调用）:
- "发邮件给..."、"给...写封信"、"通知...邮件"
- "回复客户"、"发个邮件确认"
- 任何明确或隐含的邮件发送需求

注意:
- 调用前请确认收件人邮箱是否正确
- subject 应简洁概括邮件内容
- 发送成功会返回确认信息，失败会返回错误原因""",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "收件人邮箱地址。格式: username@domain.com。必填。",
                },
                "subject": {
                    "type": "string",
                    "description": "邮件主题。建议 5-30 字，概括邮件核心内容。",
                },
                "body": {
                    "type": "string",
                    "description": "邮件正文。纯文本格式。换行用\\n。",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
}


def test_tool_detection(user_query: str, tool_schema: dict, label: str):
    """测试模型是否能正确识别需要调用工具的查询"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个办公助手。根据用户意图决定是否调用工具。"},
            {"role": "user", "content": user_query},
        ],
        tools=[tool_schema],
        tool_choice="auto",
        temperature=0.0,
    )

    msg = response.choices[0].message
    has_call = msg.tool_calls is not None
    print(f"[{label}] '{user_query}' → 调用工具: {has_call}")
    if has_call:
        args = json.loads(msg.tool_calls[0].function.arguments)
        print(f"  参数: {args}")


if __name__ == "__main__":
    test_queries = [
        "帮我给老板发个邮件，说明天请假",
        "电子邮件是什么？",  # 不应触发
        "我想写封信给HR申请调岗",
    ]

    for query in test_queries:
        test_tool_detection(query, BAD_DESC_TOOL, "差desc")
        test_tool_detection(query, GOOD_DESC_TOOL, "好desc")
        print()
```

### 2.2 parameters 的完整设计

```python
"""
演示 parameters 的各种类型、枚举、约束。

一个设计精良的 parameters 定义应该:
1. 类型明确（string/number/boolean/array/object）
2. 枚举约束可选值
3. 用 minimum/maximum 约束数值范围
4. 用 description 解释每个字段的含义
"""
import json


def create_order_tool_schema() -> dict:
    """
    设计一个"创建订单"的工具 Schema。
    展示各种 parameters 类型的用法。
    """
    return {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "创建一个新订单。包含商品信息、收货地址、支付方式。",
            "parameters": {
                "type": "object",
                "properties": {
                    # --- 数组类型: 商品列表 ---
                    "items": {
                        "type": "array",
                        "description": "订单中的商品列表。至少包含1个商品。",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "product_id": {
                                    "type": "string",
                                    "description": "商品ID，如'PROD-001'",
                                },
                                "quantity": {
                                    "type": "integer",
                                    "description": "购买数量",
                                    "minimum": 1,
                                    "maximum": 999,
                                },
                                "color": {
                                    "type": "string",
                                    "description": "颜色规格（仅服装类需要）",
                                    "enum": ["红色", "蓝色", "黑色", "白色", "灰色"],
                                },
                                "size": {
                                    "type": "string",
                                    "description": "尺码（仅服装类需要）",
                                    "enum": ["S", "M", "L", "XL", "XXL"],
                                },
                            },
                            "required": ["product_id", "quantity"],
                            "additionalProperties": False,
                        },
                    },
                    # --- 嵌套对象: 收货地址 ---
                    "shipping_address": {
                        "type": "object",
                        "description": "收货地址信息",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "收件人姓名",
                                "minLength": 1,
                                "maxLength": 50,
                            },
                            "phone": {
                                "type": "string",
                                "description": "手机号码，11位数字",
                                "pattern": r"^1[3-9]\d{9}$",
                            },
                            "province": {
                                "type": "string",
                                "description": "省份，如'北京市'、'广东省'",
                            },
                            "city": {
                                "type": "string",
                                "description": "城市，如'朝阳区'、'深圳市'",
                            },
                            "detail": {
                                "type": "string",
                                "description": "详细地址，街道门牌号",
                            },
                            "zip_code": {
                                "type": "string",
                                "description": "邮政编码（选填）",
                            },
                        },
                        "required": ["name", "phone", "province", "city", "detail"],
                        "additionalProperties": False,
                    },
                    # --- 枚举类型: 支付方式 ---
                    "payment_method": {
                        "type": "string",
                        "description": "支付方式",
                        "enum": ["wechat", "alipay", "bank_card", "cod"],
                        "enumDescriptions": {
                            "wechat": "微信支付",
                            "alipay": "支付宝",
                            "bank_card": "银行卡",
                            "cod": "货到付款",
                        },
                    },
                    # --- 布尔类型: 是否加急 ---
                    "is_urgent": {
                        "type": "boolean",
                        "description": "是否加急配送（加急需额外收费10元）",
                        "default": False,
                    },
                    # --- 数值类型: 期望送达时间 ---
                    "expected_delivery_days": {
                        "type": "integer",
                        "description": "期望送达天数。普通3-7天，加急1-2天。",
                        "minimum": 1,
                        "maximum": 30,
                    },
                    # --- 字符串: 备注 ---
                    "remark": {
                        "type": "string",
                        "description": "订单备注。如：'请发顺丰'、'包装严实些'。最多200字。",
                        "maxLength": 200,
                    },
                },
                "required": ["items", "shipping_address", "payment_method"],
                "additionalProperties": False,
            },
        },
    }


if __name__ == "__main__":
    schema = create_order_tool_schema()
    print("=== 订单创建 Tool Schema ===")
    print(json.dumps(schema, ensure_ascii=False, indent=2))
    print()
    print("这个 Schema 展示了:")
    print("  - array + items: 商品列表（嵌套对象）")
    print("  - object: 收货地址（嵌套对象）")
    print("  - enum: 颜色、尺码、支付方式")
    print("  - boolean: 是否加急")
    print("  - integer + minimum/maximum: 数量、送达天数")
    print("  - string + pattern: 手机号格式验证")
    print("  - string + maxLength: 备注字数限制")
```

### 2.3 parameters 设计的最佳实践

```python
"""
Parameters 设计自查清单（代码化）。

每定义一个 Tool Schema，检查以下条目:
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class ParameterCheckResult:
    check_name: str
    passed: bool
    suggestion: str


def validate_tool_schema(tool_schema: dict) -> list[ParameterCheckResult]:
    """
    对 Tool Schema 进行质量检查。
    这不是运行时的 JSON Schema 验证器，而是设计层面的检查。
    """
    results = []
    func = tool_schema.get("function", {})

    # 检查1: function name 是否规范的 snake_case
    name = func.get("name", "")
    if "_" in name and name == name.lower():
        results.append(ParameterCheckResult("命名规范", True, f"'{name}' 是规范的 snake_case"))
    else:
        results.append(ParameterCheckResult("命名规范", False, "建议使用 snake_case，如 get_weather, create_order"))

    # 检查2: description 是否描述了触发条件
    desc = func.get("description", "")
    trigger_keywords = ["当用户", "调用时机", "触发场景", "用户说", "用户想", "用于", "Use when"]
    has_trigger_hints = any(kw in desc for kw in trigger_keywords)
    results.append(
        ParameterCheckResult(
            "触发条件描述",
            has_trigger_hints,
            "建议在 description 中说明何时应该调用" if not has_trigger_hints else "包含了触发条件提示",
        )
    )

    # 检查3: 每个 parameter 是否有 description
    params = func.get("parameters", {}).get("properties", {})
    missing_desc = [k for k, v in params.items() if not v.get("description")]
    results.append(
        ParameterCheckResult(
            "参数描述完整性",
            len(missing_desc) == 0,
            f"缺少描述的字段: {missing_desc}" if missing_desc else "所有字段都有 description",
        )
    )

    # 检查4: enum 字段是否有限制说明
    for pname, pdef in params.items():
        if "enum" in pdef:
            results.append(
                ParameterCheckResult(
                    f"枚举字段 '{pname}'",
                    True,
                    f"有 {len(pdef['enum'])} 个可选值: {pdef['enum']}",
                )
            )

    # 检查5: 是否有 required 字段
    required = func.get("parameters", {}).get("required", [])
    results.append(
        ParameterCheckResult(
            "必填字段",
            len(required) > 0,
            f"声明了 {len(required)} 个必填字段: {required}" if required else "建议至少声明一个 required 字段",
        )
    )

    return results


if __name__ == "__main__":
    # 对之前的 send_email tool 做检查
    good_tool = {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "发送邮件。当用户说'发邮件'、'写封信'、'通知某人'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "收件人邮箱"},
                    "subject": {"type": "string", "description": "邮件主题"},
                    "body": {"type": "string", "description": "邮件正文"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    }

    for check in validate_tool_schema(good_tool):
        status = "[OK]" if check.passed else "[!!]"
        print(f"{status} {check.check_name}: {check.suggestion}")
```

---

## 3. 单工具 vs 多工具

### 3.1 tool_choice 参数详解

```python
"""
tool_choice 参数的四种取值及其行为。

tool_choice 决定了模型是否以及如何选择工具:
- "auto" (默认): 模型自己决定是否调用、调用哪个
- "none" : 禁止调用任何工具（即使有 tools 定义）
- "required": 必须调用一个工具
- {"type": "function", "function": {"name": "xxx"}}: 强制调用指定工具
"""
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "查询当前时间",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "时区，如 Asia/Shanghai"}},
                "required": [],
            },
        },
    },
]


def test_tool_choice(user_query: str, tool_choice):
    """测试不同 tool_choice 参数的行为"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "你是助手。根据需要使用工具。"},
                {"role": "user", "content": user_query},
            ],
            tools=TOOLS,
            tool_choice=tool_choice,
            temperature=0.0,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            calls = [
                f"{tc.function.name}({tc.function.arguments})"
                for tc in msg.tool_calls
            ]
            return f"调用工具: {calls}"
        else:
            return f"直接回答: {msg.content[:80]}..."
    except Exception as e:
        return f"错误: {e}"


if __name__ == "__main__":
    # 不涉及工具的对话
    simple_query = "你好，介绍一下你自己"

    # 涉及工具的对话
    tool_query = "北京现在天气怎么样"

    print("=== tool_choice 行为对比 ===\n")

    for query, label in [(simple_query, "闲聊"), (tool_query, "天气查询")]:
        print(f"--- 用户说: '{query}' ({label}) ---")
        for choice in ["auto", "none", "required"]:
            result = test_tool_choice(query, choice)
            print(f"  tool_choice='{choice}': {result}")
        # 强制调用 get_time（不合理的用法，看结果）
        forced = {"type": "function", "function": {"name": "get_time"}}
        result = test_tool_choice(query, forced)
        print(f"  tool_choice=强制get_time: {result}")
        print()
```

### 3.2 多工具时的选择逻辑

```python
"""
演示模型在多工具场景下的选择逻辑。

当有多个工具可用时，模型基于以下因素决定调用哪个（或多个）:
1. 用户意图与工具描述的匹配度
2. 参数是否可以从上下文中推断
3. 是否有明确的工具名称提及
"""
import os
import json
from datetime import datetime, timedelta
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# 模拟的真实函数
def query_database(sql: str) -> dict:
    """模拟数据库查询"""
    data = {
        "SELECT COUNT(*) FROM users": {"count": 15234},
        "SELECT * FROM orders WHERE status='pending'": {"orders": [{"id": 1, "amount": 99.9}]},
    }
    return {"query": sql, "result": data.get(sql, {"message": "查询成功"})}


def search_knowledge_base(query: str, top_k: int = 3) -> dict:
    """模拟知识库搜索"""
    return {
        "query": query,
        "results": [
            {"title": "如何重置密码", "score": 0.95},
            {"title": "密码修改指南", "score": 0.87},
            {"title": "账户安全设置", "score": 0.76},
        ],
    }


def send_notification(user_id: str, message: str) -> dict:
    """模拟发送通知"""
    return {"status": "sent", "user_id": user_id, "message": message, "timestamp": datetime.now().isoformat()}


# 多工具 Schema
MULTI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "执行SQL查询。仅在用户明确要求查询数据库、查订单、查用户数据时调用。需要用户提供或能推断出SQL语句。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "要执行的SQL查询语句"}
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "搜索知识库/帮助文档。当用户问'怎么'、'如何'、'什么是'等知识性问题时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "top_k": {"type": "integer", "description": "返回结果数量，默认3"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "向指定用户发送通知消息。当用户说'通知'、'提醒'、'告诉xxx'时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "目标用户ID"},
                    "message": {"type": "string", "description": "通知内容"},
                },
                "required": ["user_id", "message"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "query_database": query_database,
    "search_knowledge_base": search_knowledge_base,
    "send_notification": send_notification,
}


def multi_tool_conversation(user_query: str) -> dict:
    """
    处理包含多工具选择的对话。
    """
    messages = [
        {"role": "system", "content": "你是企业办公助手。你可以查询数据库、搜索知识库、发送通知。根据用户意图选择最合适的工具。"},
        {"role": "user", "content": user_query},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=MULTI_TOOLS,
        tool_choice="auto",
        temperature=0.0,
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        print(f"  模型选择: {func_name}({json.dumps(func_args, ensure_ascii=False)})")

        # 执行
        if func_name in AVAILABLE_FUNCTIONS:
            result = AVAILABLE_FUNCTIONS[func_name](**func_args)
        else:
            result = {"error": f"unknown function: {func_name}"}

        return {"tool_called": func_name, "args": func_args, "result": result}
    else:
        return {"tool_called": None, "reply": msg.content}


if __name__ == "__main__":
    test_queries = [
        "怎么修改密码？",  # 应该调用 search_knowledge_base
        "帮我查一下有多少注册用户",  # 应该调用 query_database
        "提醒张三下午三点开会",  # 应该调用 send_notification
        "今天天气不错",  # 不应调用任何工具
    ]

    print("=== 多工具选择测试 ===\n")
    for q in test_queries:
        print(f"用户: {q}")
        result = multi_tool_conversation(q)
        if result["tool_called"]:
            print(f"  → 调用 {result['tool_called']}")
            print(f"  → 结果: {json.dumps(result['result'], ensure_ascii=False)[:100]}")
        else:
            print(f"  → 直接回答: {result['reply'][:80]}...")
        print()
```

### 3.3 工具命名冲突避免

```python
"""
多工具系统中的命名冲突和解决方案。

问题: 当两个工具的功能重叠时（如 search_files 和 search_content），
模型可能选错。解决方案:
1. 命名清晰：用动词+名词，如 search_by_filename, search_by_content
2. description 中明确区分触发条件
3. 必要时用 tool_choice 强制指定
"""

# 不推荐: 两个 search 容易混淆
BAD_NAMING = [
    {"function": {"name": "search", "description": "搜索文件"}},      # 太泛
    {"function": {"name": "search2", "description": "搜索内容"}},     # 糟糕的命名
]

# 推荐: 命名本身区分用途
GOOD_NAMING = [
    {
        "function": {
            "name": "search_files_by_name",  # 明确: 按文件名搜索
            "description": "根据文件名模式搜索文件。使用场景: 用户说'找到所有.py文件'、'找名为config的文件'。注意: 本函数不搜索文件内容，只搜索文件名。",
        }
    },
    {
        "function": {
            "name": "search_content_in_files",  # 明确: 搜索文件内容
            "description": "在文件内容中搜索关键词。使用场景: 用户说'哪些文件里提到了TODO'、'找包含API_KEY的文件'。注意: 本函数搜索文件内容，不搜索文件名。",
        }
    },
]

print("=== 工具命名对比 ===\n")
print("不推荐:")
for t in BAD_NAMING:
    print(f"  {t['function']['name']}: {t['function']['description']}")
print("\n推荐:")
for t in GOOD_NAMING:
    print(f"  {t['function']['name']}: {t['function']['description'][:80]}...")
```

---

## 4. 并行工具调用

### 4.1 原理与场景

```python
"""
并行工具调用 (Parallel Tool Calls)

当用户的问题需要多个独立工具配合时，模型可以一次请求调用多个工具。
这减少了对话轮次，降低了延迟。

适用场景:
- "北京和上海的天气分别是多少？" → 两次 get_weather 并行
- "帮我查订单状态并通知客户" → query_order + send_notification 并行
- "这个技术问题怎么解决？顺便看一下相关知识库文章" → search + search_kb 并行

注意: 只有互不依赖的调用才能并行。
如果 B 需要 A 的结果 → 必须串行（分两次请求）。
"""
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# 模拟函数
def get_weather(city: str) -> dict:
    weather_map = {
        "北京": "晴 25°C",
        "上海": "小雨 20°C",
        "深圳": "多云 30°C",
        "成都": "阴 22°C",
    }
    return {"city": city, "weather": weather_map.get(city, "未知")}


def get_aqi(city: str) -> dict:
    aqi_map = {"北京": 85, "上海": 55, "深圳": 42, "成都": 120}
    return {"city": city, "aqi": aqi_map.get(city, "未知"), "level": "良"}


AVAILABLE_FUNCTIONS = {"get_weather": get_weather, "get_aqi": get_aqi}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_aqi",
            "description": "查询城市空气质量指数(AQI)",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


def parallel_tool_call_demo(user_query: str) -> str:
    """
    演示并行工具调用的完整流程。
    关键区别: 处理多个 tool_calls 时，把每个结果都以 tool 消息加入，
    然后一次性发给模型。
    """
    messages = [
        {"role": "system", "content": "你是天气助手，可以查询天气和空气质量。"},
        {"role": "user", "content": user_query},
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.0,
    )

    assistant_message = response.choices[0].message

    if not assistant_message.tool_calls:
        return assistant_message.content

    # 检查是否有多个 tool_calls（并行调用）
    print(f"[并行调用] 模型一次请求了 {len(assistant_message.tool_calls)} 个工具:")
    for tc in assistant_message.tool_calls:
        print(f"  - {tc.function.name}({tc.function.arguments})")

    # 将 assistant 消息加入历史
    messages.append(assistant_message)

    # 逐个执行所有 tool_calls
    for tool_call in assistant_message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        # 执行函数
        if func_name in AVAILABLE_FUNCTIONS:
            result = AVAILABLE_FUNCTIONS[func_name](**func_args)
        else:
            result = {"error": f"Unknown function: {func_name}"}

        # 将结果作为 tool 消息加入
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": func_name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    # 所有工具执行完后，一次性发给模型生成最终回答
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
    )

    return final_response.choices[0].message.content


if __name__ == "__main__":
    # 场景1: 单个城市，需要天气 + AQI → 并行调用
    print("=== 场景1: 单个城市天气+空气质量(并行) ===")
    print(parallel_tool_call_demo("北京的天气和空气质量怎么样？"))
    print()

    # 场景2: 两个城市天气 → 并行调用
    print("=== 场景2: 两个城市天气对比(并行) ===")
    print(parallel_tool_call_demo("北京和上海今天天气对比"))
```

### 4.2 串行调用 vs 并行调用

```python
"""
演示工具调用的串行 vs 并行模式。

串行: 第二次调用依赖第一次的结果
并行: 两次调用完全独立
"""
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


def get_user_id(name: str) -> dict:
    """模拟: 根据姓名查用户ID"""
    users = {"张三": "U001", "李四": "U002", "王五": "U003"}
    return {"name": name, "user_id": users.get(name, "unknown")}


def get_orders(user_id: str) -> dict:
    """模拟: 根据用户ID查订单"""
    orders = {
        "U001": [{"order_id": "ORD-1001", "amount": 299.0}],
        "U002": [{"order_id": "ORD-2001", "amount": 599.0}],
    }
    return {"user_id": user_id, "orders": orders.get(user_id, [])}


AVAILABLE_FUNCTIONS = {"get_user_id": get_user_id, "get_orders": get_orders}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_id",
            "description": "根据用户姓名查询其用户ID。",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": "根据用户ID查询其订单列表。",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
]


def serial_call_demo(user_query: str) -> str:
    """
    串行调用: 需要两轮 API 请求。
    第一轮获取 user_id，第二轮用 user_id 查订单。
    
    注意: 因为 get_orders 依赖 get_user_id 的结果，
    模型无法并行调用它们。需要:
    Request-1: 模型调用 get_user_id
    Request-2: 把结果传给模型，模型再调用 get_orders
    Request-3: 把订单结果传给模型，模型生成最终回答
    """
    messages = [
        {"role": "system", "content": "你是订单查询助手。先查用户ID再查订单。"},
        {"role": "user", "content": user_query},
    ]

    # ---- 第一轮: 查用户ID ----
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.0,
    )

    msg = response.choices[0].message

    if not msg.tool_calls:
        return msg.content

    # 执行 get_user_id
    messages.append(msg)
    tc = msg.tool_calls[0]
    user_id_result = get_user_id(**json.loads(tc.function.arguments))
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.function.name,
            "content": json.dumps(user_id_result, ensure_ascii=False),
        }
    )

    print(f"第1轮: 获取 user_id = {user_id_result}")

    # ---- 第二轮: 查订单（基于上一轮结果） ----
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.0,
    )

    msg = response.choices[0].message

    if not msg.tool_calls:
        return msg.content

    # 执行 get_orders
    messages.append(msg)
    tc = msg.tool_calls[0]
    orders_result = get_orders(**json.loads(tc.function.arguments))
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.function.name,
            "content": json.dumps(orders_result, ensure_ascii=False),
        }
    )

    print(f"第2轮: 获取订单 = {orders_result}")

    # ---- 第三轮: 生成最终回答 ----
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
    )

    return final_response.choices[0].message.content


if __name__ == "__main__":
    print("=== 串行调用示例 ===")
    print("用户: 帮我查张三的订单")
    print(serial_call_demo("帮我查张三的订单"))
```

---

## 5. 工具执行与结果返回

### 5.1 完整的执行循环

```python
"""
Function Calling 的完整执行循环 —— 这是生产级别的实现。

支持:
- 多轮工具调用（模型拿到结果后可能继续调工具）
- 并行工具调用
- 最大轮次限制（防止无限循环）
- 错误处理
"""
import os
import json
from openai import OpenAI
from typing import Callable

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


class ToolExecutor:
    """
    Function Calling 的执行器。

    职责:
    1. 维护对话状态（messages）
    2. 检测 tool_calls 并执行
    3. 将结果反馈给模型
    4. 循环直到模型不再调用工具或达到最大轮次
    """

    def __init__(
        self,
        tools: list[dict],
        function_map: dict[str, Callable],
        max_rounds: int = 5,
    ):
        """
        参数:
            tools: Tool Schema 列表
            function_map: {"function_name": actual_function} 映射
            max_rounds: 最大工具调用轮次，防止无限循环
        """
        self.tools = tools
        self.function_map = function_map
        self.max_rounds = max_rounds

    def execute(
        self,
        messages: list[dict],
        model: str = "gpt-4o",
        temperature: float = 0.0,
    ) -> list[dict]:
        """
        执行对话，自动处理所有工具调用。

        参数:
            messages: 初始 messages 列表
            model: 模型名称
            temperature: 生成参数

        返回:
            最终的 messages 列表（包含完整的对话历史）
        """
        for round_num in range(self.max_rounds):
            # 调用模型
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=temperature,
            )

            assistant_message = response.choices[0].message

            # 如果没有 tool_calls，对话结束
            if not assistant_message.tool_calls:
                messages.append(assistant_message)
                break

            # 将包含 tool_calls 的 assistant 消息加入历史
            messages.append(assistant_message)

            # 执行所有 tool_calls
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"[Round {round_num + 1}] 调用: {func_name}({json.dumps(func_args, ensure_ascii=False)})")

                try:
                    # 执行函数
                    result = self.function_map[func_name](**func_args)
                    result_content = json.dumps(result, ensure_ascii=False)
                except Exception as e:
                    # 函数执行失败，返回错误信息给模型
                    result_content = json.dumps({"error": str(e), "function": func_name}, ensure_ascii=False)
                    print(f"  执行失败: {e}")

                # 将结果追加到 messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": result_content,
                    }
                )
        else:
            # 达到最大轮次
            print(f"[警告] 达到最大轮次 {self.max_rounds}，强制终止")

        return messages

    def run(
        self,
        system_prompt: str,
        user_query: str,
        model: str = "gpt-4o",
    ) -> str:
        """
        便捷方法：运行一次对话并返回最终回复文本。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
        final_messages = self.execute(messages, model)

        # 返回最后一条 assistant 消息的内容
        for msg in reversed(final_messages):
            if msg["role"] == "assistant" and msg.get("content"):
                return msg["content"]

        return "（无法生成回复）"


if __name__ == "__main__":
    # 构建一个多工具执行器来测试
    def calculator(expression: str) -> dict:
        """安全的计算器（仅支持基本运算）"""
        # 安全：只允许数字和基本运算符
        allowed_chars = set("0123456789+-*/()., ")
        if not all(c in allowed_chars for c in expression):
            return {"error": "表达式包含不支持的字符"}
        try:
            result = eval(expression, {"__builtins__": {}})
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"error": f"计算错误: {e}"}

    def get_current_time() -> dict:
        """获取当前时间"""
        from datetime import datetime
        return {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    executor = ToolExecutor(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "执行数学计算。当用户需要计算时调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "数学表达式，如 '2+3*4'"}
                        },
                        "required": ["expression"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "获取当前时间。",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ],
        function_map={"calculator": calculator, "get_current_time": get_current_time},
    )

    reply = executor.run(
        system_prompt="你是助手，可以计算和查时间。",
        user_query="计算 (15.5 + 8.3) * 2.1 的结果，并告诉我现在的准确时间。",
    )
    print(f"\n最终回复:\n{reply}")
```

---

## 6. 错误处理与重试

### 6.1 工具执行失败的处理

```python
"""
Function Calling 中的错误处理策略。

常见错误类型:
1. 函数执行异常（网络超时、数据库连接失败）
2. 参数格式错误（模型传了不该传的参数）
3. 函数返回了模型无法理解的结果
4. 参数缺失（模型没有提供必填参数）

应对策略:
A. 返回错误信息给模型，让它重试（推荐）
B. 捕获错误后进行参数修正
C. 给出 fallback 回答
"""
import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


def send_email(to: str, subject: str, body: str) -> dict:
    """
    模拟发送邮件，故意在某些条件下失败来演示错误处理。
    """
    # 模拟: 如果收件人邮箱格式不对，抛出异常
    if "@" not in to:
        raise ValueError(f"收件人邮箱格式错误: '{to}'。正确格式如 'user@example.com'")

    # 模拟: 如果主题为空，抛出异常
    if not subject or len(subject.strip()) == 0:
        raise ValueError("邮件主题不能为空")

    # 模拟: 网络超时（概率性）
    import random
    if random.random() < 0.2:  # 20% 概率模拟失败
        raise ConnectionError("SMTP连接超时，邮件服务器无响应")

    return {"status": "sent", "to": to, "subject": subject, "message_id": "MSG-001"}


SEND_EMAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "发送电子邮件",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "收件人邮箱"},
                "subject": {"type": "string", "description": "邮件主题"},
                "body": {"type": "string", "description": "邮件正文"},
            },
            "required": ["to", "subject", "body"],
        },
    },
}


def run_with_retry(user_query: str, max_retries: int = 3) -> str:
    """
    带重试逻辑的 Function Calling。

    策略:
    1. 函数执行失败 → 返回错误信息给模型（告诉它什么错了）
    2. 模型基于错误信息重新生成 tool_call
    3. 达到最大重试次数后放弃，生成错误回复
    """
    messages = [
        {"role": "system", "content": "你是邮件助手。发送邮件前确保邮箱格式正确、主题不为空。"},
        {"role": "user", "content": user_query},
    ]

    retry_count = 0
    last_error = None

    while retry_count <= max_retries:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=[SEND_EMAIL_TOOL],
            tool_choice="auto",
            temperature=0.0,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            # 模型决定不再调用工具（可能是放弃了）
            return msg.content or "（无回复）"

        # 处理 tool_calls
        messages.append(msg)

        for tool_call in msg.tool_calls:
            func_args = json.loads(tool_call.function.arguments)

            print(f"[尝试 {retry_count + 1}] 调用 send_email(to={func_args.get('to')}, subject={func_args.get('subject')})")

            try:
                result = send_email(**func_args)
                # 成功！
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "send_email",
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                print(f"  成功: {result}")

                # 获取最终回复
                final_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.0,
                )
                return final_response.choices[0].message.content

            except ValueError as e:
                # 参数错误 —— 告诉模型哪里错了
                error_msg = f"参数错误: {str(e)}。请修正参数后重试。"
                last_error = str(e)
                print(f"  失败(ValueError): {e}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "send_email",
                        "content": json.dumps({"error": error_msg}, ensure_ascii=False),
                    }
                )
                retry_count += 1
                break  # 退出 for 循环，让模型重新生成 tool_call

            except ConnectionError as e:
                # 网络错误 —— 告诉模型稍后重试
                error_msg = f"网络错误: {str(e)}。请告知用户稍后重试。"
                last_error = str(e)
                print(f"  失败(ConnectionError): {e}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "send_email",
                        "content": json.dumps({"error": error_msg}, ensure_ascii=False),
                    }
                )
                # 网络错误不重试（避免重复发送），让模型告知用户
                final_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.0,
                )
                return final_response.choices[0].message.content

    return f"发送失败（已重试{max_retries}次）: {last_error}"


if __name__ == "__main__":
    # 测试1: 邮箱格式错误 → 模型应该能修正
    print("=== 测试1: 邮箱格式错误 ===")
    result = run_with_retry("帮我发邮件给张三，主题'测试'，内容'你好'")
    print(f"结果: {result}\n")

    # 测试2: 主题为空 → 模型应该自动生成主题
    print("=== 测试2: 主题为空 ===")
    result = run_with_retry("发邮件给 test@example.com，什么都不写，就发个空白邮件")
    print(f"结果: {result}\n")
```

### 6.2 超时处理

```python
"""
工具调用的超时处理。

当一个工具调用可能耗时较长时（如调用外部API、查询大型数据库），
需要设置超时防止用户等待过久。
"""
import os
import json
import signal
import asyncio
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


def slow_api_call(query: str) -> dict:
    """
    模拟一个可能很慢的外部 API 调用。
    """
    import time

    # 模拟: 如果查询包含 "slow" 则休眠 10 秒
    if "slow" in query.lower():
        time.sleep(10)
    else:
        time.sleep(0.1)

    return {"query": query, "result": "success"}


def execute_with_timeout(func, timeout: float, *args, **kwargs):
    """
    在 Windows 上使用 threading 实现带超时的函数调用。
    （Unix 上可以用 signal.alarm，Windows 需要用 threading）
    """
    import threading

    result_container = {"result": None, "error": None, "done": False}

    def target():
        try:
            result_container["result"] = func(*args, **kwargs)
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            result_container["done"] = True

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if not result_container["done"]:
        return {"error": f"函数调用超时（>{timeout}秒）"}

    if result_container["error"]:
        return {"error": result_container["error"]}

    return result_container["result"]


if __name__ == "__main__":
    print("=== 超时处理测试 ===")

    # 正常调用
    result = execute_with_timeout(slow_api_call, 2.0, "normal query")
    print(f"正常查询: {result}")

    # 超时调用
    result = execute_with_timeout(slow_api_call, 2.0, "this is a slow query")
    print(f"慢查询: {result}")
```

---

## 7. 工具设计模式

### 7.1 三种基本模式

```python
"""
Function Calling 的三种基本工具设计模式。

1. 查询类 (Query): 只读操作，获取信息
   - get_weather, search_docs, query_database, get_time
   - 特点: 幂等，无副作用，可安全重试

2. 操作类 (Action): 写操作，改变外部状态
   - send_email, create_file, delete_record, place_order
   - 特点: 有副作用，需谨慎重试，需要确认机制

3. 验证类 (Validation): 检查前置条件
   - validate_email, check_permission, verify_credentials
   - 特点: 通常在其他操作前调用，失败则阻止后续操作
"""
import os
import json
from openai import OpenAI
from typing import Any

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ============================================================
# 模拟后端系统
# ============================================================
class BackendSystem:
    """模拟一个后端系统的状态"""

    def __init__(self):
        self.emails_sent: list[dict] = []
        self.files_created: list[str] = []
        self.valid_users = {"admin": "admin123", "user1": "pass1"}

    # 查询类
    def get_email_count(self) -> dict:
        return {"count": len(self.emails_sent), "emails": self.emails_sent}

    def get_file_list(self) -> dict:
        return {"files": self.files_created}

    # 验证类
    def check_auth(self, username: str, password: str) -> dict:
        if username in self.valid_users and self.valid_users[username] == password:
            return {"authenticated": True, "username": username}
        return {"authenticated": False, "error": "用户名或密码错误"}

    def validate_email(self, address: str) -> dict:
        if "@" in address and "." in address.split("@")[1]:
            return {"valid": True, "address": address}
        return {"valid": False, "error": f"'{address}' 不是有效的邮箱地址"}

    # 操作类
    def send_email(self, to: str, subject: str, body: str) -> dict:
        self.emails_sent.append({"to": to, "subject": subject, "body": body})
        return {"status": "sent", "email_id": len(self.emails_sent)}

    def create_file(self, filename: str, content: str) -> dict:
        if filename in self.files_created:
            return {"error": f"文件 '{filename}' 已存在"}
        self.files_created.append(filename)
        return {"status": "created", "filename": filename}


# ============================================================
# 完整的三种模式 Schema
# ============================================================
THREE_PATTERN_TOOLS = [
    # --- 验证类 ---
    {
        "type": "function",
        "function": {
            "name": "check_auth",
            "description": "验证用户身份。在执行任何敏感操作前必须调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {"type": "string", "description": "用户名"},
                    "password": {"type": "string", "description": "密码"},
                },
                "required": ["username", "password"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_email",
            "description": "验证邮箱地址格式是否有效。发邮件前建议先验证。",
            "parameters": {
                "type": "object",
                "properties": {
                    "address": {"type": "string", "description": "要验证的邮箱地址"}
                },
                "required": ["address"],
            },
        },
    },
    # --- 查询类 ---
    {
        "type": "function",
        "function": {
            "name": "get_email_count",
            "description": "查询已发送的邮件数量和列表。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_file_list",
            "description": "查询已创建的文件列表。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # --- 操作类 ---
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "发送邮件（操作类，有副作用）。调用前建议先 validate_email 和 check_auth。",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string", "description": "收件人邮箱"},
                    "subject": {"type": "string", "description": "邮件主题"},
                    "body": {"type": "string", "description": "邮件正文"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "创建新文件（操作类）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string", "description": "文件名，如 'report.txt'"},
                    "content": {"type": "string", "description": "文件内容"},
                },
                "required": ["filename", "content"],
            },
        },
    },
]


def run_three_pattern_demo(user_query: str, system: BackendSystem):
    """运行三种模式协作的对话"""
    function_map = {
        "check_auth": system.check_auth,
        "validate_email": system.validate_email,
        "get_email_count": system.get_email_count,
        "get_file_list": system.get_file_list,
        "send_email": system.send_email,
        "create_file": system.create_file,
    }

    messages = [
        {
            "role": "system",
            "content": """你是系统管理助手。重要规则:
1. 执行任何操作类工具前，必须先通过 check_auth 验证身份
2. 发送邮件前，先 validate_email 验证收件人地址
3. 查询类工具可以随时调用，无需验证
4. 操作失败时，报告具体的错误原因""",
        },
        {"role": "user", "content": user_query},
    ]

    max_rounds = 8
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=THREE_PATTERN_TOOLS,
            tool_choice="auto",
            temperature=0.0,
        )

        msg = response.choices[0].message
        if not msg.tool_calls:
            messages.append(msg)
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            fn = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  [{fn}] {json.dumps(args, ensure_ascii=False)}")

            try:
                result = function_map[fn](**args)
            except Exception as e:
                result = {"error": str(e)}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    # 最后一条 assistant 消息
    for m in reversed(messages):
        if m["role"] == "assistant" and m.get("content"):
            return m["content"]
    return "（无回复）"


if __name__ == "__main__":
    system = BackendSystem()

    print("=== 场景1: 未授权就发邮件 ===")
    reply = run_three_pattern_demo("帮我发邮件给 test@example.com，主题'Hello'，内容'World'", system)
    print(f"回复: {reply}\n")

    print("=== 场景2: 先登录再操作 ===")
    reply = run_three_pattern_demo(
        "用 admin/admin123 登录，然后查一下发了多少邮件，再查一下有哪些文件",
        system,
    )
    print(f"回复: {reply}\n")

    print("=== 场景3: 完整的安全流程 ===")
    reply = run_three_pattern_demo(
        "1. 用 admin/admin123 登录 2. 验证 test@example.com 是否有效 3. 发邮件给 test@example.com，主题'通知'，内容'系统维护中' 4. 创建文件 maintenance.log 内容'维护时间: 2024-06-15'",
        system,
    )
    print(f"回复: {reply}\n")
```

---

## 基础练习

### 练习 1: 单工具天气助手
**场景**: 实现一个天气查询助手，包含当前天气、未来天气预报两个工具。
**要求**: 完整的 Schema 定义 + 执行循环 + 友好的自然语言回复。
**文件**: `exercise/ai-application/ch02_function_calling/ex1_weather_assistant.py`

### 练习 2: 多工具办公助手
**场景**: 实现会议安排助手，包括查日历、预定会议室、发会议通知三个工具。
**要求**: 处理串行依赖（先查日历再预定），实现错误恢复。
**文件**: `exercise/ai-application/ch02_function_calling/ex2_office_assistant.py`

### 练习 3: 工具链编排
**场景**: 实现一个"下单 → 支付 → 发货"的电商工具链。
**要求**: 每个步骤验证前置条件，失败时回滚或补偿。
**文件**: `exercise/ai-application/ch02_function_calling/ex3_order_pipeline.py`

## 进阶练习

### 练习 4: ToolExecutor 框架
**场景**: 实现一个通用的 ToolExecutor 类，支持任意工具集的注册和执行。
**要求**: 支持并行调用、最大轮次限制、超时、重试、日志记录。
**文件**: `exercise/ai-application/ch02_function_calling/ex4_tool_executor.py`

### 练习 5: 对话式数据库查询
**场景**: 实现自然语言查询数据库的助手，将用户意图转为 SQL 执行并解释结果。
**要求**: SQL 安全校验（只允许 SELECT）、结果友好化展示、支持追问。
**文件**: `exercise/ai-application/ch02_function_calling/ex5_nl2sql.py`

---

## 常见错误

### 错误 1: 忘记将 assistant (含 tool_calls) 加入 messages

```python
# 错误:
msg = response.choices[0].message
# msg.tool_calls 不为 None
# 直接追加 tool 消息，没有先追加 assistant 消息
messages.append({"role": "tool", "tool_call_id": ..., "content": ...})
# API 报错: "Expected role 'assistant' before role 'tool'"

# 修正:
messages.append(msg)  # 先追加 assistant 消息
messages.append({"role": "tool", ...})  # 再追加 tool 消息
```

### 错误 2: tool_call_id 不匹配

```python
# 错误:
for tc in msg.tool_calls:
    messages.append({
        "role": "tool",
        "tool_call_id": "my_own_id",  # 错了！必须是 tc.id
        ...
    })
# API 报错: tool_call_id does not match any tool call

# 修正:
for tc in msg.tool_calls:
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,  # 必须使用模型返回的 id
        ...
    })
```

### 错误 3: function.arguments 是字符串，忘记 json.loads()

```python
# 错误:
args = tc.function.arguments  # 这是 JSON 字符串！
func(**args)  # TypeError: 传入了字符串而非 dict

# 修正:
args = json.loads(tc.function.arguments)  # 解析为 dict
func(**args)
```

### 错误 4: 工具执行后没有再次调用模型

```python
# 错误: 把工具结果追加到 messages 后就以为结束了
# 模型拿到了工具结果，但还没有机会基于结果生成回复
# 用户看到的是空白或上一次的回复

# 修正: 追加 tool 消息后，必须再次调用 client.chat.completions.create()
final = client.chat.completions.create(model="gpt-4o", messages=messages)
print(final.choices[0].message.content)
```

### 错误 5: tool_choice="required" 但用户问题不需要工具

```python
# 错误: 用户说"你好"，tool_choice="required" 强制模型调工具
# 结果: 模型硬生生编造一个工具调用（参数乱填）

# 修正: 日常对话用 tool_choice="auto"，只在明确需要工具时用 "required"
```

### 错误 6: 在并行调用中混用依赖

```python
# 错误: 设计工具时 A 的输出是 B 的输入，但期望模型并行调用它们
# 模型无法预测 A 的输出，所以 B 的参数会是编造的

# 修正:
# - 如果需要并行: 确保两个工具完全独立
# - 如果有依赖: 分两次请求，第一次获取 A 结果，第二次传给 B
```

---

## 本章小结

本章深入学习了 Function Calling 的完整体系：

| 知识点 | 核心要点 |
|--------|----------|
| 调用链路 | 定义 Schema → 模型判断 → 执行函数 → 返回结果 → 模型回复 |
| Schema 设计 | description 写触发条件，parameters 写类型约束 |
| tool_choice | auto/none/required/指定工具，四种策略对应不同场景 |
| 并行调用 | 独立工具一次请求并行，依赖工具分多次请求串行 |
| 消息顺序 | system → user → assistant(tool_calls) → tool → assistant(final) |
| 错误处理 | 捕获异常 → 返回错误信息 → 让模型重试或 fallback |
| 设计模式 | 查询类(幂等) + 操作类(副作用) + 验证类(前置条件) |

下一章将学习流式处理（Streaming），让模型的输出像打字一样逐字显示，大幅提升用户体验。
