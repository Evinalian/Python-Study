"""
章节：第06章 多智能体协作
题目：实现多 Agent 消息总线
类型：基础练习

题目描述：
实现一个完整的 MessageBus 系统，支持 Agent 之间的异步消息通信。
消息总线是多 Agent 系统中平等协商模式的基础设施，它让 Agent 之间可以
解耦地发送、接收和订阅消息。

要求：
1. 定义 AgentMessage 数据类：
   - id: 消息唯一 ID（UUID）
   - sender, receiver: 发送者和接收者（"broadcast" 表示广播）
   - msg_type: 消息类型（REQUEST / RESPONSE / PROPOSAL / OBJECTION / AGREEMENT / QUESTION / INFORMATION）
   - content: 消息正文
   - timestamp: ISO 时间戳
   - reply_to: 回复的消息 ID
   - metadata: 附加元数据字典
2. 实现 MessageBus 类：
   - register(agent_name): 注册 Agent 到总线
   - subscribe(agent_name, msg_type): 订阅特定消息类型
   - send(message): 发送消息（支持单播和广播）
   - receive(agent_name, timeout): 接收消息（支持超时）
   - get_history(between_tuple): 查询消息历史
3. 实现 PeerAgent 类：
   - 可以通过总线发送和接收消息
   - 实现 think_and_respond(message): 收到消息后自主决定如何回应
   - 支持行为特征：assertiveness（自信度）和 cooperativeness（合作度）
4. 实现一个简单的消息路由演示：
   - 创建 3 个 Agent 注册到总线
   - 模拟一个协商场景（如讨论"产品定价策略"）
   - 观察消息在 Agent 间的流动

提示：
- 使用 asyncio.Queue 实现每个 Agent 的独立消息队列
- broadcast 时遍历所有注册的 Agent（跳过发送者）
- get_history 支持按发送者/接收者对过滤
- think_and_respond 调用 LLM 决定：同意/反对/忽略/提问
- 消息类型用 Enum 定义，不要用裸字符串
"""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from dataclasses import dataclass, field

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
#
# 第1步：定义 MessageType 枚举
# - REQUEST: 请求帮助
# - RESPONSE: 回复请求
# - PROPOSAL: 提出方案
# - OBJECTION: 反对某个方案
# - AGREEMENT: 同意某个方案
# - QUESTION: 提问
# - INFORMATION: 共享信息
#
# 第2步：实现 AgentMessage 数据类
# - 包含所有上述字段
# - 使用 field(default_factory=...) 设置默认值
# - id 默认生成 UUID
# - timestamp 默认当前 ISO 时间
#
# 第3步：实现 MessageBus 类
# - __init__(): 初始化 _queues (dict), _history (list), _subscriptions (dict)
# - register(agent_name): 为该 Agent 创建 asyncio.Queue
# - subscribe(agent_name, msg_type): 添加到订阅列表
# - send(message):
#   * 追加到 _history
#   * 如果 receiver == "broadcast" → 推送给所有 Agent（除发送者）
#   * 否则 → 推送给指定 Agent
#   * 同时检查订阅：如果是订阅的消息类型，额外推送
# - receive(agent_name, timeout):
#   * 从对应队列获取消息
#   * 支持超时（asyncio.wait_for）
# - get_history(between): 按条件过滤历史消息
#
# 第4步：实现 PeerAgent 类
# - __init__(name, domain, system_prompt, llm_client, bus, assertiveness, cooperativeness)
# - 注册到总线: bus.register(name)
# - think_and_respond(message):
#   * 构建决策 prompt（包含个性参数）
#   * 调用 LLM 决定回应策略
#   * 返回 AgentMessage 或 None（忽略）
# - 回应策略: IGNORE / AGREE / OBJECT / QUESTION
#
# 第5步：实现消息处理循环
# - 持续从总线接收消息
# - 对每条消息调用 think_and_respond
# - 通过总线发送回应
# - 实现超时退出机制
#
# 第6步：编写演示场景
# - 创建 3 个 PeerAgent（如：产品经理、营销专家、数据分析师）
# - 模拟一个主题讨论
# - 打印消息时间线
# - 查询特定 Agent 间的对话历史

# 提示：
# - asyncio.Queue 的 put 和 get 都是异步操作
# - broadcast 需要遍历 self._queues，注意不要修改正在遍历的字典
# - 消息的 reply_to 字段用于链接对话线程
# - PeerAgent 的 think_and_respond 方法中，prompt 要包含足够上下文
