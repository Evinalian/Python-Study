"""
练习 2: 多工具办公助手（含错误恢复）

场景:
    实现一个会议安排助手，包括三个工具：
    1. check_calendar: 查询指定时间段是否有空
    2. book_room: 预定会议室
    3. send_notification: 发送会议通知

    这三个工具有依赖关系：先查日历 → 有空则预定会议室 → 发通知。
    需要在某个步骤失败时优雅地处理错误。

要求:
    1. 定义三个工具的完整 Schema
       - check_calendar: date, start_time, end_time
       - book_room: room_name, date, start_time, end_time, title
       - send_notification: recipients, title, date, time, room

    2. 实现模拟函数，包含以下"坑":
       - check_calendar 在 13:00-14:00 总是返回"已占用"（午休时间）
       - book_room 在 room_name 为 "A101" 时 30% 概率失败（模拟冲突）
       - send_notification 在 recipients 为空时抛出异常

    3. 实现带错误恢复的 run_scheduling(user_query):
       - 模型调用工具 → 工具返回错误 → 模型看到错误 → 重新决策
       - 例如: 会议室预定失败 → 建议换一个会议室 → 重试
       - 时间冲突 → 建议换时间 → 重试

    4. 测试场景:
       - "安排明天上午10点-11点在B201的周会，通知张三和李四"
       - "帮我看看明天下午有没有空，有空的话定个会议室讨论项目"

TODO:
    1. 定义 TOOLS 列表
    2. 实现 check_calendar, book_room, send_notification（含模拟失败逻辑）
    3. 实现 run_scheduling(user_query) —— 支持最多 3 轮重试
    4. 测试包含错误恢复的完整对话
"""

import os
import json
import random
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

# ============================================================
# 后端状态（模拟数据库）
# ============================================================
# 已占用的时间段: {(date, start_time, end_time): "占用原因"}
BUSY_SLOTS: dict = {
    ("2024-06-17", "13:00", "14:00"): "午休时间（系统保留）",
    ("2024-06-17", "09:00", "10:00"): "已预定: 全员早会",
}

# 已预定的会议室: {(room, date, start, end): "预定人"}
BOOKED_ROOMS: dict = {}

# 发送的通知记录
NOTIFICATIONS_SENT: list = []


# ============================================================
# TODO 1: 定义三个工具的 Schema
# ============================================================
TOOLS = [
    # TODO: check_calendar
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "check_calendar",
    #         "description": "查询指定时间段是否有空。返回空闲或冲突信息。",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "date": {"type": "string", "description": "日期 YYYY-MM-DD"},
    #                 "start_time": {"type": "string", "description": "开始时间 HH:MM"},
    #                 "end_time": {"type": "string", "description": "结束时间 HH:MM"},
    #             },
    #             "required": ["date", "start_time", "end_time"],
    #         },
    #     },
    # },
    # TODO: book_room
    # TODO: send_notification
]

# ============================================================
# TODO 2: 实现三个模拟函数
# ============================================================
def check_calendar(date: str, start_time: str, end_time: str) -> dict:
    """
    检查指定时间段是否空闲。

    返回格式:
    - 空闲: {"available": true, "date": "...", "start_time": "...", "end_time": "..."}
    - 冲突: {"available": false, "conflict": "冲突原因"}
    """
    # TODO:
    # 1. 检查 BUSY_SLOTS 中是否有重叠的时间段
    # 2. 如果有冲突，返回冲突原因
    # 3. 如果空闲，返回 available=True
    pass


def book_room(room_name: str, date: str, start_time: str, end_time: str, title: str) -> dict:
    """
    预定会议室。

    失败条件:
    - room_name == "A101": 30% 概率返回"会议室已被预定"
    - 时间段与 BOOKED_ROOMS 冲突: 返回冲突信息
    """
    # TODO:
    # 1. 检查 BOOKED_ROOMS 是否有冲突（同房间同时间段）
    # 2. A101 特殊处理: random.random() < 0.3 时模拟失败
    # 3. 成功时写入 BOOKED_ROOMS 并返回成功
    pass


def send_notification(recipients: list[str], title: str, date: str, time: str, room: str) -> dict:
    """
    发送会议通知。

    失败条件:
    - recipients 为空列表: 抛出 ValueError("收件人列表不能为空")
    """
    # TODO:
    # 1. 如果 recipients 为空，raise ValueError
    # 2. 将通知记录加入 NOTIFICATIONS_SENT
    # 3. 返回成功信息
    pass


# ============================================================
# TODO 3: 实现带错误恢复的对话处理
# ============================================================
def run_scheduling(user_query: str, max_rounds: int = 5) -> str:
    """
    处理会议安排对话。

    错误恢复策略:
    - 工具返回错误时，把错误信息以 tool 消息返回给模型
    - 模型基于错误信息决定下一步（换时间 / 换会议室 / 告知用户）
    - 最多 max_rounds 轮工具调用
    """
    messages = [
        {
            "role": "system",
            "content": """你是会议安排助手。

工作流程:
1. 先 check_calendar 确认时间空闲
2. 空闲则 book_room 预定会议室
3. 预定成功后 send_notification 通知参会人

错误处理:
- 时间冲突: 建议用户换一个时间段
- 会议室预定失败: 建议换一个会议室或时间
- 通知失败: 报告具体错误

注意: 在用户没有明确说哪个会议室时，先查日历再决定。""",
        },
        {"role": "user", "content": user_query},
    ]

    # TODO: 实现多轮工具调用循环
    # 每轮:
    #   1. 调用 API (带 tools)
    #   2. 无 tool_calls → 返回 assistant 回复
    #   3. 有 tool_calls → 执行函数（try/except）
    #   4. 追加 tool 消息（成功结果 或 错误信息）
    #   5. 继续下一轮
    pass


# ============================================================
# TODO 4: 测试
# ============================================================
if __name__ == "__main__":
    print("=== 测试多工具办公助手 ===\n")

    # 场景1: 正常安排
    print("--- 场景1: 正常安排 ---")
    # reply = run_scheduling("安排明天上午10点到11点在B201的周会，通知张三和李四")
    # print(reply)

    # 场景2: 时间冲突
    print("\n--- 场景2: 时间冲突 ---")
    # reply = run_scheduling("帮我安排明天上午9点到10点的会议")
    # print(reply)

    # 场景3: 没有指定会议室，需要先查日历
    print("\n--- 场景3: 没有指定会议室 ---")
    # reply = run_scheduling("明天下午3点我有个项目评审，帮我安排一下")
    # print(reply)

    print("\n请完成 TODO 后取消注释运行。")
