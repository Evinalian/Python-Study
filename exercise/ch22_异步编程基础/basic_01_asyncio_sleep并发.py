"""
章节：第22章 异步编程基础
题目：用 asyncio.sleep 模拟并发等待
类型：基础练习

题目描述：
写一个程序，用 `asyncio.gather()` 同时运行 3 个协程，分别等待 3 秒、2 秒、1 秒后打印完成信息。
验证总耗时约为 3 秒（而非串行的 6 秒），从而理解异步并发的优势。

要求：
1. 定义一个 `wait_and_print(name, delay)` 协程
2. 用 asyncio.gather() 同时启动 3 个协程
3. 记录总耗时并打印
4. 打印每个协程的返回值（gather 返回结果列表）

前置准备：
无需安装额外包（asyncio 是 Python 内置模块）

提示：
- await asyncio.sleep(n) 模拟 I/O 等待，让出控制权给其他协程
- 不要用 time.sleep()——它会阻塞整个线程，让异步并发失效
- asyncio.gather() 返回的结果顺序和传入参数的顺序一致，不是按完成顺序
- 使用 time.perf_counter() 做高精度计时
"""

import asyncio
import time

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义 async def wait_and_print(name: str, delay: float) -> str
# 2. 用 asyncio.gather() 并发运行 3 个协程（3s/2s/1s）
# 3. 用 time.perf_counter() 记录总耗时
# 4. 对比：并发耗时 ≈ 最长任务（3s），串行预计耗时 = 3+2+1 = 6s
# 5. 计算加速比：串行耗时 / 异步实际耗时
#
# 提示：
# - 务必用 await asyncio.sleep()，不要用 time.sleep()
# - gather 返回结果顺序 = 传入参数顺序（非完成顺序）
# - 对比同步和异步的执行时间，体会异步并发的优势
