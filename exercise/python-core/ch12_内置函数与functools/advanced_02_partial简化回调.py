"""
章节：第12章 内置函数与 functools
题目：用 partial 简化回调函数
类型：进阶练习

题目描述：
有一个通用日志函数 `log(level, message)`，用 `functools.partial` 创建三个专用日志函数：
- `log_info(message)`：固定 level 为 "INFO"
- `log_warning(message)`：固定 level 为 "WARNING"
- `log_error(message)`：固定 level 为 "ERROR"

要求：每条日志打印时自动带上时间戳，格式为 [HH:MM:SS] [LEVEL] message。

示例输入/输出：
    [14:30:22] [INFO] 程序启动
    [14:30:22] [WARNING] 磁盘空间已使用 80%
    [14:30:22] [ERROR] 数据库连接失败，请检查网络

提示：
- partial(func, arg1) 固定位置参数
- partial(func, keyword=value) 固定关键字参数
- partial 返回一个新函数，调用时只需传剩余参数
"""

from functools import partial
from datetime import datetime


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 log(level, message) 通用日志函数
# 2. 用 partial 创建 log_info、log_warning、log_error
# 3. 在时间戳中用 datetime.now().strftime()
#
# 提示：参考第12章 functools.partial 示例
#
# 完成后运行: python advanced_02_partial简化回调.py


def log(level, message):
    """通用日志函数"""
    pass  # TODO: 实现日志打印


# TODO: 用 partial 创建 log_info、log_warning、log_error


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
