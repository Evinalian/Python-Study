"""
章节：第7章 异常处理与上下文管理器
题目：支持重试的文件读取
类型：进阶练习

题目描述：
实现函数 `read_file_with_retry(filepath, max_retries=3)`，读取指定文件内容。
如果读取失败（文件不存在、权限不足等），等待递增的时间后重试：
- 第 1 次重试等待 0.5 秒
- 第 2 次重试等待 1 秒
- 第 3 次重试等待 2 秒
全部失败后抛出带原始异常链的自定义异常 FileReadError。

示例输入/输出：
    第 1/3 次尝试失败: [Errno 2] No such file or directory: '不存在的文件.txt'
    第 2/3 次尝试失败: [Errno 2] No such file or directory: '不存在的文件.txt'
    第 3/3 次尝试失败: [Errno 2] No such file or directory: '不存在的文件.txt'
    最终失败: 文件读取失败，原始错误: ...

提示：
- 使用指数退避策略：wait = 0.5 * (2 ** (attempt - 1))
- 自定义异常类继承 Exception，通过 __cause__ 保留原始异常链
- 使用 raise ... from ... 抛出异常
"""

import time


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 定义 FileReadError 自定义异常类
# 2. 实现 read_file_with_retry 函数（指数退避重试）
# 3. 用 raise ... from ... 保留原始异常链
#
# 提示：参考第7章自定义异常和指数退避示例
#
# 完成后运行: python advanced_01_重试文件读取.py


class FileReadError(Exception):
    """文件读取失败的自定义异常"""
    pass  # TODO: 实现异常类


def read_file_with_retry(filepath, max_retries=3):
    """带重试机制的文件读取"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
