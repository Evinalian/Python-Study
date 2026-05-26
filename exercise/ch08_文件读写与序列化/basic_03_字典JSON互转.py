"""
章节：第8章 文件读写与序列化
题目：字典 JSON 互转
类型：基础练习

题目描述：
把一个字典存为 JSON 文件，再读取回来验证内容一致。
要求：
- 使用 json.dump() 写入，json.load() 读取
- 写入时使用 ensure_ascii=False 保证中文正常显示
- 使用 indent=2 美化输出
- 最后验证原始数据和读取数据是否完全一致

示例输入/输出：
    原始数据: {'书名': 'Python 入门', '价格': 59.9, ...}
    读取数据: {'书名': 'Python 入门', '价格': 59.9, ...}
    内容一致: True

提示：
- dumps/loads 操作字符串，dump/load 操作文件
- ensure_ascii=False 防止中文被转成 \uXXXX
"""

import json


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 main() 函数
# 2. 用 json.dump() 写入，json.load() 读取
# 3. 使用 ensure_ascii=False 和 indent=2
#
# 提示：参考第8章 JSON 序列化示例
#
# 完成后运行: python basic_03_字典JSON互转.py


def main():
    """字典 JSON 互转的主函数"""
    pass  # TODO: 实现函数体


if __name__ == "__main__":
    main()
