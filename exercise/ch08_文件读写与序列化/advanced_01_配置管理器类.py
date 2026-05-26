"""
章节：第8章 文件读写与序列化
题目：配置管理器类
类型：进阶练习

题目描述：
实现一个 `ConfigManager` 类，底层用 JSON 文件存储配置。功能要求：
- 创建时如果 JSON 文件不存在，自动用默认值创建
- 支持 config["key"] 读取配置（实现 __getitem__）
- 支持 config["key"] = value 修改配置并自动保存（实现 __setitem__）
- 支持 config.get("key", default) 安全取值

示例输入/输出：
    config = ConfigManager("app_config.json", defaults={"host": "localhost", "port": 8080})
    print(config["host"])              # localhost
    config["port"] = 9090              # 修改并自动保存
    print(config.get("debug", False))  # False
    print(config)                      # ConfigManager(app_config.json, keys=['host', 'port'])

提示：
- 使用 pathlib.Path 进行文件路径操作
- 使用 json.dumps() / json.loads() 进行序列化/反序列化
- 重写 __getitem__、__setitem__、__repr__ 让类更 Pythonic
"""

import json
from pathlib import Path


# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 ConfigManager 类的 __init__、_load、_save
# 2. 实现 __getitem__、__setitem__、get、__repr__
# 3. __setitem__ 中修改后自动调用 _save 持久化
#
# 提示：参考第8章 JSON 读写与魔术方法示例
#
# 完成后运行: python advanced_01_配置管理器类.py


class ConfigManager:
    """基于 JSON 文件的配置管理器"""

    def __init__(self, filepath, defaults=None):
        pass  # TODO: 实现构造方法

    def _load(self):
        """加载配置：文件存在则读取，不存在则用默认值创建"""
        pass  # TODO: 实现加载逻辑

    def _save(self):
        """保存配置到 JSON 文件"""
        pass  # TODO: 实现保存逻辑

    def __getitem__(self, key):
        """支持 config["key"] 读取"""
        pass  # TODO: 实现读取

    def __setitem__(self, key, value):
        """支持 config["key"] = value 修改并自动保存"""
        pass  # TODO: 实现设置

    def get(self, key, default=None):
        """安全取值，key 不存在时返回默认值"""
        pass  # TODO: 实现安全取值

    def __repr__(self):
        pass  # TODO: 实现字符串表示


if __name__ == "__main__":
    pass  # TODO: 编写测试代码
