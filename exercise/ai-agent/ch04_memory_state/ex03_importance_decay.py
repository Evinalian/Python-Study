"""
练习 4-3: 实现简单的重要性衰减

为 VectorMemory 系统添加基于时间的记忆重要性衰减机制。

TODO:
1. 为 VectorMemory 类添加以下新方法：

   a) boost_on_access(self, memory_id: str, boost: float = 0.05)
      - 当一条记忆被 search() 检索并返回时，自动调用此方法
      - 提升该记忆的重要性（但不超过 1.0）
      - 同时更新 last_accessed_at 时间戳

   b) decay_by_age(self, max_age_days: float = 7.0, decay_amount: float = 0.1)
      - 对所有记忆执行衰减检查
      - 如果记忆的 last_accessed_at 距今超过 max_age_days 天
      - 降低其重要性（decay_amount）
      - 注意：用户画像类记忆（memory_type="fact" 且 tags 包含 "user_profile"）
        应该衰减得更慢（decay_amount * 0.3），因为这类信息更持久

   c) purge_low_importance(self, threshold: float = 0.05)
      - 删除所有重要性低于 threshold 的记忆
      - 返回被删除的记忆数量

2. 修改 search() 方法，使其在返回结果后自动调用 boost_on_access()

3. 测试流程：
   a) 创建 10 条记忆，重要性初始化为 0.3-0.8
   b) 模拟 3 次不同主题的检索（如"编程"、"咖啡"、"天气"）
      - 编程相关的记忆会被检索到并boost
      - 咖啡相关的记忆会被检索到并boost
      - 天气相关的记忆（可能不存在）不会被boost
   c) 调用 decay_by_age()（模拟 7 天过去）
   d) 调用 purge_low_importance()
   e) 检查哪些记忆被保留，哪些被删除

4. 分析结果（作为注释或输出）：
   - 被频繁访问的"编程"和"咖啡"相关记忆是否比"天气"相关的记忆存活更久？
   - 衰减参数（threshold, decay_amount）应该怎么设置？太激进或太保守各有什么问题？
   - 在实际项目中，应该多久执行一次 decay_by_age？
"""

import time  # noqa


# 请在此处编写你的代码


if __name__ == "__main__":
    # TODO: 实现衰减机制并测试
    pass
