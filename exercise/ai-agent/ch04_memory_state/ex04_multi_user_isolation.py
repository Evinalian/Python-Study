"""
练习 4-4（进阶）: 实现多用户的记忆隔离

当前实现的 VectorMemory 是一个全局的单例。修改它使其支持多用户记忆隔离。

TODO:
1. 重构 VectorMemory 类以支持多用户：

   a) 内部存储结构改为: self._user_memories: Dict[str, Dict[str, Memory]]
      - 第一层 key 是 user_id
      - 第二层 key 是 memory_id

   b) 所有方法添加 user_id 参数：
      - add(content, user_id, ...)
      - search(query, user_id, top_k=5, ...)
      - update(memory_id, user_id, ...)
      - forget(memory_id, user_id)
      - decay(user_id, ...)

   c) 添加用户级别的管理方法：
      - get_user_memory_count(user_id) -> int
      - clear_user_memories(user_id) -> int  (返回清除的记忆数量)
      - list_users() -> List[str]

   d) 所有检索操作只搜索当前用户的记忆空间（严格隔离）

2. 确保跨用户隔离：
   - 用户 A 的检索绝对不能返回用户 B 的记忆
   - 用户 A 不能修改或删除用户 B 的记忆
   - 添加一个隔离验证方法 verify_isolation(user_a, user_b) 来测试

3. 测试：
   a) 为 3 个不同用户分别添加 3-5 条不同的记忆
   b) 以每个用户的身份检索，验证只返回自己的记忆
   c) 尝试用 user_A 的身份更新 user_B 的记忆，验证被拒绝
   d) 调用 stats() 改为支持按用户统计，或返回全局统计
   e) 测试 clear_user_memories 是否只清除指定用户的记忆

4. 额外挑战：
   - 实现记忆的"共享"机制（某些记忆可以跨用户可见，如公共知识库）
   - 添加 shared 标记，shared=True 的记忆在检索时对所有用户可见
"""


# 请在此处编写你的代码


if __name__ == "__main__":
    # TODO: 实现多用户隔离并测试
    pass
