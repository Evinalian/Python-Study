"""
练习 2-5（进阶）: 设计一个完整的工具权限系统

设计并实现一个完整的工具权限系统。

TODO:
1. 定义用户角色和权限：
   - admin: 可以使用所有工具
   - editor: 可以使用 read_only 和 read_write 工具
   - viewer: 只能使用 read_only 工具

2. 为每个工具分配权限组：
   - read_only: search_web, query_database, read_file
   - read_write: save_file, send_email, update_record
   - admin_only: delete_record, manage_users, system_config

3. 准备至少 6 个工具，分属三个权限组

4. 实现 PermissionManager 类：
   - __init__(role): 根据角色初始化
   - can_use(tool_name): 检查是否有权限使用某工具
   - filter_tools(all_tools): 从所有工具中过滤出当前角色可用的工具
   - 记录权限拒绝日志

5. 在 Agent 循环中集成权限检查：
   - 工具执行前验证当前用户是否有权限
   - 无权限时返回友好提示："您当前的权限级别(viewer)不足以使用工具'delete_record'。"
   - 不要在当前用户的可用工具列表中包含无权限的工具

6. 测试：
   - 用 viewer 角色尝试调用 delete_record
   - 用 editor 角色尝试调用 manage_users
   - 用 admin 角色调用所有工具
   - 验证 get_tools_for_role 返回的工具列表是否正确

7. 审计日志：打印所有权限拒绝事件，包含时间戳、用户角色、请求的工具名
"""


# 请在此处编写你的代码


if __name__ == "__main__":
    # TODO: 实现权限系统并测试
    pass
