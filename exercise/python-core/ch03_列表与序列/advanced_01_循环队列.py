"""
章节：第3章 列表与序列
题目：实现一个循环队列（用定长 list）
类型：进阶练习

题目描述：
用定长 list 实现一个循环队列类 CircularQueue，支持：
- __init__(self, capacity) 初始化容量
- enqueue(item) 入队（队满时覆盖最老的元素）
- dequeue() 出队（队空时返回 None）
- is_empty() 判断是否为空
- is_full() 判断是否已满

示例用法：
    q = CircularQueue(3)
    q.enqueue(1); q.enqueue(2); q.enqueue(3)
    print(q)           # CircularQueue([1, 2, 3])
    print(q.is_full()) # True
    q.enqueue(4)       # 队满，1 被覆盖
    print(q)           # CircularQueue([2, 3, 4])

提示：
- 使用 self._head 和 self._tail 两个指针
- 移动指针使用取模运算 (index + 1) % capacity
- 需要跟踪当前元素个数 self._size
"""

# ========== TODO：请完成以下练习 ==========
#
# 步骤指引：
# 1. 实现 CircularQueue 类的 __init__
# 2. 实现 enqueue 和 dequeue，用取模运算维护头尾指针
# 3. 实现 is_empty、is_full 和 __repr__
# 4. 测试：创建容量 3 的队列，测试满覆盖和空出队
#
# 提示：
# - _head 指向队头，_tail 指向下一个空位
# - (index + 1) % capacity 实现循环
#
# 完成后，运行 python advanced_01_循环队列.py 测试你的代码。


class CircularQueue:
    """用定长 list 实现的循环队列"""

    def __init__(self, capacity):
        # TODO: 初始化 _data, _capacity, _head, _tail, _size
        pass

    def enqueue(self, item):
        """入队：队满时覆盖最老的元素"""
        pass  # TODO: 实现入队

    def dequeue(self):
        """出队：返回队头元素，队空返回 None"""
        pass  # TODO: 实现出队

    def is_empty(self):
        """判断队列是否为空"""
        pass  # TODO: 实现判空

    def is_full(self):
        """判断队列是否已满"""
        pass  # TODO: 实现判满

    def __repr__(self):
        """方便查看当前状态"""
        pass  # TODO: 实现字符串表示
