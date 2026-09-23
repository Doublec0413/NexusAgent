import asyncio
import threading

HEARTBEAT_PREFIX = "【系统内部心跳触发】"

task_queue = asyncio.Queue()

# 已取消任务 ID：用于丢弃队列中尚未处理的心跳消息
_cancelled_task_ids: set[str] = set()
_cancelled_lock = threading.Lock()
_drain_requested = threading.Event()


async def emit_task(content: str):
    await task_queue.put(content)


def mark_tasks_cancelled(task_ids: set[str] | list[str]):
    """同步侧标记已取消任务，并请求清理队列中对应心跳。"""
    with _cancelled_lock:
        _cancelled_task_ids.update(task_ids)
    _drain_requested.set()


def is_cancelled_heartbeat(message: str) -> bool:
    """心跳消息是否对应已取消的任务。"""
    if not isinstance(message, str) or not message.startswith(HEARTBEAT_PREFIX):
        return False
    with _cancelled_lock:
        if not _cancelled_task_ids:
            return False
        for task_id in _cancelled_task_ids:
            if f"任务ID: {task_id}" in message:
                return True
    return False


async def purge_cancelled_heartbeats():
    """
    从 task_queue 中移除已取消任务对应的心跳消息，其它消息原样放回。
    仅在有新的取消标记时执行一次扫描。
    """
    if not _drain_requested.is_set():
        return
    _drain_requested.clear()

    with _cancelled_lock:
        if not _cancelled_task_ids:
            return
        cancelled = set(_cancelled_task_ids)

    kept: list[str] = []
    while True:
        try:
            item = task_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        # 与 get 配对，避免 join 计数泄漏；保留项稍后重新 put
        task_queue.task_done()

        if (
            isinstance(item, str)
            and item.startswith(HEARTBEAT_PREFIX)
            and any(f"任务ID: {tid}" in item for tid in cancelled)
        ):
            continue
        kept.append(item)

    for item in kept:
        await task_queue.put(item)
