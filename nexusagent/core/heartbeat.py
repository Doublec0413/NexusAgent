import os
import json
import asyncio
import calendar
from datetime import datetime, timedelta
from .config import TASKS_FILE
from .tools.schedule_tools import tasks_lock
from .bus import HEARTBEAT_PREFIX

def _advance_once(target_dt: datetime, repeat_freq: str) -> datetime | None:
    """按频率将时间推进一个周期。"""
    if repeat_freq == "hourly":
        return target_dt + timedelta(hours=1)
    if repeat_freq == "daily":
        return target_dt + timedelta(days=1)
    if repeat_freq == "weekly":
        return target_dt + timedelta(days=7)
    if repeat_freq == "monthly":
        month = target_dt.month + 1
        year = target_dt.year
        if month > 12:
            month = 1
            year += 1
        last_day = calendar.monthrange(year, month)[1]
        day = min(target_dt.day, last_day)
        return target_dt.replace(year=year, month=month, day=day)
    return None


def _catch_up_to_future(
    target_dt: datetime,
    repeat_freq: str,
    now: datetime,
    repeat_count: int | None,
) -> tuple[datetime, int | None] | None:
    """
    过期循环任务一次追赶到未来。

    只消耗 1 次触发额度（不会因漏跑多天而连扣多次），
    并将下次时间跳到第一个严格晚于 now 的周期。
    若次数耗尽则返回 None（触发后不再续期）。
    """
    if repeat_count is not None and repeat_count <= 1:
        return None

    next_count = (repeat_count - 1) if repeat_count is not None else None
    next_dt = _advance_once(target_dt, repeat_freq)
    if next_dt is None:
        return None

    # 一次跳过所有已过期周期，避免每 10s 只推一天导致刷屏
    safety = 0
    while next_dt <= now and safety < 100000:
        advanced = _advance_once(next_dt, repeat_freq)
        if advanced is None:
            return None
        next_dt = advanced
        safety += 1

    return next_dt, next_count


def process_due_tasks(now: datetime | None = None) -> list[dict]:
    """
    扫描任务文件，返回本轮应触发的任务，并写回未到期/已续期任务。
    供 pacemaker_loop 与单测直接调用。
    """
    if now is None:
        now = datetime.now()

    pending_tasks = []
    triggered_tasks = []

    with tasks_lock:
        if not os.path.exists(TASKS_FILE):
            return []

        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                tasks = json.loads(content)
        except Exception:
            return []

        if not tasks:
            return []

        for t in tasks:
            try:
                target_dt = datetime.strptime(t["target_time"], "%Y-%m-%d %H:%M:%S")
                if now < target_dt:
                    pending_tasks.append(t)
                    continue

                triggered_tasks.append(t)

                repeat_freq = t.get("repeat")
                if not repeat_freq:
                    continue

                caught = _catch_up_to_future(
                    target_dt, repeat_freq, now, t.get("repeat_count")
                )
                if caught is None:
                    continue

                next_dt, next_count = caught
                renewed = dict(t)
                renewed["target_time"] = next_dt.strftime("%Y-%m-%d %H:%M:%S")
                renewed["repeat_count"] = next_count
                pending_tasks.append(renewed)
            except Exception:
                pass

        if triggered_tasks:
            try:
                with open(TASKS_FILE, "w", encoding="utf-8") as f:
                    json.dump(pending_tasks, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    return triggered_tasks


def format_heartbeat_message(task: dict) -> str:
    return (
        f"{HEARTBEAT_PREFIX}\n"
        f"任务ID: {task.get('id', '')}\n"
        f"你设定的定时任务已到期，请立即主动提醒用户或执行动作。\n"
        f"任务内容：{task['description']}"
    )


async def pacemaker_loop(task_queue: asyncio.Queue, check_interval: int = 10):
    """
    后台心脏起搏器协程（带并发锁和循环任务续期功能）
    """
    while True:
        await asyncio.sleep(check_interval)
        triggered_tasks = process_due_tasks()
        for t in triggered_tasks:
            await task_queue.put(format_heartbeat_message(t))
