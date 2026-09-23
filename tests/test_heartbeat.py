import unittest
import os
import sys
import json
import tempfile
import asyncio
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestHeartbeatPacemaker(unittest.TestCase):

    def setUp(self):
        """每个测试前创建临时任务文件"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
        self.original_tasks_file = None

        import nexusagent.core.config
        self.original_tasks_file = nexusagent.core.config.TASKS_FILE
        nexusagent.core.config.TASKS_FILE = self.temp_file.name

        import nexusagent.core.heartbeat
        nexusagent.core.heartbeat.TASKS_FILE = self.temp_file.name

    def tearDown(self):
        """每个测试后清理临时文件"""
        self.temp_file.close()
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

        import nexusagent.core.config
        nexusagent.core.config.TASKS_FILE = self.original_tasks_file

        import nexusagent.core.heartbeat
        nexusagent.core.heartbeat.TASKS_FILE = self.original_tasks_file

    def _write_tasks(self, tasks):
        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

    def _read_tasks(self):
        with open(self.temp_file.name, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return json.loads(content) if content else []

    def test_no_tasks_file(self):
        """任务文件不存在时不抛异常且无触发"""
        from nexusagent.core.heartbeat import process_due_tasks
        os.unlink(self.temp_file.name)
        self.assertEqual(process_due_tasks(), [])

    def test_empty_tasks_file(self):
        """空任务文件无触发"""
        from nexusagent.core.heartbeat import process_due_tasks
        with open(self.temp_file.name, 'w') as f:
            f.write("")
        self.assertEqual(process_due_tasks(), [])

    def test_task_not_yet_due(self):
        """未到时间的任务不会被触发，仍保留在文件中"""
        from nexusagent.core.heartbeat import process_due_tasks
        future_time = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        self._write_tasks([{
            "id": "task1",
            "target_time": future_time,
            "description": "未来任务",
            "repeat": None,
            "repeat_count": None
        }])

        triggered = process_due_tasks()
        self.assertEqual(triggered, [])
        self.assertEqual(len(self._read_tasks()), 1)

    def test_oneshot_due_triggered_and_removed(self):
        """单次到期任务触发后从文件移除"""
        from nexusagent.core.heartbeat import process_due_tasks
        past_time = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        self._write_tasks([{
            "id": "task1",
            "target_time": past_time,
            "description": "到期任务",
            "repeat": None,
            "repeat_count": None
        }])

        triggered = process_due_tasks()
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0]["description"], "到期任务")
        self.assertEqual(self._read_tasks(), [])

    def test_daily_overdue_catch_up_once(self):
        """过期多天的 daily 任务只触发一次，并一次追赶到未来"""
        from nexusagent.core.heartbeat import process_due_tasks
        now = datetime.now().replace(microsecond=0)
        overdue = now - timedelta(days=7)

        self._write_tasks([{
            "id": "wake",
            "target_time": overdue.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "叫你起床",
            "repeat": "daily",
            "repeat_count": None
        }])

        triggered = process_due_tasks(now=now)
        self.assertEqual(len(triggered), 1)

        remaining = self._read_tasks()
        self.assertEqual(len(remaining), 1)
        next_dt = datetime.strptime(remaining[0]["target_time"], "%Y-%m-%d %H:%M:%S")
        self.assertGreater(next_dt, now)
        # 应落在「下一个同刻」附近，而不是只 +1 天仍过期
        self.assertLessEqual(next_dt, now + timedelta(days=1, seconds=1))

        # 立刻再扫一轮不应重复触发
        triggered_again = process_due_tasks(now=now)
        self.assertEqual(triggered_again, [])

    def test_repeating_with_count_decrements_once(self):
        """有限次数循环：过期追赶只扣 1 次额度"""
        from nexusagent.core.heartbeat import process_due_tasks
        now = datetime.now().replace(microsecond=0)
        overdue = now - timedelta(days=5)

        self._write_tasks([{
            "id": "med",
            "target_time": overdue.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "吃药",
            "repeat": "daily",
            "repeat_count": 3
        }])

        process_due_tasks(now=now)
        remaining = self._read_tasks()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["repeat_count"], 2)

    def test_repeat_count_one_no_reschedule(self):
        """repeat_count=1 触发后不再续期"""
        from nexusagent.core.heartbeat import process_due_tasks
        now = datetime.now().replace(microsecond=0)
        overdue = now - timedelta(hours=1)

        self._write_tasks([{
            "id": "last",
            "target_time": overdue.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "最后一次",
            "repeat": "hourly",
            "repeat_count": 1
        }])

        triggered = process_due_tasks(now=now)
        self.assertEqual(len(triggered), 1)
        self.assertEqual(self._read_tasks(), [])

    def test_invalid_time_format_handled(self):
        """无效时间格式被跳过，不导致崩溃"""
        from nexusagent.core.heartbeat import process_due_tasks
        future_time = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        self._write_tasks([
            {
                "id": "bad",
                "target_time": "invalid-time-format",
                "description": "无效时间任务",
                "repeat": None,
                "repeat_count": None
            },
            {
                "id": "ok",
                "target_time": future_time,
                "description": "正常任务",
                "repeat": None,
                "repeat_count": None
            }
        ])

        triggered = process_due_tasks()
        self.assertEqual(triggered, [])
        remaining = self._read_tasks()
        # 没有 triggered 时不写回，原文件仍含两项
        self.assertEqual(len(remaining), 2)

    def test_multiple_tasks_mixed(self):
        """到期单次任务触发移除，未到期任务保留"""
        from nexusagent.core.heartbeat import process_due_tasks
        past_time = (datetime.now() - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        future_time = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        self._write_tasks([
            {
                "id": "task1",
                "target_time": past_time,
                "description": "已到期任务",
                "repeat": None,
                "repeat_count": None
            },
            {
                "id": "task2",
                "target_time": future_time,
                "description": "未到期任务",
                "repeat": "daily",
                "repeat_count": None
            }
        ])

        triggered = process_due_tasks()
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0]["id"], "task1")
        remaining = self._read_tasks()
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["id"], "task2")

    def test_heartbeat_message_contains_task_id(self):
        from nexusagent.core.heartbeat import format_heartbeat_message
        from nexusagent.core.bus import HEARTBEAT_PREFIX
        msg = format_heartbeat_message({"id": "abc123", "description": "吃饭"})
        self.assertTrue(msg.startswith(HEARTBEAT_PREFIX))
        self.assertIn("任务ID: abc123", msg)
        self.assertIn("吃饭", msg)


class TestBusHeartbeatPurge(unittest.TestCase):
    """取消任务后清理队列中的心跳消息"""

    def setUp(self):
        import nexusagent.core.bus as bus
        self.bus = bus
        bus._cancelled_task_ids.clear()
        bus._drain_requested.clear()
        while True:
            try:
                bus.task_queue.get_nowait()
                bus.task_queue.task_done()
            except asyncio.QueueEmpty:
                break

    def test_mark_and_detect_cancelled_heartbeat(self):
        self.bus.mark_tasks_cancelled({"dc6c1657"})
        hb = (
            f"{self.bus.HEARTBEAT_PREFIX}\n"
            f"任务ID: dc6c1657\n"
            f"你设定的定时任务已到期\n"
            f"任务内容：提醒你吃饭"
        )
        self.assertTrue(self.bus.is_cancelled_heartbeat(hb))
        self.assertFalse(self.bus.is_cancelled_heartbeat("下午好"))
        other = (
            f"{self.bus.HEARTBEAT_PREFIX}\n"
            f"任务ID: other999\n"
            f"任务内容：叫你起床"
        )
        self.assertFalse(self.bus.is_cancelled_heartbeat(other))

    def test_purge_removes_only_cancelled_heartbeats(self):
        async def run():
            await self.bus.task_queue.put("用户消息")
            await self.bus.task_queue.put(
                f"{self.bus.HEARTBEAT_PREFIX}\n任务ID: aaa\n任务内容：A"
            )
            await self.bus.task_queue.put(
                f"{self.bus.HEARTBEAT_PREFIX}\n任务ID: bbb\n任务内容：B"
            )
            self.bus.mark_tasks_cancelled({"aaa"})
            await self.bus.purge_cancelled_heartbeats()

            items = []
            while True:
                try:
                    items.append(self.bus.task_queue.get_nowait())
                    self.bus.task_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            return items

        items = asyncio.run(run())
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0], "用户消息")
        self.assertIn("任务ID: bbb", items[1])


if __name__ == '__main__':
    unittest.main()
