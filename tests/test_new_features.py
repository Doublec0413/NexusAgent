"""
NexusAgent 新功能综合测试套件
测试覆盖：多Agent编排、RAG知识库、安全评分仪表盘
"""
import unittest
import os
import sys
import time
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestRAGEngine(unittest.TestCase):
    """RAG 知识库引擎测试"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.knowledge_dir = os.path.join(self.test_dir, "knowledge")
        os.makedirs(self.knowledge_dir, exist_ok=True)
        os.environ["NEXUSAGENT_WORKSPACE"] = self.test_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_tfidf_tokenize(self):
        """测试中英文混合分词"""
        from nexusagent.core.rag_engine import TFIDFEngine
        engine = TFIDFEngine()
        tokens = engine._tokenize("Hello 你好 Python")
        self.assertIn("hello", tokens)
        self.assertIn("python", tokens)
        self.assertIn("你", tokens)
        self.assertIn("好", tokens)

    def test_tfidf_cosine_similarity(self):
        """测试余弦相似度计算"""
        from nexusagent.core.rag_engine import TFIDFEngine
        vec_a = {"hello": 1.0, "world": 0.5}
        vec_b = {"hello": 0.8, "world": 0.3}
        vec_c = {"foo": 1.0, "bar": 0.5}

        sim_ab = TFIDFEngine._cosine_similarity(vec_a, vec_b)
        sim_ac = TFIDFEngine._cosine_similarity(vec_a, vec_c)

        self.assertGreater(sim_ab, 0.9)  # 相似向量
        self.assertEqual(sim_ac, 0.0)    # 不相似向量

    def test_document_chunking(self):
        """测试文档切片"""
        from nexusagent.core.rag_engine import RAGKnowledgeBase
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)

        long_text = "这是一段测试文本。" * 200
        chunks = kb._split_text(long_text, "test.txt", chunk_size=100, overlap=20)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), 120)  # 允许边界裕量
            self.assertEqual(chunk.source_file, "test.txt")

    def test_add_and_search_document(self):
        """测试文档导入与检索"""
        from nexusagent.core.rag_engine import RAGKnowledgeBase
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)

        # 创建测试文档
        doc_path = os.path.join(self.knowledge_dir, "test_doc.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("Python是一种高级编程语言，广泛用于数据科学和人工智能领域。\n"
                    "JavaScript是Web开发的核心语言，常用于前端和后端开发。\n"
                    "Rust是一种系统级编程语言，以安全性和性能著称。")

        count = kb.add_document(doc_path)
        self.assertGreater(count, 0)

        results = kb.search("Python 数据科学", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0].score, 0)

    def test_document_dedup(self):
        """测试文档去重"""
        from nexusagent.core.rag_engine import RAGKnowledgeBase
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)

        doc_path = os.path.join(self.knowledge_dir, "dup_test.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("重复测试内容")

        count1 = kb.add_document(doc_path)
        count2 = kb.add_document(doc_path)  # 重复导入

        self.assertGreater(count1, 0)
        self.assertEqual(count2, 0)  # 去重，不应新增

    def test_build_context(self):
        """测试 RAG 上下文构建"""
        from nexusagent.core.rag_engine import RAGKnowledgeBase
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)

        doc_path = os.path.join(self.knowledge_dir, "context_test.txt")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write("NexusAgent是一个多Agent协作的智能体框架，支持RAG知识增强。")

        kb.add_document(doc_path)
        context = kb.build_context("什么是NexusAgent")
        self.assertIn("知识库检索结果", context)

    def test_get_stats(self):
        """测试统计信息"""
        from nexusagent.core.rag_engine import RAGKnowledgeBase
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)

        stats = kb.get_stats()
        self.assertIn("total_chunks", stats)
        self.assertIn("total_documents", stats)

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.knowledge_dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_directory_scan_indexes_existing_files(self):
        """建库前放入的文件，不调用导入也能检索。"""
        from nexusagent.core.rag_engine import RAGKnowledgeBase

        self._write("auto.txt", "自动扫描可以找到量子纠缠的说明。")
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)

        results = kb.search("量子纠缠", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk.source_file, "auto.txt")

    def test_reload_uses_saved_term_frequencies(self):
        """重启后直接读出词频和 IDF，未改动文件的词频保持不变。"""
        import json
        from nexusagent.core.rag_engine import RAGKnowledgeBase

        self._write("a.txt", "苹果派的配方保持不变，面粉和黄油的比例不要改。")
        self._write("b.txt", "香蕉昔的旧配方只有香蕉和牛奶。")
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)
        first = kb.search("苹果派", top_k=1)
        self.assertGreater(len(first), 0)

        with open(kb.index_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        tf_before = next(item["tf"] for item in saved["chunks"] if item["source_file"] == "a.txt")
        self.assertIn("idf", saved)
        self.assertNotIn("tfidf_vector", saved["chunks"][0])
        self.assertEqual(saved["doc_count"], len(saved["chunks"]))

        self._write("b.txt", "香蕉昔的新配方加入肉桂。")
        reloaded = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)
        with open(reloaded.index_path, "r", encoding="utf-8") as f:
            updated = json.load(f)
        tf_after = next(item["tf"] for item in updated["chunks"] if item["source_file"] == "a.txt")
        self.assertEqual(tf_before, tf_after)
        self.assertTrue(any("肉桂" in chunk.content for chunk in reloaded.chunks))
        self.assertTrue(all("旧配方" not in chunk.content for chunk in reloaded.chunks))
        again = reloaded.search("苹果派", top_k=1)
        self.assertGreater(len(again), 0)

    def test_changed_file_replaces_old_chunks(self):
        """同一路径内容变化后，只留下新切片。"""
        from nexusagent.core.rag_engine import RAGKnowledgeBase

        path = self._write("note.txt", "旧结论是火星有运河。")
        RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)
        self._write("note.txt", "新结论是火星有冰盖。")
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)

        self.assertTrue(all("运河" not in chunk.content for chunk in kb.chunks))
        self.assertTrue(any("冰盖" in chunk.content for chunk in kb.chunks))
        self.assertEqual(kb.add_document(path), 0)

    def test_deleted_file_drops_chunks(self):
        """目录里删除的文件，下次建库时切片一并移除。"""
        from nexusagent.core.rag_engine import RAGKnowledgeBase

        path = self._write("gone.txt", "独有词苜蓿草只出现在这份文件。")
        RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)
        os.remove(path)
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)

        self.assertEqual(kb.chunks, [])
        self.assertEqual(kb.search("苜蓿草"), [])

    def test_same_content_different_paths_are_both_kept(self):
        """去重按路径。两份相同正文都会留下。"""
        from nexusagent.core.rag_engine import RAGKnowledgeBase

        text = "两份相同正文都应该能被检索到。"
        self._write("a.txt", text)
        self._write("b.txt", text)
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)

        self.assertEqual(set(kb._files), {"a.txt", "b.txt"})
        self.assertEqual({chunk.source_file for chunk in kb.chunks}, {"a.txt", "b.txt"})

    def test_legacy_fingerprint_file_is_rebuilt_from_disk(self):
        """只有 MD5、没有切片的旧索引不会挡住重建。"""
        import json
        from nexusagent.core.rag_engine import RAGKnowledgeBase

        self._write("legacy.txt", "旧索引里没有正文，磁盘上的原文仍可检索。")
        with open(os.path.join(self.knowledge_dir, "_index.json"), "w", encoding="utf-8") as f:
            json.dump({"hashes": ["abc"], "chunk_count": 3}, f)

        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)
        self.assertTrue(any("旧索引" in chunk.content for chunk in kb.chunks))
        self.assertNotIn("_index.json", {chunk.source_file for chunk in kb.chunks})

    def test_nested_file_uses_relative_source(self):
        """子目录文件用相对路径区分同名文件。"""
        from nexusagent.core.rag_engine import RAGKnowledgeBase

        self._write(os.path.join("sub", "a.txt"), "嵌套文档里的关键词是琥珀。")
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)
        self.assertEqual(kb.chunks[0].source_file, "sub/a.txt")

    def test_external_file_updates_and_disappears(self):
        """目录外的文件按绝对路径更新；文件删除后切片也删除。"""
        from nexusagent.core.rag_engine import RAGKnowledgeBase

        outside = os.path.join(self.test_dir, "outside.txt")
        with open(outside, "w", encoding="utf-8") as f:
            f.write("外部旧词麒麟。")
        kb = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)
        self.assertGreater(kb.add_document(outside), 0)

        with open(outside, "w", encoding="utf-8") as f:
            f.write("外部新词凤凰。")
        self.assertGreater(kb.add_document(outside), 0)
        self.assertTrue(all("麒麟" not in chunk.content for chunk in kb.chunks))

        reloaded = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)
        self.assertTrue(any("凤凰" in chunk.content for chunk in reloaded.chunks))
        os.remove(outside)
        after_delete = RAGKnowledgeBase(knowledge_dir=self.knowledge_dir)
        self.assertTrue(all("凤凰" not in chunk.content for chunk in after_delete.chunks))


class TestSecurityDashboard(unittest.TestCase):
    """安全评分仪表盘测试"""

    def test_initial_state(self):
        """测试初始安全状态"""
        from nexusagent.core.security_scorer import SecurityScorer
        scorer = SecurityScorer()
        data = scorer.get_dashboard_data()

        self.assertEqual(data["risk_score"], 0)
        self.assertEqual(data["total_tool_calls"], 0)
        self.assertIn("安全", data["risk_level"])

    def test_risk_scoring(self):
        """测试风险评分计算"""
        from nexusagent.core.security_scorer import SecurityScorer
        scorer = SecurityScorer()

        # 低风险操作
        scorer.record_tool_call("get_current_time")
        data1 = scorer.get_dashboard_data()

        # 高风险操作
        scorer.record_tool_call("execute_office_shell")
        data2 = scorer.get_dashboard_data()

        self.assertGreater(data2["risk_score"], data1["risk_score"])

    def test_conversation_decay(self):
        """测试对话衰减机制"""
        from nexusagent.core.security_scorer import SecurityScorer
        scorer = SecurityScorer()

        scorer.record_tool_call("execute_office_shell")
        high_score = scorer.get_dashboard_data()["risk_score"]

        scorer.record_conversation()
        decayed_score = scorer.get_dashboard_data()["risk_score"]

        self.assertLessEqual(decayed_score, high_score)

    def test_conversation_decay_persists_after_tool_call(self):
        """对话衰减写入窗口后，后续工具调用不应把分数弹回衰减前水平"""
        from nexusagent.core.security_scorer import SecurityScorer
        scorer = SecurityScorer()

        scorer.record_tool_call("execute_office_shell")
        high_score = scorer.get_dashboard_data()["risk_score"]

        for _ in range(3):
            scorer.record_conversation()
        decayed_score = scorer.get_dashboard_data()["risk_score"]

        scorer.record_tool_call("calculator")
        after_tool_score = scorer.get_dashboard_data()["risk_score"]

        self.assertLess(decayed_score, high_score)
        self.assertLess(after_tool_score, high_score)

    def test_consecutive_tool_detection(self):
        """测试连续工具调用检测"""
        from nexusagent.core.security_scorer import SecurityScorer
        scorer = SecurityScorer()

        for _ in range(8):
            scorer.record_tool_call("write_office_file")

        data = scorer.get_dashboard_data()
        self.assertGreater(data["risk_score"], 0)
        self.assertGreater(len(data["recent_alerts"]), 0)

    def test_risk_levels(self):
        """测试风险等级划分"""
        from nexusagent.core.security_scorer import SecurityScorer
        scorer = SecurityScorer()

        level, _ = scorer.get_risk_level()
        self.assertIn("安全", level)

    def test_report_generation(self):
        """测试安全报告生成"""
        from nexusagent.core.security_scorer import SecurityScorer
        scorer = SecurityScorer()

        scorer.record_tool_call("execute_office_shell")
        scorer.record_tool_call("write_office_file")
        scorer.record_tool_call("get_current_time")

        report = scorer.generate_report()
        self.assertIn("安全审计报告", report)
        self.assertIn("风险概览", report)
        self.assertIn("工具调用分布", report)

    def test_tool_distribution(self):
        """测试工具调用分布统计"""
        from nexusagent.core.security_scorer import SecurityScorer
        scorer = SecurityScorer()

        scorer.record_tool_call("calculator")
        scorer.record_tool_call("calculator")
        scorer.record_tool_call("get_current_time")

        data = scorer.get_dashboard_data()
        self.assertEqual(data["tool_distribution"]["calculator"], 2)
        self.assertEqual(data["tool_distribution"]["get_current_time"], 1)

    def test_blocked_tool_result_detection(self):
        """测试沙盒拦截结果识别与被拦截计数"""
        from nexusagent.core.security_scorer import (
            SecurityScorer,
            is_blocked_tool_result,
        )

        self.assertTrue(is_blocked_tool_result("❌ 权限拒绝：检测到危险的目录跳转指令。"))
        self.assertTrue(is_blocked_tool_result("❌ 越权拦截：你试图访问沙盒外的路径"))
        self.assertFalse(is_blocked_tool_result(" ● 成功以 覆盖/新建 模式写入文件"))

        scorer = SecurityScorer()
        scorer.record_tool_call("execute_office_shell", success=False)
        self.assertEqual(scorer.get_dashboard_data()["blocked_operations"], 1)

    def test_blocked_operation_penalty(self):
        """被拦截操作应比同等成功操作风险更高"""
        from nexusagent.core.security_scorer import SecurityScorer

        ok_scorer = SecurityScorer()
        blocked_scorer = SecurityScorer()
        ok_scorer.record_tool_call("execute_office_shell", success=True)
        blocked_scorer.record_tool_call("execute_office_shell", success=False)

        self.assertGreater(
            blocked_scorer.get_dashboard_data()["risk_score"],
            ok_scorer.get_dashboard_data()["risk_score"],
        )

    def test_idle_decay(self):
        """测试无操作空闲衰减"""
        from nexusagent.core.security_scorer import (
            CONVERSATION_DECAY,
            RISK_DECAY_INTERVAL,
            SecurityScorer,
        )

        scorer = SecurityScorer()
        scorer.record_tool_call("execute_office_shell")
        high_score = scorer.get_dashboard_data()["risk_score"]

        scorer._last_activity_time -= RISK_DECAY_INTERVAL + 1
        decayed_score = scorer.get_dashboard_data()["risk_score"]

        self.assertLess(decayed_score, high_score)
        self.assertLessEqual(decayed_score, high_score + CONVERSATION_DECAY)

    def test_policy_refusal_detection(self):
        """测试提示词层协议拒绝识别"""
        from nexusagent.core.sandbox_protocol import (
            detect_jailbreak_intent,
            is_policy_refusal,
            is_refusal_response,
            should_record_policy_violation,
        )

        refusal = (
            "系统拦截：该操作违反 NexusAgent 核心安全协议。"
            "你只能在受限的 office 工位目录内操作文件。"
        )
        self.assertTrue(is_policy_refusal(refusal))
        self.assertTrue(is_refusal_response(refusal))
        self.assertFalse(is_policy_refusal("好的，我来帮你在 office 里创建文件。"))

        user_msg = "在/Users/jan/Desktop/NexusAgent/nexusagent/core/tools文件夹里新增一个代码生成工具"
        self.assertTrue(detect_jailbreak_intent(user_msg))

        alt_refusal = "抱歉，这个路径在沙盒外，我无法访问。请在 office 工位目录内操作。"
        self.assertTrue(should_record_policy_violation(user_msg, alt_refusal))

        redirect = "这个路径不行，但我可以帮你在 office 里创建一个类似的工具文件。"
        self.assertFalse(should_record_policy_violation(user_msg, redirect))

    def test_policy_violation_scoring(self):
        """提示词层协议拒绝应计入风险分与被拦截次数"""
        from nexusagent.core.security_scorer import (
            POLICY_VIOLATION_PENALTY,
            SecurityScorer,
        )

        scorer = SecurityScorer()
        scorer.record_policy_violation()
        data = scorer.get_dashboard_data()

        self.assertEqual(data["risk_score"], float(POLICY_VIOLATION_PENALTY))
        self.assertEqual(data["blocked_operations"], 1)
        self.assertEqual(data["total_tool_calls"], 0)
        self.assertGreater(len(data["recent_alerts"]), 0)

    def test_jailbreak_intent_without_standard_refusal(self):
        """用户越权且模型非标准拒绝措辞时也应计分"""
        from nexusagent.core.sandbox_protocol import should_record_policy_violation
        from nexusagent.core.security_scorer import SecurityScorer

        user_msg = "在/Users/jan/Desktop/NexusAgent/nexusagent/core/tools文件夹里新增工具"
        response = "不行，那个目录不在我的权限范围内，没法帮你写文件。"

        self.assertTrue(should_record_policy_violation(user_msg, response))

        scorer = SecurityScorer()
        scorer.record_policy_violation()
        self.assertGreater(scorer.get_dashboard_data()["risk_score"], 0)


class TestMultiAgentOrchestrator(unittest.TestCase):
    """多Agent编排器测试（无需LLM调用的结构测试）"""

    def test_subtask_creation(self):
        """测试子任务数据结构"""
        from nexusagent.core.multi_agent import SubTask, TaskStatus

        task = SubTask(description="测试任务")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.retry_count, 0)
        self.assertEqual(len(task.id), 8)

    def test_orchestration_result(self):
        """测试编排结果数据结构"""
        from nexusagent.core.multi_agent import OrchestrationResult

        result = OrchestrationResult(
            task_id="test123",
            subtasks=[],
            final_answer="测试完成",
            total_time=5.0,
            success_rate=100.0,
            subtask_count=0
        )
        self.assertEqual(result.task_id, "test123")
        self.assertEqual(result.success_rate, 100.0)

    def test_task_status_enum(self):
        """测试任务状态枚举"""
        from nexusagent.core.multi_agent import TaskStatus

        self.assertEqual(TaskStatus.PENDING.value, "pending")
        self.assertEqual(TaskStatus.RUNNING.value, "running")
        self.assertEqual(TaskStatus.COMPLETED.value, "completed")
        self.assertEqual(TaskStatus.FAILED.value, "failed")

    def test_plan_pairs_task_with_deps(self):
        """依赖跟任务配对，短行、重复序号、空描述、自依赖都不会连错边"""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch
        from nexusagent.core.multi_agent import MultiAgentOrchestrator

        content = """好的，我来拆解：
SUBTASK|x
SUBTASK|1|做A|2
SUBTASK|1|重复的A|none
SUBTASK|4||none
SUBTASK|2|做B|none
SUBTASK|3|做C|1,3,9
"""
        fake_response = MagicMock()
        fake_response.content = content
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value=fake_response)

        with patch("nexusagent.core.multi_agent.get_provider", return_value=fake_llm):
            orchestrator = MultiAgentOrchestrator()
            subtasks = asyncio.run(orchestrator._plan("任务", "test"))

        self.assertEqual([t.description for t in subtasks], ["做A", "做B", "做C"])
        by_desc = {t.description: t for t in subtasks}
        self.assertEqual(by_desc["做A"].depends_on, [by_desc["做B"].id])
        self.assertEqual(by_desc["做B"].depends_on, [])
        self.assertEqual(by_desc["做C"].depends_on, [by_desc["做A"].id])

    def test_failed_dependency_skips_successors(self):
        """前置失败时跳过直接和传递后继，无关任务照常执行"""
        import asyncio
        from unittest.mock import MagicMock, patch
        from nexusagent.core.multi_agent import MultiAgentOrchestrator, SubTask, TaskStatus

        with patch("nexusagent.core.multi_agent.get_provider", return_value=MagicMock()):
            orchestrator = MultiAgentOrchestrator()

        root = SubTask(description="调研市场份额")
        independent = SubTask(description="对比性能")
        child = SubTask(description="给出选型建议", depends_on=[root.id, independent.id])
        grandchild = SubTask(description="写成报告", depends_on=[child.id])
        executed = []

        async def fake_execute(task, task_map, thread_id):
            executed.append(task.description)
            if task.description == "调研市场份额":
                task.status = TaskStatus.FAILED
                task.result = "执行失败（重试2次）：Connection timeout"
            else:
                task.status = TaskStatus.COMPLETED
                task.result = "完成"

        orchestrator._execute_single = fake_execute
        asyncio.run(orchestrator._execute_dag(
            [root, independent, child, grandchild], "test"
        ))

        self.assertCountEqual(executed, ["调研市场份额", "对比性能"])
        self.assertEqual(child.status, TaskStatus.FAILED)
        self.assertEqual(child.result, "前置任务失败，已跳过")
        self.assertEqual(grandchild.status, TaskStatus.FAILED)
        self.assertEqual(grandchild.result, "前置任务失败，已跳过")
        self.assertEqual(independent.status, TaskStatus.COMPLETED)

    def test_successful_dependency_still_runs(self):
        """前置成功时后继照常执行"""
        import asyncio
        from unittest.mock import MagicMock, patch
        from nexusagent.core.multi_agent import MultiAgentOrchestrator, SubTask, TaskStatus

        with patch("nexusagent.core.multi_agent.get_provider", return_value=MagicMock()):
            orchestrator = MultiAgentOrchestrator()

        first = SubTask(description="做A")
        second = SubTask(description="做B", depends_on=[first.id])
        executed = []

        async def fake_execute(task, task_map, thread_id):
            executed.append(task.description)
            task.status = TaskStatus.COMPLETED
            task.result = "完成"

        orchestrator._execute_single = fake_execute
        asyncio.run(orchestrator._execute_dag([first, second], "test"))

        self.assertEqual(executed, ["做A", "做B"])
        self.assertEqual(second.status, TaskStatus.COMPLETED)

    def test_dependency_cycle_marks_deadlock(self):
        """循环依赖仍标为死锁，不会当成前置失败跳过"""
        import asyncio
        from unittest.mock import MagicMock, patch
        from nexusagent.core.multi_agent import MultiAgentOrchestrator, SubTask, TaskStatus

        with patch("nexusagent.core.multi_agent.get_provider", return_value=MagicMock()):
            orchestrator = MultiAgentOrchestrator()

        left = SubTask(description="做A")
        right = SubTask(description="做B")
        left.depends_on = [right.id]
        right.depends_on = [left.id]
        executed = []

        async def fake_execute(task, task_map, thread_id):
            executed.append(task.description)

        orchestrator._execute_single = fake_execute
        asyncio.run(orchestrator._execute_dag([left, right], "test"))

        self.assertEqual(executed, [])
        self.assertEqual(left.status, TaskStatus.FAILED)
        self.assertEqual(right.status, TaskStatus.FAILED)
        self.assertEqual(left.result, "依赖死锁：前置任务未完成")
        self.assertEqual(right.result, "依赖死锁：前置任务未完成")

    def test_orchestrate_emits_progress_while_running(self):
        """流水线每个阶段都推进度，主界面不用等全部结束才有输出"""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch
        from nexusagent.core.multi_agent import MultiAgentOrchestrator

        plan = MagicMock(content="SUBTASK|1|做A|none\nSUBTASK|2|做B|1\n")
        worker = MagicMock(content="结果")
        review = MagicMock(content="最终答案")
        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(side_effect=[plan, worker, worker, review])
        events = []

        with patch("nexusagent.core.multi_agent.get_provider", return_value=fake_llm), \
             patch("nexusagent.core.multi_agent.get_stream_writer", return_value=events.append):
            orchestrator = MultiAgentOrchestrator()
            result = asyncio.run(orchestrator.orchestrate("对比两个框架", "test"))

        self.assertEqual(result.final_answer, "最终答案")
        self.assertEqual(
            [(event["phase"], event.get("task")) for event in events],
            [
                ("planning", None),
                ("planned", None),
                ("worker_start", "做A"),
                ("worker_done", "做A"),
                ("worker_start", "做B"),
                ("worker_done", "做B"),
                ("reviewing", None),
            ],
        )
        self.assertEqual(events[1]["tasks"], ["做A", "做B"])
        self.assertEqual(events[3]["status"], "completed")
        self.assertIsNone(orchestrator._writer)

    def test_orchestrate_progress_is_optional(self):
        """没有流式上下文，或推送失败时，流水线仍返回最终答案"""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch
        from nexusagent.core.multi_agent import MultiAgentOrchestrator

        def llm():
            plan = MagicMock(content="SUBTASK|1|做A|none\n")
            worker = MagicMock(content="结果")
            review = MagicMock(content="最终答案")
            fake = MagicMock()
            fake.ainvoke = AsyncMock(side_effect=[plan, worker, review])
            return fake

        with patch("nexusagent.core.multi_agent.get_provider", return_value=llm()), \
             patch("nexusagent.core.multi_agent.get_stream_writer", side_effect=RuntimeError("no stream")):
            orchestrator = MultiAgentOrchestrator()
            result = asyncio.run(orchestrator.orchestrate("做一件事", "test"))
        self.assertEqual(result.final_answer, "最终答案")

        def boom(_event):
            raise RuntimeError("stream closed")

        with patch("nexusagent.core.multi_agent.get_provider", return_value=llm()), \
             patch("nexusagent.core.multi_agent.get_stream_writer", return_value=boom):
            orchestrator = MultiAgentOrchestrator()
            result = asyncio.run(orchestrator.orchestrate("做一件事", "test"))
        self.assertEqual(result.final_answer, "最终答案")

    def test_plan_failure_emits_progress(self):
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch
        from nexusagent.core.multi_agent import MultiAgentOrchestrator

        fake_llm = MagicMock()
        fake_llm.ainvoke = AsyncMock(return_value=MagicMock(content="我不会拆"))
        events = []

        with patch("nexusagent.core.multi_agent.get_provider", return_value=fake_llm), \
             patch("nexusagent.core.multi_agent.get_stream_writer", return_value=events.append):
            orchestrator = MultiAgentOrchestrator()
            result = asyncio.run(orchestrator.orchestrate("做一件事", "test"))

        self.assertEqual([event["phase"] for event in events], ["planning", "plan_failed"])
        self.assertIn("任务拆解失败", result.final_answer)

    def test_pipeline_timeout_cancels_running_worker(self):
        """整段流水线到点就停，并取消还在跑的 Worker"""
        import asyncio
        from unittest.mock import MagicMock, patch
        from nexusagent.core.multi_agent import MultiAgentOrchestrator

        cancelled = []

        async def fake_ainvoke(messages):
            first = messages[0].content
            if "任务规划专家" in first:
                return MagicMock(content="SUBTASK|1|做A|none\n")
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
            return MagicMock(content="不会走到")

        fake_llm = MagicMock()
        fake_llm.ainvoke = fake_ainvoke
        events = []

        with patch("nexusagent.core.multi_agent.get_provider", return_value=fake_llm), \
             patch("nexusagent.core.multi_agent.get_stream_writer", return_value=events.append):
            orchestrator = MultiAgentOrchestrator(total_timeout=0.15, call_timeout=30)
            started = time.time()
            result = asyncio.run(orchestrator.orchestrate("对比两个框架", "test"))

        self.assertLess(time.time() - started, 2)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.final_answer, "多Agent流水线超时（0.15秒），请简化任务后重试。")
        self.assertEqual(cancelled, [True])
        self.assertEqual(events[-1]["phase"], "timeout")
        self.assertEqual(orchestrator._inflight, set())

    def test_call_timeout_retries_then_review_continues(self):
        """单次调用超时只让该子任务失败，流水线继续做审查"""
        import asyncio
        from unittest.mock import MagicMock, patch
        from nexusagent.core.multi_agent import MultiAgentOrchestrator, TaskStatus

        calls = {"n": 0}

        async def fake_ainvoke(messages):
            calls["n"] += 1
            first = messages[0].content
            if "任务规划专家" in first:
                return MagicMock(content="SUBTASK|1|做A|none\n")
            if "审查专家" in first:
                return MagicMock(content="汇总")
            await asyncio.sleep(1)
            return MagicMock(content="不会走到")

        fake_llm = MagicMock()
        fake_llm.ainvoke = fake_ainvoke

        with patch("nexusagent.core.multi_agent.get_provider", return_value=fake_llm), \
             patch("nexusagent.core.multi_agent.get_stream_writer", return_value=lambda _event: None):
            orchestrator = MultiAgentOrchestrator(total_timeout=5, call_timeout=0.05)
            result = asyncio.run(orchestrator.orchestrate("做一件事", "test"))

        self.assertFalse(result.timed_out)
        self.assertEqual(result.final_answer, "汇总")
        self.assertEqual(result.subtasks[0].status, TaskStatus.FAILED)
        self.assertIn("模型调用超时（0.05秒）", result.subtasks[0].result)
        self.assertEqual(calls["n"], 5)

    def test_tool_returns_timeout_message_directly(self):
        import asyncio
        from unittest.mock import AsyncMock, patch
        from nexusagent.core.multi_agent import OrchestrationResult
        from nexusagent.core.tools.multi_agent_tools import bind_multi_agent_collaborate

        tool = bind_multi_agent_collaborate("aliyun", "glm-5")
        fake = OrchestrationResult(
            task_id="t",
            subtasks=[],
            final_answer="多Agent流水线超时（5分钟），请简化任务后重试。",
            total_time=300,
            timed_out=True,
        )
        with patch("nexusagent.core.multi_agent.MultiAgentOrchestrator") as cls:
            cls.return_value.orchestrate = AsyncMock(return_value=fake)
            text = asyncio.run(tool.ainvoke({"task": "写一份很长的报告"}))

        self.assertEqual(text, "多Agent流水线超时（5分钟），请简化任务后重试。")

    def test_skipped_and_deadlock_emit_progress(self):
        import asyncio
        from unittest.mock import MagicMock, patch
        from nexusagent.core.multi_agent import MultiAgentOrchestrator, SubTask, TaskStatus

        with patch("nexusagent.core.multi_agent.get_provider", return_value=MagicMock()):
            orchestrator = MultiAgentOrchestrator()
        events = []
        orchestrator._writer = events.append

        root = SubTask(description="调研")
        child = SubTask(description="建议", depends_on=[root.id])

        async def fake_execute(task, task_map, thread_id):
            task.status = TaskStatus.FAILED
            task.result = "boom"

        orchestrator._execute_single = fake_execute
        asyncio.run(orchestrator._execute_dag([root, child], "test"))
        self.assertEqual(
            [(event["phase"], event["task"], event.get("status")) for event in events],
            [
                ("worker_start", "调研", None),
                ("worker_done", "调研", "failed"),
                ("worker_skipped", "建议", "failed"),
            ],
        )

        events.clear()
        left = SubTask(description="做A")
        right = SubTask(description="做B")
        left.depends_on = [right.id]
        right.depends_on = [left.id]
        asyncio.run(orchestrator._execute_dag([left, right], "test"))
        self.assertEqual(
            [(event["phase"], event["task"]) for event in events],
            [("deadlock", "做A"), ("deadlock", "做B")],
        )

    def test_format_orchestration_progress(self):
        from entry.main import format_orchestration_progress

        lines, status = format_orchestration_progress({"phase": "planning"})
        self.assertIn("正在拆解任务", lines[0])
        self.assertEqual(status, "Planner 正在拆解任务...")

        lines, status = format_orchestration_progress({"phase": "planned", "tasks": ["做A", "做B"]})
        self.assertEqual(len(lines), 3)
        self.assertIn("做A", lines[1])
        self.assertEqual(status, "Workers 执行中...")

        lines, status = format_orchestration_progress(
            {"phase": "worker_done", "task": "做A", "status": "failed"}
        )
        self.assertIn("失败", lines[0])
        self.assertIn("失败", status)

        lines, status = format_orchestration_progress({"phase": "reviewing"})
        self.assertIn("Reviewer", lines[0])
        self.assertIn("合并", status)
        lines, status = format_orchestration_progress({"phase": "timeout", "limit": "5分钟"})
        self.assertIn("5分钟", lines[0])
        self.assertEqual(status, "流水线超时（5分钟）")
        self.assertEqual(format_orchestration_progress({"token": "hi"}), ([], ""))

    def test_multi_agent_tool_registered(self):
        """测试多Agent协作工具已注册到内置工具列表"""
        from nexusagent.core.tools import BUILTIN_TOOLS

        tool_names = [t.name for t in BUILTIN_TOOLS]
        self.assertIn("multi_agent_collaborate", tool_names)
        self.assertEqual(tool_names[0], "multi_agent_collaborate")

    def test_bound_tool_uses_main_agent_model(self):
        """协作工具使用绑定时传入的模型，不跟环境变量走"""
        import asyncio
        from unittest.mock import AsyncMock, patch
        from nexusagent.core.multi_agent import OrchestrationResult
        from nexusagent.core.tools.multi_agent_tools import bind_multi_agent_collaborate

        tool = bind_multi_agent_collaborate("aliyun", "glm-5")
        self.assertEqual(tool.name, "multi_agent_collaborate")
        self.assertEqual(tool.metadata["provider_name"], "aliyun")
        self.assertEqual(tool.metadata["model_name"], "glm-5")

        fake = OrchestrationResult(
            task_id="t",
            subtasks=[],
            final_answer="选型建议",
            total_time=0.2,
            success_rate=100.0,
            subtask_count=0,
        )
        with patch.dict(os.environ, {"DEFAULT_PROVIDER": "openai", "DEFAULT_MODEL": "gpt-4o-mini"}):
            with patch("nexusagent.core.multi_agent.MultiAgentOrchestrator") as cls:
                cls.return_value.orchestrate = AsyncMock(return_value=fake)
                text = asyncio.run(tool.ainvoke({"task": "对比两个框架并给出建议"}))

        cls.assert_called_once_with(
            provider_name="aliyun",
            model_name="glm-5",
            max_concurrent_workers=3,
        )
        self.assertIn("选型建议", text)

    def test_agent_app_binds_main_model_onto_collaborate_tool(self):
        """主 Agent 创建时，协作工具继承当时的 provider 和 model"""
        from unittest.mock import MagicMock, patch
        from nexusagent.core.agent import create_agent_app

        llm = MagicMock()
        llm.bind_tools.return_value = llm
        with patch("nexusagent.core.agent.get_provider", return_value=llm), \
             patch("nexusagent.core.agent.load_dynamic_skills", return_value=[]):
            app = create_agent_app(provider_name="aliyun", model_name="qwen-max")

        tool = _collaborate_tool(app)
        self.assertEqual(tool.metadata["provider_name"], "aliyun")
        self.assertEqual(tool.metadata["model_name"], "qwen-max")


def _collaborate_tool(app):
    node = app.nodes["tools"]
    for obj in (node, getattr(node, "bound", None), getattr(node, "runnable", None)):
        tools = getattr(obj, "_tools_by_name", None)
        if tools and "multi_agent_collaborate" in tools:
            return tools["multi_agent_collaborate"]
    raise AssertionError("图里没有 multi_agent_collaborate")

class TestIntegration(unittest.TestCase):
    """集成测试：确保各模块可以正确导入和互操作"""

    def test_all_modules_import(self):
        """测试所有新模块可以正确导入"""
        from nexusagent.core import rag_engine
        from nexusagent.core import security_scorer
        from nexusagent.core import multi_agent

        self.assertTrue(hasattr(rag_engine, 'RAGKnowledgeBase'))
        self.assertTrue(hasattr(security_scorer, 'SecurityScorer'))
        self.assertTrue(hasattr(multi_agent, 'MultiAgentOrchestrator'))

    def test_security_scorer_singleton(self):
        """测试安全评分器全局实例"""
        from nexusagent.core.security_scorer import security_scorer
        self.assertIsNotNone(security_scorer)


if __name__ == '__main__':
    # 设置测试环境
    os.environ.setdefault("NEXUSAGENT_WORKSPACE", tempfile.mkdtemp())

    unittest.main(verbosity=2)
