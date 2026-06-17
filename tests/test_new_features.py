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


class TestSecurityDashboard(unittest.TestCase):
    """安全评分仪表盘测试"""

    def test_initial_state(self):
        """测试初始安全状态"""
        from nexusagent.core.security_dashboard import SecurityScorer
        scorer = SecurityScorer()
        data = scorer.get_dashboard_data()

        self.assertEqual(data["risk_score"], 0)
        self.assertEqual(data["total_tool_calls"], 0)
        self.assertIn("安全", data["risk_level"])

    def test_risk_scoring(self):
        """测试风险评分计算"""
        from nexusagent.core.security_dashboard import SecurityScorer
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
        from nexusagent.core.security_dashboard import SecurityScorer
        scorer = SecurityScorer()

        scorer.record_tool_call("execute_office_shell")
        high_score = scorer.get_dashboard_data()["risk_score"]

        scorer.record_conversation()
        decayed_score = scorer.get_dashboard_data()["risk_score"]

        self.assertLessEqual(decayed_score, high_score)

    def test_consecutive_tool_detection(self):
        """测试连续工具调用检测"""
        from nexusagent.core.security_dashboard import SecurityScorer
        scorer = SecurityScorer()

        for _ in range(8):
            scorer.record_tool_call("write_office_file")

        data = scorer.get_dashboard_data()
        self.assertGreater(data["risk_score"], 0)
        self.assertGreater(len(data["recent_alerts"]), 0)

    def test_risk_levels(self):
        """测试风险等级划分"""
        from nexusagent.core.security_dashboard import SecurityScorer
        scorer = SecurityScorer()

        level, _ = scorer.get_risk_level()
        self.assertIn("安全", level)

    def test_report_generation(self):
        """测试安全报告生成"""
        from nexusagent.core.security_dashboard import SecurityScorer
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
        from nexusagent.core.security_dashboard import SecurityScorer
        scorer = SecurityScorer()

        scorer.record_tool_call("calculator")
        scorer.record_tool_call("calculator")
        scorer.record_tool_call("get_current_time")

        data = scorer.get_dashboard_data()
        self.assertEqual(data["tool_distribution"]["calculator"], 2)
        self.assertEqual(data["tool_distribution"]["get_current_time"], 1)


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


class TestIntegration(unittest.TestCase):
    """集成测试：确保各模块可以正确导入和互操作"""

    def test_all_modules_import(self):
        """测试所有新模块可以正确导入"""
        from nexusagent.core import rag_engine
        from nexusagent.core import security_dashboard
        from nexusagent.core import multi_agent

        self.assertTrue(hasattr(rag_engine, 'RAGKnowledgeBase'))
        self.assertTrue(hasattr(security_dashboard, 'SecurityScorer'))
        self.assertTrue(hasattr(multi_agent, 'MultiAgentOrchestrator'))

    def test_security_scorer_singleton(self):
        """测试安全评分器全局实例"""
        from nexusagent.core.security_dashboard import security_scorer
        self.assertIsNotNone(security_scorer)


if __name__ == '__main__':
    # 设置测试环境
    os.environ.setdefault("NEXUSAGENT_WORKSPACE", tempfile.mkdtemp())

    unittest.main(verbosity=2)
