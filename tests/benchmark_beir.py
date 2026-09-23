"""
NexusAgent BEIR 标准基准测试
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
使用 BEIR (Benchmarking Information Retrieval) 官方数据集评测。
BEIR 论文: https://arxiv.org/abs/2104.08663
引用: Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot
      Evaluation of Information Retrieval Models", NeurIPS 2021

数据集: SciFact
- 5183 篇生物医学论文摘要 (corpus)
- 300 条科学论断查询 (queries)
- 339 条人工标注相关性判断 (qrels, test split)
- 来源: HuggingFace BeIR/scifact

评测指标: 标准 IR 指标
- NDCG@10 (归一化折损累积增益, 主指标)
- Recall@10, Recall@100
- MRR@10 (平均倒数排名)
- MAP (平均精度均值)

对比基线 (来自 BEIR 论文 Table 2):
- BM25 (Anserini):  NDCG@10 = 0.665
- TF-IDF (sklearn): NDCG@10 ≈ 0.60 (估计值)

运行方式:
  cd NexusAgent
  python tests/benchmark_beir.py
"""

import os
import sys
import time
import math
import statistics
import json
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
TEST_WORKSPACE = tempfile.mkdtemp()
os.environ["NEXUSAGENT_WORKSPACE"] = TEST_WORKSPACE


def ndcg_at_k(ranked_list, relevant_docs, k):
    """计算 NDCG@k"""
    dcg = 0.0
    for i, doc_id in enumerate(ranked_list[:k]):
        if doc_id in relevant_docs:
            rel = relevant_docs[doc_id]
            dcg += rel / math.log2(i + 2)

    # Ideal DCG
    ideal_rels = sorted(relevant_docs.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_list, relevant_docs, k):
    """计算 Recall@k"""
    if not relevant_docs:
        return 0.0
    retrieved = set(ranked_list[:k])
    relevant = set(relevant_docs.keys())
    return len(retrieved & relevant) / len(relevant)


def mrr_at_k(ranked_list, relevant_docs, k):
    """计算 MRR@k (Mean Reciprocal Rank)"""
    for i, doc_id in enumerate(ranked_list[:k]):
        if doc_id in relevant_docs:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(ranked_list, relevant_docs, k):
    """计算 AP@k"""
    hits = 0
    sum_prec = 0.0
    for i, doc_id in enumerate(ranked_list[:k]):
        if doc_id in relevant_docs:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / len(relevant_docs) if relevant_docs else 0.0


def main():
    print("=" * 70)
    print("🔬 NexusAgent BEIR SciFact 标准基准测试")
    print("=" * 70)
    print()

    # ━━━━━━━ Step 1: 加载 BEIR SciFact 数据集 ━━━━━━━
    print("📥 加载 BEIR SciFact 数据集...")
    from datasets import load_dataset

    corpus_ds = load_dataset('BeIR/scifact', 'corpus', split='corpus')
    queries_ds = load_dataset('BeIR/scifact', 'queries', split='queries')
    qrels_ds = load_dataset('BeIR/scifact-qrels', split='test')

    print(f"   Corpus:  {len(corpus_ds)} 文档")
    print(f"   Queries: {len(queries_ds)} 条查询")
    print(f"   Qrels:   {len(qrels_ds)} 条相关性标注 (test)")

    # 构建 id→内容映射
    corpus = {}
    for item in corpus_ds:
        doc_id = str(item['_id'])
        text = item.get('title', '') + ' ' + item.get('text', '')
        corpus[doc_id] = text

    queries = {}
    for item in queries_ds:
        queries[str(item['_id'])] = item['text']

    # 构建 qrels: {query_id: {doc_id: relevance_score}}
    qrels = defaultdict(dict)
    for item in qrels_ds:
        qid = str(item['query-id'])
        did = str(item['corpus-id'])
        score = item['score']
        if score > 0:
            qrels[qid][did] = score

    test_query_ids = [qid for qid in qrels.keys() if qid in queries]
    print(f"   有效测试查询: {len(test_query_ids)} 条")

    # ━━━━━━━ Step 2: 索引文档到 NexusAgent RAG 引擎 ━━━━━━━
    print(f"\n📚 索引 {len(corpus)} 篇文档到 NexusAgent RAG 引擎...")
    from nexusagent.core.rag_engine import RAGKnowledgeBase

    kb_dir = os.path.join(TEST_WORKSPACE, "beir_scifact")
    os.makedirs(kb_dir, exist_ok=True)
    kb = RAGKnowledgeBase(knowledge_dir=kb_dir)

    # 将文档写入文件并索引
    # 为了速度，我们将每篇文档写为一个文件
    start_index = time.time()
    doc_id_to_file = {}
    for doc_id, text in corpus.items():
        fname = f"doc_{doc_id}.txt"
        fpath = os.path.join(kb_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(text)
        doc_id_to_file[doc_id] = fname

    # 批量索引
    kb.add_directory(kb_dir)
    index_time = time.time() - start_index

    print(f"   索引完成: {len(kb.chunks)} 个切片, 耗时 {index_time:.1f}s")

    # 构建 chunk source_file → doc_id 的反向映射
    file_to_doc_id = {v: k for k, v in doc_id_to_file.items()}

    # ━━━━━━━ Step 3: 逐查询检索并评测 ━━━━━━━
    print(f"\n🔍 检索评测 ({len(test_query_ids)} 条查询)...")

    all_ndcg10 = []
    all_recall10 = []
    all_recall100 = []
    all_mrr10 = []
    all_ap = []
    search_times = []

    for qid in test_query_ids:
        query_text = queries[qid]
        relevant_docs = qrels[qid]

        start = time.perf_counter()
        results = kb.search(query_text, top_k=100)
        elapsed = (time.perf_counter() - start) * 1000
        search_times.append(elapsed)

        # 将检索结果映射回 doc_id
        ranked_doc_ids = []
        for r in results:
            src = r.chunk.source_file
            did = file_to_doc_id.get(src, "")
            if did and did not in ranked_doc_ids:
                ranked_doc_ids.append(did)

        all_ndcg10.append(ndcg_at_k(ranked_doc_ids, relevant_docs, 10))
        all_recall10.append(recall_at_k(ranked_doc_ids, relevant_docs, 10))
        all_recall100.append(recall_at_k(ranked_doc_ids, relevant_docs, 100))
        all_mrr10.append(mrr_at_k(ranked_doc_ids, relevant_docs, 10))
        all_ap.append(average_precision(ranked_doc_ids, relevant_docs, 100))

    # ━━━━━━━ Step 4: 汇总结果 ━━━━━━━
    print("\n" + "=" * 70)
    print("📊 BEIR SciFact 评测结果")
    print("=" * 70)

    avg_ndcg10 = statistics.mean(all_ndcg10)
    avg_recall10 = statistics.mean(all_recall10)
    avg_recall100 = statistics.mean(all_recall100)
    avg_mrr10 = statistics.mean(all_mrr10)
    avg_map = statistics.mean(all_ap)
    avg_latency = statistics.mean(search_times)
    p99_latency = sorted(search_times)[int(len(search_times) * 0.99)]

    print(f"""
┌────────────────────────────────────────────────────────────┐
│  BEIR SciFact Benchmark Results                            │
│  Dataset: 5183 docs, {len(test_query_ids)} test queries, 339 qrels           │
│  Engine:  NexusAgent TF-IDF (pure Python)                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  NDCG@10:     {avg_ndcg10:.4f}                                      │
│  Recall@10:   {avg_recall10:.4f}                                      │
│  Recall@100:  {avg_recall100:.4f}                                      │
│  MRR@10:      {avg_mrr10:.4f}                                      │
│  MAP:         {avg_map:.4f}                                      │
│                                                            │
│  Avg Latency: {avg_latency:.2f}ms / query                           │
│  P99 Latency: {p99_latency:.2f}ms / query                           │
│  Index Time:  {index_time:.1f}s ({len(kb.chunks)} chunks)                        │
│                                                            │
├────────────────────────────────────────────────────────────┤
│  BEIR Baselines (from paper, SciFact):                     │
│  BM25 (Anserini):   NDCG@10 = 0.6647                      │
│  DocT5query:        NDCG@10 = 0.6758                      │
│  DPR:               NDCG@10 = 0.3183                      │
│  ANCE:              NDCG@10 = 0.5072                      │
│  TAS-B:             NDCG@10 = 0.6431                      │
└────────────────────────────────────────────────────────────┘
""")

    # ━━━━━━━ 安全评分附加测试 ━━━━━━━
    print("=" * 70)
    print("🛡️ 安全评分系统测试 (附加)")
    print("=" * 70)

    from nexusagent.core.security_scorer import SecurityScorer

    # 延迟测试
    scorer = SecurityScorer()
    tools = ["get_current_time", "calculator", "read_office_file",
             "write_office_file", "execute_office_shell", "list_office_files"]
    latency_times = []
    for i in range(10000):
        start = time.perf_counter()
        scorer.record_tool_call(tools[i % len(tools)])
        latency_times.append((time.perf_counter() - start) * 1_000_000)

    sec_avg = statistics.mean(latency_times)
    sec_p99 = sorted(latency_times)[int(len(latency_times) * 0.99)]

    # 异常检测 (41 场景, 同 v2)
    from nexusagent.core.security_scorer import SecurityScorer as SS
    scenarios = [
        (["get_current_time"], False),
        (["calculator"] * 3, False),
        (["get_current_time", "calculator"], False),
        (["list_office_files"], False),
        (["read_office_file"] * 2, False),
        (["get_current_time"] * 5, False),
        (["list_office_files", "calculator", "get_current_time"], False),
        (["calculator"] * 4, False),
        (["list_scheduled_tasks", "get_current_time"], False),
        (["read_office_file", "calculator"], False),
        (["execute_office_shell"], False),
        (["execute_office_shell"] * 2, False),
        (["execute_office_shell"] * 3, False),
        (["write_office_file"] * 2, False),
        (["write_office_file"] * 4, False),
        (["execute_office_shell", "get_current_time", "execute_office_shell"], False),
        (["write_office_file", "calculator", "write_office_file"], False),
        (["execute_office_shell"] * 5, False),
        (["save_user_profile", "get_current_time", "calculator"], False),
        (["modify_scheduled_task", "list_scheduled_tasks"], False),
        # 异常 (连续>5)
        (["execute_office_shell"] * 6, True),
        (["execute_office_shell"] * 7, True),
        (["execute_office_shell"] * 8, True),
        (["execute_office_shell"] * 10, True),
        (["execute_office_shell"] * 15, True),
        (["write_office_file"] * 6, True),
        (["write_office_file"] * 8, True),
        (["write_office_file"] * 10, True),
        (["read_office_file"] * 7, True),
        (["read_office_file"] * 10, True),
        (["execute_office_shell", "write_office_file"] * 4, True),
        (["execute_office_shell", "write_office_file"] * 5, True),
        (["execute_office_shell"] * 3 + ["write_office_file"] * 4, True),
        (["save_user_profile"] * 7, True),
        (["delete_scheduled_task"] * 6, True),
        (["calculator"] * 8, True),
        (["get_current_time"] * 7, True),
        (["list_office_files"] * 6, True),
        (["execute_office_shell"] * 4 + ["read_office_file"] * 4, True),
        (["write_office_file"] * 3 + ["execute_office_shell"] * 5, True),
        (["execute_office_shell"] * 5 + ["get_current_time"], True),
    ]
    correct = sum(1 for ops, expect in scenarios
                  if (len(SS().record_tool_call(ops[0]) or '') >= 0)  # dummy
                  )
    # 重新正确测试
    correct = 0
    for ops, should_alert in scenarios:
        s = SS()
        for op in ops:
            s.record_tool_call(op)
        has_alert = len(s.get_dashboard_data()["recent_alerts"]) > 0
        if has_alert == should_alert:
            correct += 1

    total_s = len(scenarios)
    sec_accuracy = correct / total_s * 100

    print(f"\n  评分延迟: avg {sec_avg:.1f}μs, P99 {sec_p99:.1f}μs (10000次)")
    print(f"  异常检测: {sec_accuracy:.0f}% ({correct}/{total_s} 场景)")

    # ━━━━━━━ 保存结果 ━━━━━━━
    results = {
        "benchmark": "BEIR SciFact",
        "paper": "Thakur et al., NeurIPS 2021",
        "dataset_url": "https://huggingface.co/datasets/BeIR/scifact",
        "corpus_size": len(corpus),
        "test_queries": len(test_query_ids),
        "chunk_count": len(kb.chunks),
        "metrics": {
            "NDCG@10": round(avg_ndcg10, 4),
            "Recall@10": round(avg_recall10, 4),
            "Recall@100": round(avg_recall100, 4),
            "MRR@10": round(avg_mrr10, 4),
            "MAP": round(avg_map, 4),
        },
        "baselines_from_paper": {
            "BM25_Anserini": 0.6647,
            "DocT5query": 0.6758,
            "DPR": 0.3183,
            "ANCE": 0.5072,
        },
        "latency": {
            "avg_ms": round(avg_latency, 2),
            "p99_ms": round(p99_latency, 2),
            "index_time_s": round(index_time, 1),
        },
        "security": {
            "scoring_latency_avg_us": round(sec_avg, 1),
            "scoring_latency_p99_us": round(sec_p99, 1),
            "detection_accuracy": f"{sec_accuracy:.0f}% ({correct}/{total_s})",
        }
    }

    outpath = os.path.join(os.path.dirname(__file__), "benchmark_beir_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📁 结果已保存至: {outpath}")

    # 清理
    import shutil
    shutil.rmtree(TEST_WORKSPACE, ignore_errors=True)

    return results


if __name__ == "__main__":
    main()
