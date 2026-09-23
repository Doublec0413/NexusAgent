from .base import nexusagent_tool


@nexusagent_tool
def query_knowledge_base(query: str) -> str:
    """
    在 RAG 知识库中检索信息。
    当用户提出的问题可能与已导入的知识文档相关时，调用此工具进行语义检索。
    参数 query 是用户的查询问题。
    返回知识库中与问题最相关的文档片段。
    """
    from ..rag_engine import RAGKnowledgeBase
    kb = RAGKnowledgeBase()
    results = kb.search(query, top_k=3)
    if not results:
        return "知识库中未找到相关内容。请先导入文档到 workspace/knowledge/ 目录。"

    output = "📚 知识库检索结果：\n"
    for r in results:
        output += f"\n--- [{r.rank}] 来源: {r.chunk.source_file} (相关度: {r.score:.2f}) ---\n"
        output += f"{r.chunk.content[:400]}\n"
    return output
