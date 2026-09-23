import os

from .base import nexusagent_tool


@nexusagent_tool
async def multi_agent_collaborate(task: str) -> str:
    """
    启动多Agent协作引擎处理复杂任务（Planner → Worker → Reviewer 三阶段）。

    【必须调用此工具的场景】：
    - 任务需要拆解为多个独立子任务后综合输出（如调研 + 对比 + 报告）
    - 涉及多维度/多角度分析后给出整合结论（如技术选型、架构评估、策划方案）
    - 用户明确要求「多Agent协作」「任务拆解」「并行处理」
    - 单次回答无法覆盖的复杂综合性任务

    【不要调用此工具的场景】：
    - 简单问答、单步计算、查时间、读写单个文件等可直接完成的轻量任务

    参数 task：完整的用户任务描述，请原样传入，不要省略要求。
    """
    from ..multi_agent import MultiAgentOrchestrator

    provider = os.getenv("DEFAULT_PROVIDER", "openai")
    model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

    orchestrator = MultiAgentOrchestrator(
        provider_name=provider,
        model_name=model,
        max_concurrent_workers=3,
    )
    result = await orchestrator.orchestrate(task, thread_id="local_geek_master")

    lines = [
        f"多Agent协作完成 | 子任务: {result.subtask_count} | "
        f"成功率: {result.success_rate:.0f}% | 耗时: {result.total_time:.1f}s",
        "",
    ]
    for i, st in enumerate(result.subtasks, 1):
        lines.append(f"  [{i}] {st.description} → {st.status.value}")
    lines.extend(["", "---", "", result.final_answer])
    return "\n".join(lines)
