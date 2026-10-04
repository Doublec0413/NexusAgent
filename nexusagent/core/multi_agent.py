"""
NexusAgent 多Agent协作引擎 (Multi-Agent Orchestrator)

实现了 Planner → Worker → Reviewer 三阶段协作模式：
1. Planner Agent: 接收用户复杂任务，拆解为子任务DAG
2. Worker Agents: 并行/串行执行子任务（支持并发控制）
3. Reviewer Agent: 审查所有子任务输出，合并为最终答案

集成方式：
- 由主 Agent（LangGraph StateGraph）通过 multi_agent_collaborate 工具调用，非 LangGraph 节点

核心技术：
- asyncio + 手写 DAG 拓扑调度（_execute_dag），无依赖子任务并行执行
- 子任务依赖分析，确保执行顺序正确
- 失败重试机制，单个子任务最多重试2次
"""

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from .provider import get_provider
from .logger import audit_logger


def _capture_stream_writer():
    """只在 astream(stream_mode 含 custom) 里可用。直接调用编排器时没有这个上下文。"""
    try:
        return get_stream_writer()
    except (RuntimeError, KeyError, LookupError):
        return None


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SubTask:
    """子任务数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    assigned_role: str = "worker"
    retry_count: int = 0
    max_retries: int = 2
    execution_time: float = 0.0


@dataclass
class OrchestrationResult:
    """编排结果"""
    task_id: str
    subtasks: List[SubTask]
    final_answer: str = ""
    total_time: float = 0.0
    success_rate: float = 0.0
    subtask_count: int = 0


class MultiAgentOrchestrator:
    """
    多Agent协作编排器

    工作流程:
        用户输入 → Planner(任务拆解) → Workers(并行执行) → Reviewer(审查合并) → 最终输出
    """

    def __init__(
        self,
        provider_name: str = "openai",
        model_name: str = "gpt-4o-mini",
        max_concurrent_workers: int = 3,
    ):
        self.llm = get_provider(provider_name=provider_name, model_name=model_name)
        self.max_concurrent = max_concurrent_workers
        self._semaphore = asyncio.Semaphore(max_concurrent_workers)
        self._writer = None

    def _emit(self, event: dict) -> None:
        """把阶段进度推进 custom 流。推送失败不能打断流水线。"""
        writer = self._writer
        if writer is None:
            return
        try:
            writer(event)
        except Exception:
            return

    async def orchestrate(self, user_task: str, thread_id: str = "orchestrator") -> OrchestrationResult:
        """
        主编排入口：接收用户任务，返回编排结果。
        各阶段通过 stream writer 推进度，主界面不用等整段流水线结束才有输出。
        """
        start_time = time.time()
        task_id = str(uuid.uuid4())[:12]
        self._writer = _capture_stream_writer()

        try:
            return await self._orchestrate(user_task, thread_id, task_id, start_time)
        finally:
            self._writer = None

    async def _orchestrate(
        self, user_task: str, thread_id: str, task_id: str, start_time: float
    ) -> OrchestrationResult:
        audit_logger.log_event(
            thread_id=thread_id,
            event="system_action",
            content=f"多Agent编排启动 | Task: {task_id}"
        )

        # Phase 1: Planner 拆解任务
        self._emit({"phase": "planning"})
        subtasks = await self._plan(user_task, thread_id)

        if not subtasks:
            self._emit({"phase": "plan_failed"})
            return OrchestrationResult(
                task_id=task_id,
                subtasks=[],
                final_answer="任务拆解失败，请尝试更具体地描述你的需求。",
                total_time=time.time() - start_time
            )

        self._emit({
            "phase": "planned",
            "count": len(subtasks),
            "tasks": [t.description for t in subtasks],
        })

        # Phase 2: Workers 并行执行（基于DAG依赖）
        await self._execute_dag(subtasks, thread_id)

        # Phase 3: Reviewer 审查合并
        self._emit({"phase": "reviewing"})
        final_answer = await self._review(user_task, subtasks, thread_id)

        total_time = time.time() - start_time
        completed_count = sum(1 for t in subtasks if t.status == TaskStatus.COMPLETED)
        success_rate = (completed_count / len(subtasks) * 100) if subtasks else 0

        result = OrchestrationResult(
            task_id=task_id,
            subtasks=subtasks,
            final_answer=final_answer,
            total_time=total_time,
            success_rate=success_rate,
            subtask_count=len(subtasks)
        )

        audit_logger.log_event(
            thread_id=thread_id,
            event="system_action",
            content=f"多Agent编排完成 | 子任务: {len(subtasks)} | 成功率: {success_rate:.1f}% | 耗时: {total_time:.2f}s"
        )

        return result

    async def _plan(self, user_task: str, thread_id: str) -> List[SubTask]:
        """
        Planner Agent: 将复杂任务拆解为可执行的子任务列表
        """
        plan_prompt = SystemMessage(content=(
            "你是一个任务规划专家。你的职责是将用户的复杂任务拆解为3-6个可独立执行的子任务。\n\n"

            "输出格式（严格遵守）：\n"
            "SUBTASK|<编号>|<子任务描述>|<依赖的编号列表,用逗号分隔,无依赖填none>\n\n"

            "示例：\n"
            "SUBTASK|1|调研Python Web框架的市场份额|none\n"
            "SUBTASK|2|对比Django和FastAPI的性能差异|none\n"
            "SUBTASK|3|根据调研结果给出框架选型建议|1,2\n\n"
            
            "要求：\n"
            "1. 每个子任务要足够具体，可以由一个Agent独立完成\n"
            "2. 避免生成过大的任务，例如“完成系统开发”。"
            "3. 正确标注依赖关系（哪些任务需要在其他任务完成后才能开始），注意不允许循环依赖。\n"
            "4. 无依赖的任务可以并行执行\n"
            "5. 只输出SUBTASK行，不要输出任何其他内容"
        ))

        response = await self.llm.ainvoke([plan_prompt, HumanMessage(content=user_task)])

        audit_logger.log_event(
            thread_id=thread_id,
            event="tool_call",
            tool="planner_agent",
            args={"task": user_task}
        )

        parsed = []
        id_map = {}

        for line in response.content.splitlines():
            line = line.strip()
            if not line.startswith("SUBTASK|"):
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue

            seq_num, desc, deps_str = parts[1].strip(), parts[2].strip(), parts[3].strip()
            if not desc:
                continue
            if seq_num in id_map:
                continue

            task = SubTask(description=desc)
            id_map[seq_num] = task.id
            parsed.append((task, deps_str))

        for task, deps_str in parsed:
            if deps_str.lower() in ("none", ""):
                continue
            task.depends_on = [
                id_map[d.strip()] for d in deps_str.split(",")
                if d.strip() in id_map and id_map[d.strip()] != task.id
            ]

        return [t for t, _ in parsed]

    async def _execute_dag(self, subtasks: List[SubTask], thread_id: str):
        """
        基于DAG依赖关系调度执行子任务
        无依赖的任务并行执行，有依赖的等待前置任务成功完成。
        前置任务失败时，直接和传递依赖的后继任务标记为失败并跳过。
        """
        task_map = {t.id: t for t in subtasks}
        completed_ids = set()

        while len(completed_ids) < len(subtasks):
            ready_tasks = []
            blocked_tasks = []

            for t in subtasks:
                if t.id in completed_ids:
                    continue

                deps = [task_map.get(dep_id) for dep_id in t.depends_on]
                if any(dep is not None and dep.status == TaskStatus.FAILED for dep in deps):
                    blocked_tasks.append(t)
                elif all(dep is not None and dep.status == TaskStatus.COMPLETED for dep in deps):
                    ready_tasks.append(t)

            for t in blocked_tasks:
                t.status = TaskStatus.FAILED
                t.result = "前置任务失败，已跳过"
                completed_ids.add(t.id)
                self._emit({
                    "phase": "worker_skipped",
                    "task": t.description,
                    "status": TaskStatus.FAILED.value,
                })

            if not ready_tasks:
                if blocked_tasks:
                    continue
                # 没有可执行任务但还有未完成任务：依赖成环
                remaining = [t for t in subtasks if t.id not in completed_ids]
                if remaining:
                    for t in remaining:
                        t.status = TaskStatus.FAILED
                        t.result = "依赖死锁：前置任务未完成"
                        completed_ids.add(t.id)
                        self._emit({
                            "phase": "deadlock",
                            "task": t.description,
                            "status": TaskStatus.FAILED.value,
                        })
                break

            # 在父任务里推送进度。writer 内部会读当前任务的 LangGraph 上下文，
            # 不能放到 gather 出去的子任务里。
            for t in ready_tasks:
                self._emit({"phase": "worker_start", "task": t.description})

            pending = {
                asyncio.create_task(self._execute_single(t, task_map, thread_id)): t
                for t in ready_tasks
            }
            errors = []
            while pending:
                done, _ = await asyncio.wait(set(pending), return_when=asyncio.FIRST_COMPLETED)
                for fut in done:
                    task = pending.pop(fut)
                    exc = fut.exception()
                    if exc is not None:
                        errors.append(exc)
                        status = TaskStatus.FAILED.value
                    else:
                        status = task.status.value
                    self._emit({
                        "phase": "worker_done",
                        "task": task.description,
                        "status": status,
                    })
                    completed_ids.add(task.id)
            if errors:
                raise errors[0]

    async def _execute_single(self, task: SubTask, task_map: Dict[str, SubTask], thread_id: str):
        """
        执行单个子任务（带重试和并发控制）
        """
        async with self._semaphore:
            task.status = TaskStatus.RUNNING
            start = time.time()

            # 收集依赖任务的结果作为上下文
            dep_context = ""
            for dep_id in task.depends_on:
                dep_task = task_map.get(dep_id)
                if dep_task and dep_task.status == TaskStatus.COMPLETED and dep_task.result:
                    dep_context += f"\n[前置任务 '{dep_task.description}' 的结果]:\n{dep_task.result}\n"

            worker_prompt = (
                f"你是一个专注高效的执行者。请完成以下任务：\n\n"
                f"任务：{task.description}\n"
            )
            if dep_context:
                worker_prompt += f"\n参考上下文（前置任务的输出）：{dep_context}\n"

            worker_prompt += (
                "\n要求：\n"
                "1. 直接给出可用结果，不要开场白、客套和重复总结\n"
                "2. 保留关键事实、数据、对比点和必要论据，勿因「简洁」省略实质内容\n"
                "3. 若任务适合结构化输出，用简洁条目或小标题组织，便于后续汇总"
            )

            for attempt in range(task.max_retries + 1):
                try:
                    response = await self.llm.ainvoke([HumanMessage(content=worker_prompt)])
                    task.result = response.content
                    task.status = TaskStatus.COMPLETED
                    task.execution_time = time.time() - start

                    audit_logger.log_event(
                        thread_id=thread_id,
                        event="tool_result",
                        tool=f"worker_agent_{task.id}",
                        result_summary=task.result[:200]
                    )
                    break
                except Exception as e:
                    task.retry_count += 1
                    if attempt >= task.max_retries:
                        task.status = TaskStatus.FAILED
                        task.result = f"执行失败（重试{task.max_retries}次）：{str(e)}"
                        task.execution_time = time.time() - start

    async def _review(self, original_task: str, subtasks: List[SubTask], thread_id: str) -> str:
        """
        Reviewer Agent: 审查所有子任务结果，合并为最终答案
        """
        results_summary = ""
        for i, task in enumerate(subtasks, 1):
            status_icon = "✅" if task.status == TaskStatus.COMPLETED else "❌"
            results_summary += (
                f"\n--- 子任务 {i} {status_icon} ---\n"
                f"描述: {task.description}\n"
                f"状态: {task.status.value}\n"
                f"结果: {task.result}\n"
            )

        review_prompt = (
            f"你是一个审查专家。以下是一个复杂任务被拆解后各子任务的执行结果。\n\n"
            f"【原始任务】{original_task}\n\n"
            f"【各子任务结果】{results_summary}\n\n"
            f"请你：\n"
            f"1. 检查各子任务的输出是否正确、是否有矛盾\n"
            f"2. 将所有结果整合为一个完整、连贯的最终答案\n"
            f"3. 如果有子任务失败，说明影响并给出补救建议\n"
            f"4. 若原始任务指定了交付形式（报告、代码、表格、清单、方案等），必须按该形式输出完整正文（含结构、章节、字段和细节），禁止仅给摘要、大纲或一句话结论\n"
            f"5. 直接输出最终答案，保留子任务中的关键细节"
        )

        response = await self.llm.ainvoke([HumanMessage(content=review_prompt)])

        audit_logger.log_event(
            thread_id=thread_id,
            event="tool_call",
            tool="reviewer_agent",
            args={"subtask_count": len(subtasks)}
        )

        return response.content
