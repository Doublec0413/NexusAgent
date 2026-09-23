<div align="center">

# NexusAgent

多 Agent 协作 · 本地 RAG · 可审计的工具执行

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.x-blue.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[能力](#能力) · [架构](#架构) · [模块](#模块) · [基准](#基准) · [快速开始](#快速开始)

</div>

---

## 能力

NexusAgent 是一个 Python 智能体框架。主循环用 LangGraph 调度工具。复杂任务可以交给 Planner、Worker、Reviewer 协作完成。知识检索在本地用 TF-IDF 完成，工具调用限制在沙盒内，并留下风险评分和审计日志。

| 模块 | 实现 | 说明 |
|------|------|------|
| 多 Agent 协作 | Planner → Worker → Reviewer | 按 DAG 调度，无依赖的子任务并行执行 |
| RAG | TF-IDF + 边界切片 | 纯 Python，不依赖外部向量库 |
| 安全评分 | 滑动窗口，0–100 | 时间衰减、连续异常检测、Markdown 报告 |
| 执行约束 | help → run，路径与 shell 拦截 | 技能先读说明，再在沙盒里执行 |
| 记忆 | Markdown 画像 + SQLite 摘要 | 对话过长时自动裁剪 |
| 审计 | JSONL + Rich 终端 | 记录模型输入、工具调用和工具结果 |
| 技能加载 | 元数据扫描 + LRU | 启动时不读全文，首次调用再加载 |
| 定时任务 | asyncio 队列 | 支持按日、周、月重复 |

---

## 架构

```
用户输入 / 定时任务
        │
        ▼
记忆与检索
长期画像 · 短期摘要 · TF-IDF
        │
        ▼
主 Agent
LangGraph：推理 ↔ 工具
        │
        ├── 内置工具与技能
        └── 多 Agent：Planner → Worker → Reviewer
        │
        ▼
沙盒执行 · 风险评分 · JSONL 审计
```

---

## 模块

### 多 Agent 协作

`nexusagent/core/multi_agent.py`，由主 Agent 的 `multi_agent_collaborate` 工具调用。

```
用户任务
   │
   ▼
Planner     拆成带依赖的子任务
   │
   ▼
Worker      无依赖任务并行执行，失败最多重试 2 次
   │
   ▼
Reviewer    检查结果并合并为最终回答
```

调度使用拓扑排序。循环依赖会被标为失败，不会一直等待。

### RAG

`nexusagent/core/rag_engine.py`

文档加载后按段落或句子边界切片，用字符 n-gram 做中英文混合分词，再建 TF-IDF 索引。相同内容用 MD5 跳过。检索结果注入主 Agent 的上下文。

### 安全评分

`nexusagent/core/security_scorer.py`

每次工具调用更新分数。分数在约 10 分钟内线性衰减。等级从安全、低、中、高到危险。需要留档时生成 Markdown 报告。

### 其他

| 文件 | 作用 |
|------|------|
| `agent.py` | LangGraph 主循环 |
| `context.py` | 对话裁剪与双水位记忆 |
| `skill_loader.py` | 技能懒加载 |
| `heartbeat.py` | 定时任务 |
| `logger.py` | 异步 JSONL 审计 |
| `tools/` | 通用、知识、调度、沙盒、安全工具 |

---

## 基准

### SciFact 检索

数据集为 BEIR SciFact（Thakur et al., NeurIPS 2021）：1083 篇生物医学摘要，5728 个切片，300 条查询。

| 指标 | NexusAgent TF-IDF | BM25（Anserini） |
|------|-------------------|------------------|
| NDCG@10 | **0.6759** | 0.6647 |
| Recall@10 | 0.8087 | — |
| Recall@100 | 0.9091 | — |
| MRR@10 | 0.6412 | — |
| 平均延迟 | 89.44 ms/query | — |
| P99 延迟 | 115.05 ms/query | — |

同一张 BEIR 表中，DPR、ANCE、TAS-B 的 NDCG@10 分别为 0.3183、0.5072、0.6431。这是 SciFact 上的稀疏检索结果。

### 运行开销

| 项目 | 结果 |
|------|------|
| 懒加载启动（100 个技能） | 约 0.41 ms，内存约 50 KB |
| 预加载对照 | 约 200 ms，内存约 250 KB |
| 风险评分 | 平均约 13 μs，P99 约 22 μs |
| 内部异常场景（41 个） | 41/41 命中，该集合上未出现误报或漏报 |

两段式调用在内部对照里把危险执行从 50% 降到 10%。这是项目内场景，不是外部安全评测。

---

## 快速开始

```bash
git clone https://github.com/Doublec0413/NexusAgent.git
cd NexusAgent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

在 `.env` 中填写模型提供商和 API Key。支持 OpenAI 兼容接口、Anthropic 和本地 Ollama。

```bash
nexusagent config     # 交互式配置
nexusagent run        # 启动
nexusagent monitor    # 另开终端查看审计日志
```

安装与排错见 [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)。技能懒加载见 [docs/LAZY_LOADING_GUIDE.md](docs/LAZY_LOADING_GUIDE.md)。

---

## 项目结构

```
NexusAgent/
├── nexusagent/core/
│   ├── agent.py              # 主循环
│   ├── multi_agent.py        # 多 Agent 调度
│   ├── rag_engine.py         # TF-IDF 检索
│   ├── security_scorer.py    # 风险评分
│   ├── context.py            # 记忆与裁剪
│   ├── skill_loader.py       # 懒加载
│   ├── heartbeat.py          # 定时任务
│   ├── logger.py             # 审计日志
│   └── tools/                # 内置工具
├── entry/                    # 命令行
├── tests/                    # 测试与基准脚本
├── docs/                     # 部署与懒加载说明
└── workspace/                # 运行时数据（不入库）
```

---

## 许可证

[MIT License](LICENSE)
