# NexusAgent

Python 智能体框架。主循环用 LangGraph 调度工具；复杂任务可以拆成 Planner、Worker、Reviewer 协作；知识检索用本地 TF-IDF；工具调用带沙盒约束和风险评分。

Python 3.10+ · MIT License

## 能力

| 模块 | 实现 |
|------|------|
| 多 Agent 协作 | Planner 拆任务，Worker 按 DAG 并行执行，Reviewer 合并结果。主 Agent 通过 `multi_agent_collaborate` 调用 |
| RAG | 纯 Python TF-IDF，按段落边界切片，MD5 去重。不依赖外部向量库 |
| 安全评分 | 滑动窗口打分（0–100），带时间衰减和连续异常检测，可输出 Markdown 报告 |
| 执行约束 | 技能先 `help` 再 `run`；沙盒限制路径和 shell |
| 记忆 | 长期画像写在 Markdown，短期摘要放在 SQLite，超长对话自动裁剪 |
| 审计 | 工具调用、模型输入输出等事件写入 JSONL，可用 Rich 终端查看 |
| 技能加载 | 启动时只读元数据，首次调用再加载正文，LRU 缓存 |
| 定时任务 | asyncio 后台检查队列，支持按日、周、月重复 |

## 架构

```
用户输入 / 定时任务
        │
        ▼
记忆与检索（画像、摘要、TF-IDF）
        │
        ▼
主 Agent（LangGraph：推理 ↔ 工具）
        │
        ├── 内置工具与技能
        └── 多 Agent（Planner → Worker → Reviewer）
        │
        ▼
沙盒执行、风险评分、JSONL 审计
```

## 检索基准

BEIR SciFact（Thakur et al., NeurIPS 2021）：1083 篇生物医学摘要，5728 个切片，300 条查询。

| 指标 | NexusAgent TF-IDF | BM25（Anserini） |
|------|-------------------|------------------|
| NDCG@10 | 0.6759 | 0.6647 |
| Recall@10 | 0.8087 | — |
| Recall@100 | 0.9091 | — |
| MRR@10 | 0.6412 | — |
| 平均延迟 | 89.44 ms/query | — |
| P99 延迟 | 115.05 ms/query | — |

同一张 BEIR 表里，DPR、ANCE、TAS-B 的 NDCG@10 分别为 0.3183、0.5072、0.6431。这是稀疏检索在 SciFact 上的结果，不是通用问答榜单。

## 快速开始

```bash
git clone https://github.com/Doublec0413/NexusAgent.git
cd NexusAgent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

在 `.env` 里填写模型提供商和 API Key，然后：

```bash
nexusagent config    # 交互式配置
nexusagent run       # 启动
nexusagent monitor   # 另开一个终端查看审计日志
```

支持 OpenAI 兼容接口、Anthropic 和本地 Ollama。部署细节见 [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md)，技能懒加载见 [docs/LAZY_LOADING_GUIDE.md](docs/LAZY_LOADING_GUIDE.md)。

## 项目结构

```
nexusagent/core/     主循环、多 Agent、RAG、安全评分、记忆与审计
nexusagent/core/tools/  内置工具（通用、知识、调度、沙盒、安全）
entry/               命令行入口
tests/               测试与基准脚本
docs/                部署和懒加载说明
```

## 许可证

[MIT](LICENSE)
