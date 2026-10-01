🔮

# NexusAgent

下一代透明智能体

多 Agent 协作 · 本地检索 · 沙盒隔离 · 行为审计

![Python](https://img.shields.io/badge/Python-3.10+-8d52ff?style=flat-square&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-111111?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-6b7280?style=flat-square)

[概述](#概述) · [快速开始](#快速开始) · [架构](#架构) · [模块](#模块) · [基准测试](#基准测试) · [项目结构](#项目结构)

---

## 概述

NexusAgent 是一个 Python 智能体框架。主循环由 LangGraph 调度：模型读取对话历史后直接回复，或发起工具调用，再根据工具结果继续推理。会话状态写入 SQLite，进程重启后可以恢复。检索是纯 Python 实现的 TF-IDF，不依赖向量数据库。


| 能力     | 实现                                         |
| ------ | ------------------------------------------ |
| 执行可追踪  | 模型输入、工具调用和工具结果写入日志，并提供监控终端                 |
| 权限受控   | 文件与命令限制在 `workspace/office/`，每个动作计算风险分     |
| 复杂任务拆解 | Planner 拆分子任务，Worker 按依赖并行执行，Reviewer 合并结果 |


支持的模型：OpenAI、Anthropic（Claude）、阿里云、腾讯云、智谱、Ollama，以及兼容 OpenAI 接口的服务。

---



## 快速开始

### 安装

```bash
git clone https://github.com/Doublec0413/NexusAgent.git
cd NexusAgent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .
```



### 配置

```bash
nexusagent config
```

向导会选择提供商、填写模型名和 API Key，并测试连通性。也可以复制 `.env.example` 为 `.env` 后手动编辑。



### 运行

```bash
nexusagent run       # 终端对话
nexusagent monitor   # 另一个终端查看行为日志
```



### 示例

下面这段对话会依次写入长期记忆、在沙盒中写文件并执行、触发多 Agent、创建定时任务，并查询风险分。

```text
你好，记住我叫小王，以后叫我王工
帮我在 office 里写一个快速排序 quick_sort.py 并运行它
调研 Python 三大 Web 框架并输出对比报告
明天早上 8 点提醒我开会
现在的安全评分是多少？
```

导入知识库时，将 `.txt`、`.md` 等文件放入 `workspace/knowledge/`，再调用 `RAGKnowledgeBase().add_directory(...)` 建索引。用法见 `tests/benchmark_all.py`。

---



## 架构

一条消息依次经过五层。


| 层     | 内容                                         |
| ----- | ------------------------------------------ |
| 输入    | 终端输入，或心跳任务到点触发                             |
| 记忆与检索 | 用户画像、对话摘要、知识库中的相关段落                        |
| 决策    | LangGraph 主循环；复杂任务交给规划、并行执行和审查             |
| 工具    | 16 个内置工具和外部技能；文件与命令限制在 `workspace/office/` |
| 安全与监控 | 0–100 风险分、JSONL 审计日志、监控终端和 Markdown 报告     |


---



## 模块

### 多 Agent 协作

复杂任务由主 Agent 调用 `multi_agent_collaborate`（`nexusagent/core/multi_agent.py`）。


| 阶段       | 做什么                     |
| -------- | ----------------------- |
| Planner  | 拆成 3–6 个子任务，并标注依赖       |
| Workers  | 无依赖的子任务并行执行；失败后最多重试 2 次 |
| Reviewer | 检查输出是否矛盾，合并为最终答案        |


子任务依赖是有向无环图。调度器每轮选出前置任务都已完成的子任务。若出现环（A 依赖 B，且 B 依赖 A），相关任务会被标为失败并结束。并行使用 `asyncio`，同时运行的 Worker 不超过 3 个。



### RAG 知识库

回答前，先从导入文档中检索相关段落，写入系统提示词（`nexusagent/core/rag_engine.py`）。


| 步骤  | 规则                                          |
| --- | ------------------------------------------- |
| 切片  | 约 512 字符一段，相邻段重叠 128 字符，优先在段落和句号处切开         |
| 索引  | TF-IDF：词在本段越多、在其他段越少，权重越高。英文按词，中文按字，并加入相邻词对 |
| 检索  | 查询同样转为向量，按余弦相似度取得分最高的段落                     |
| 注入  | 检索结果拼入系统提示词                                 |


导入时用 MD5 去重，同一文件不会重复建索引。



### 安全评分

每个动作映射到 0–100 的风险分，分数越高表示近期行为越危险（`nexusagent/core/security_scorer.py`）。


| 事件           | 分值  |
| ------------ | --- |
| 执行 Shell     | 35  |
| 写文件          | 18  |
| 读文件          | 6   |
| 查询时间         | 0   |
| 连续工具调用超过 5 次 | +68 |
| 触发沙盒拦截       | +55 |
| 对话中拒绝越权诱导    | +67 |
| 普通对话         | −10 |
| 空闲每 60 秒     | −10 |


取最近 50 次事件，在 10 分钟内按时间加权平均，结果截断到 0–100。越近的事件权重越高。分档为安全（0–20）、低风险、中风险、高风险、危险（80+）。当前等级会写入系统提示词，`get_security_report` 可导出 Markdown 审计报告。



### 沙盒隔离

文件和命令限制在 `workspace/office/`（`sandbox_tools.py`、`sandbox_protocol.py`）。


| 检查  | 规则                                                |
| --- | ------------------------------------------------- |
| 路径  | 先解析为绝对路径，再判断是否位于 office 内。`../../etc/passwd` 会被拒绝 |
| 命令  | 拒绝相对路径越权（`..`）、Unix 绝对路径、`~`、Windows 根目录和盘符       |
| 提示词 | 系统提示词要求拒绝越权诱导。对话层的拒绝计入安全评分                        |
| 限额  | 命令超过 60 秒会被终止；单次读取超过 1 万字符时截断                     |




### 两段式技能

外部技能位于 `workspace/office/skills/`，每个目录包含一份 `SKILL.md`（`nexusagent/core/skill_loader.py`）。


| 模式     | 行为          |
| ------ | ----------- |
| `help` | 返回说明书，不执行命令 |
| `run`  | 执行技能中的命令    |


调用顺序是先 `help` 再 `run`。模型先读到技能说明，再决定是否执行。`tests/test_two_phase_skills.py` 用 20 个场景比较两种调用，每个场景有一个名称普通、说明书标明高风险的工具。技能懒加载：启动时只读名称和简介，正文在首次 `help` 时读取。常用技能留在 LRU 缓存中，并按文件修改时间热更新。



### 记忆


| 类型  | 存储                                 | 规则                                        |
| --- | ---------------------------------- | ----------------------------------------- |
| 长期  | `workspace/memory/user_profile.md` | 用户表达长期偏好时，调用 `save_user_profile` 更新，跨会话保留 |
| 短期  | SQLite                             | 超过 40 轮时，较早内容压缩为不超过 150 字的摘要，完整保留最近 10 轮  |




### 心跳任务

后台协程每 10 秒检查一次任务队列（`nexusagent/core/heartbeat.py`）。支持单次提醒，以及按小时、天、周、月重复的任务，重复次数可以设置上限。到点后，提醒作为一条消息交给 Agent。进程停机错过触发时间时，重启后循环任务跳到下一个未来时刻，不补发已错过的提醒。

时间存在上午/下午歧义时，先向用户确认再创建任务。一条描述匹配到多个任务时，不批量删除或修改，要求用户指定编号。



### 审计日志

五类事件以 JSONL 写入 `logs/`：模型输入、工具调用、工具结果、模型回复、协议拦截（`logger.py`、`entry/monitor.py`）。写入经过内存队列和后台线程，不阻塞主循环。另一个终端运行 `nexusagent monitor` 可以实时查看。

---



## 基准测试

数字来自 `tests/benchmark_all.py` 和 `tests/benchmark_beir.py`，结果文件为 `tests/benchmark_results.json` 和 `tests/benchmark_beir_results.json`。

### BEIR SciFact

BEIR（Thakur et al., NeurIPS 2021）是信息检索评测集。SciFact 由生物医学论文摘要和科学论断查询组成。下表在 1083 篇文档上测得（全部相关文档，另加 800 篇干扰文档），切片 5728 个。


| 指标         | 数值    | 说明                   |
| ---------- | ----- | -------------------- |
| NDCG@10    | 0.676 | 前 10 条结果的排序质量，范围 0–1 |
| Recall@10  | 0.809 | 相关文档出现在前 10 条中的比例    |
| Recall@100 | 0.909 | 相关文档出现在前 100 条中的比例   |
| MRR@10     | 0.641 | 首条相关结果的平均倒数排名        |
| 平均检索延迟     | 89 ms | 每个查询                 |


复现：`python tests/benchmark_beir.py`



### 小规模知识库

30 条查询，`benchmark_all.py`。


| 指标        | 数值              |
| --------- | --------------- |
| Top-1 命中率 | 63%（19/30）      |
| Top-3 命中率 | 87%（26/30）      |
| Top-5 命中率 | 93%（28/30）      |
| 平均检索延迟    | 1.8 ms（100 篇文档） |




### 安全评分

41 个场景，覆盖正常操作、连续工具调用、沙盒越权和提示词越狱。


| 指标      | 数值    |
| ------- | ----- |
| 通过场景    | 41/41 |
| 误报 / 漏报 | 0 / 0 |
| 单次评分延迟  | 13 微秒 |




### 两段式调用

20 个场景。每个场景有一个名称普通、说明书标明高风险的工具，用来比较单阶段调用和两段式调用。

```bash
python tests/test_two_phase_skills.py
```



### 技能加载


| 技能数量 | 首次扫描    | 缓存命中    |
| ---- | ------- | ------- |
| 10   | 0.35 ms | 0.04 ms |
| 100  | 2.99 ms | 0.42 ms |


---



## 项目结构

```text
NexusAgent/
├── nexusagent/core/
│   ├── agent.py               # 主循环
│   ├── multi_agent.py         # 多 Agent
│   ├── rag_engine.py          # TF-IDF 检索
│   ├── security_scorer.py     # 风险评分
│   ├── sandbox_protocol.py    # 沙盒规则
│   ├── context.py             # 上下文与摘要
│   ├── provider.py            # 模型适配
│   ├── skill_loader.py        # 技能加载
│   ├── heartbeat.py           # 定时任务
│   ├── logger.py              # 审计日志
│   ├── bus.py                 # 消息队列
│   ├── config.py              # 配置
│   └── tools/                 # 内置工具
├── entry/
│   ├── cli.py                 # config / run / monitor
│   ├── main.py                # 终端界面
│   └── monitor.py             # 实时日志
├── tests/
├── workspace/
└── docs/
```

单元测试不需要 API Key。

```bash
pip install pytest
pytest tests/ --ignore=tests/test_two_phase_skills.py
python tests/benchmark_all.py
```

---



## 许可证

MIT License