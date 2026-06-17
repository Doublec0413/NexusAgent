# NexusAgent 部署运行指南

> 本指南覆盖从零开始到完整运行的全流程，包括开发环境、生产部署、常见问题排查。

---

## 1. 环境要求

| 要求 | 最低版本 | 推荐版本 |
|------|---------|---------|
| Python | 3.10 | 3.11+ |
| pip | 21.0 | 最新 |
| 操作系统 | Win10 / Ubuntu 20.04 / macOS 12 | Ubuntu 22.04 |
| 内存 | 512MB | 2GB+ |
| 磁盘 | 100MB | 500MB+ (含知识库) |

### LLM API 要求（选一）

| 提供商 | 模型示例 | API Key 变量 |
|--------|---------|-------------|
| 阿里云 DashScope | qwen-max, glm-5 | `OPENAI_API_KEY` |
| OpenAI | gpt-4o-mini | `OPENAI_API_KEY` |
| Anthropic | claude-3.5-sonnet | `ANTHROPIC_API_KEY` |
| 腾讯混元 | hunyuan-pro | `OPENAI_API_KEY` |
| Z.AI (智谱) | glm-4 | `OPENAI_API_KEY` |
| Ollama (本地) | llama3, qwen2 | 无需 API Key |

---

## 2. 安装步骤

### 方式一：标准安装

```bash
# 1. 克隆项目
git clone <your-repo>/NexusAgent.git
cd NexusAgent

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. 安装项目（自动安装所有依赖）
pip install -e .

# 4. 验证安装
nexusagent --help
```

### 方式二：手动安装依赖

```bash
pip install -r requirements.txt
```

### 依赖列表说明

```
# CLI 交互
typer>=0.9.0              # CLI 框架
questionary>=2.0.0        # 交互式选择
rich>=13.0.0              # 终端美化
prompt_toolkit>=3.0.0     # 异步输入

# LangChain 生态
langchain-core>=0.2.0     # 核心抽象
langgraph>=0.1.0          # 状态图编排
langchain-openai>=0.1.0   # OpenAI 适配
openai>=1.0.0             # OpenAI SDK

# 数据存储
aiosqlite>=0.19.0         # 异步 SQLite
langgraph-checkpoint-sqlite>=3.0.0  # 检查点存储

# 工具
python-dotenv>=1.0.0      # 环境变量管理
pydantic>=2.0.0           # 数据验证
```

---

## 3. 配置

### 方式一：交互式向导（推荐）

```bash
nexusagent config
```

向导流程：
1. 选择模型提供商（上下键选择）
2. 输入模型名称（如 `qwen-max`）
3. 输入 API Key（密文输入，不显示）
4. 输入 Base URL（可选，直连回车跳过）
5. 自动连接测试 → 成功后保存配置

### 方式二：手动配置

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```bash
# ===== 必填 =====
DEFAULT_PROVIDER=aliyun          # 提供商名称
DEFAULT_MODEL=qwen-max           # 模型名称
OPENAI_API_KEY=sk-your-key-here  # API Key

# ===== 可选 =====
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 各提供商配置示例

#### 阿里云 DashScope

```bash
DEFAULT_PROVIDER=aliyun
DEFAULT_MODEL=qwen-max
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
```

#### OpenAI

```bash
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-xxxxxxxx
```

#### Ollama（本地模型，无需API Key）

```bash
DEFAULT_PROVIDER=ollama
DEFAULT_MODEL=qwen2
OLLAMA_BASE_URL=http://localhost:11434
```

---

## 4. 运行

### 启动主程序

```bash
nexusagent run
```

启动后你会看到 ASCII Art LOGO + 交互式对话界面。

### 启动监控终端（可选）

在另一个终端窗口运行：

```bash
nexusagent monitor
```

实时查看 5 类审计事件。

### 对话示例

```
  ❯ 现在几点了？
  ● Tool Call: get_current_time
  ❯ 当前本地系统时间是: 2026-06-17 14:30:00

  ❯ 帮我算一下 25 * 48
  ● Tool Call: calculator
  ❯ 表达式 '25 * 48' 的计算结果是: 1200

  ❯ 每天早上8点提醒我喝水
  ● Tool Call: get_current_time
  ● Tool Call: schedule_task
  ❯ 任务已加入队列。循环模式：daily

  ❯ 当前风险评分
  ● Tool Call: get_risk_score
  ❯ 🛡️ 风险评分: 5.2/100  风险等级: 🟢 安全

  ❯ 给我看安全报告
  ● Tool Call: get_security_report
  ❯ [生成完整的 Markdown 安全审计报告]

  ❯ /exit
  ✦ 记忆已固化，NexusAgent 进入休眠。
```

---

## 5. RAG 知识库使用

### 导入文档

将文档放入 `workspace/knowledge/` 目录：

```bash
# 支持的格式：.txt .md .py .json .csv .log
cp your_docs/*.md workspace/knowledge/
```

### 通过代码导入

```python
from nexusagent.core.rag_engine import RAGKnowledgeBase

kb = RAGKnowledgeBase()

# 导入单个文件
kb.add_document("path/to/doc.txt")

# 导入整个目录
results = kb.add_directory("path/to/docs/")
print(f"导入结果: {results}")

# 检索测试
results = kb.search("你的问题", top_k=5)
for r in results:
    print(f"[{r.rank}] {r.chunk.source_file} (score: {r.score:.2f})")
```

### 验证效果

导入文档后，在对话中提出相关问题，Agent 会自动引用知识库内容回答。

---

## 6. 测试

### 运行全部测试

```bash
python -m pytest tests/ -v
```

### 运行新功能测试

```bash
python -m pytest tests/test_new_features.py -v
```

### 预期输出

```
tests/test_new_features.py::TestRAGEngine::test_tfidf_tokenize PASSED
tests/test_new_features.py::TestRAGEngine::test_add_and_search_document PASSED
tests/test_new_features.py::TestSecurityDashboard::test_risk_scoring PASSED
tests/test_new_features.py::TestMultiAgentOrchestrator::test_subtask_creation PASSED
...
============================== 19 passed in 0.15s ==============================
```

---

## 7. 常见问题

### Q: 启动报错 "未找到 API Key"

```bash
# 检查 .env 文件是否存在且配置正确
cat .env | grep API_KEY

# 重新配置
nexusagent config
```

### Q: 模型连接超时

```bash
# 检查网络
curl -v https://dashscope.aliyuncs.com

# 如果需要代理
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
```

### Q: 知识库检索结果不准确

```bash
# 检查知识库状态
python -c "
from nexusagent.core.rag_engine import RAGKnowledgeBase
kb = RAGKnowledgeBase()
print(kb.get_stats())
"
```

### Q: 如何添加自定义工具

1. 在 `nexusagent/core/tools/builtins.py` 中添加：

```python
@nexusagent_tool
def my_tool(param: str) -> str:
    """工具描述（LLM 根据此描述决定何时调用）"""
    return f"结果: {param}"
```

2. 添加到 `BUILTIN_TOOLS` 列表中

### Q: 如何在 Docker 中运行

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e .
CMD ["nexusagent", "run"]
```

```bash
docker build -t nexusagent .
docker run -it --env-file .env nexusagent
```

---

## 8. 开发模式

```bash
# 安装开发依赖
pip install -e ".[dev]"
pip install pytest

# 运行测试
python -m pytest tests/ -v

# 代码修改后无需重新安装（-e 模式）
# 直接运行 nexusagent run 即可生效
```
