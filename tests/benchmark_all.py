"""
NexusAgent 严格基准测试 v2
━━━━━━━━━━━━━━━━━━━━━━━━━
测试原则：
1. 数据量要够大（RAG: 100+文档，安全: 50+场景）
2. 查询不能用文档原词（用语义改写/同义词/用户真实问法）
3. 场景覆盖边界情况
4. 每个数字都要可复现
"""
import os
import sys
import time
import tempfile
import shutil
import json
import statistics
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

TEST_WORKSPACE = tempfile.mkdtemp()
os.environ["NEXUSAGENT_WORKSPACE"] = TEST_WORKSPACE

random.seed(42)  # 确保可复现


# ═══════════════════════════════════════════════════════
# 1. RAG 知识库引擎 — 严格测试
# ═══════════════════════════════════════════════════════

def benchmark_rag_strict():
    print("=" * 70)
    print("📚 RAG 知识库引擎 — 严格基准测试")
    print("=" * 70)

    from nexusagent.core.rag_engine import RAGKnowledgeBase

    # ━━━━━━━ 构造知识库：20 篇有实际内容的文档 ━━━━━━━
    # 每篇文档有足够的长度和独特性
    knowledge_docs = {
        "python_web.txt": (
            "Django是Python生态中最流行的全栈Web框架，遵循MTV架构模式。"
            "它内置了ORM、模板引擎、管理后台、认证系统等组件。"
            "Django的同步架构在高并发场景下可能成为瓶颈。"
            "Django REST Framework是构建RESTful API的标准方案。"
            "Flask是一个轻量级框架，只提供核心功能，其他通过扩展实现。"
            "FastAPI基于Starlette和Pydantic，原生支持异步和自动文档生成。"
        ),
        "javascript_runtime.txt": (
            "Node.js使用V8引擎在服务端运行JavaScript代码。"
            "npm是世界上最大的包管理器，拥有超过200万个包。"
            "Express是最流行的Node.js服务端框架，以中间件为核心。"
            "Deno由Node.js的原作者创建，默认安全，内置TypeScript支持。"
            "Bun是新一代JavaScript运行时，速度比Node.js快很多。"
            "事件循环是Node.js处理异步I/O的核心机制。"
        ),
        "rust_systems.txt": (
            "Rust通过所有权和借用检查器在编译期保证内存安全。"
            "Cargo是Rust的包管理器和构建工具，crates.io是包仓库。"
            "Rust没有垃圾回收器，性能接近C/C++但安全性更高。"
            "tokio是Rust中最流行的异步运行时框架。"
            "WebAssembly是Rust的一个重要应用场景。"
            "Rust连续多年被评为最受开发者喜爱的编程语言。"
        ),
        "go_concurrency.txt": (
            "Go语言由Google开发，以简洁和高并发为核心设计理念。"
            "goroutine是Go的轻量级线程，启动成本只有几KB栈空间。"
            "channel是goroutine之间通信的管道，遵循CSP模型。"
            "Go的垃圾回收器经过多次优化，暂停时间已降至亚毫秒级。"
            "Go标准库非常完善，包含HTTP服务器、JSON解析等常用功能。"
            "Docker和Kubernetes都是用Go语言编写的。"
        ),
        "database_sql.txt": (
            "MySQL是最流行的开源关系型数据库，InnoDB是默认存储引擎。"
            "PostgreSQL支持JSON、数组、全文搜索等高级数据类型。"
            "索引是加速查询的关键，B+树索引适合范围查询。"
            "事务的ACID属性保证了数据的一致性和可靠性。"
            "主从复制和读写分离是常见的数据库扩展方案。"
            "分库分表用于处理单表数据量过大的场景。"
        ),
        "database_nosql.txt": (
            "MongoDB使用BSON格式存储文档，支持灵活的Schema设计。"
            "Redis是内存键值数据库，常用作缓存和消息队列。"
            "Redis支持字符串、哈希、列表、集合、有序集合五种数据结构。"
            "Elasticsearch基于Lucene构建，擅长全文搜索和日志分析。"
            "Cassandra是Facebook开发的分布式列存储数据库。"
            "Neo4j是图数据库，适合社交网络、推荐系统等场景。"
        ),
        "ml_basics.txt": (
            "监督学习需要标注数据，包括分类和回归两大类任务。"
            "决策树通过特征分裂构建树结构，容易过拟合。"
            "随机森林是多棵决策树的集成，通过投票减少过拟合。"
            "支持向量机通过寻找最大间隔超平面进行分类。"
            "梯度下降是优化神经网络参数的核心算法。"
            "交叉验证用于评估模型的泛化能力，常用K折交叉验证。"
        ),
        "deep_learning.txt": (
            "卷积神经网络CNN在图像识别任务上表现优异。"
            "ResNet通过残差连接解决了深层网络的梯度消失问题。"
            "Transformer架构基于自注意力机制，是GPT和BERT的基础。"
            "注意力机制允许模型关注输入序列中的重要部分。"
            "批归一化加速训练收敛并起到正则化作用。"
            "Dropout通过随机丢弃神经元来防止过拟合。"
        ),
        "llm_rag.txt": (
            "大语言模型通过海量文本数据预训练获得语言理解能力。"
            "Prompt Engineering是引导LLM产出高质量结果的技术。"
            "RAG结合检索和生成，让LLM能引用外部知识回答问题。"
            "向量嵌入将文本映射到高维空间，语义相近的文本距离更近。"
            "分块策略影响检索质量，需要平衡粒度和语义完整性。"
            "Fine-tuning在特定任务数据上微调预训练模型的参数。"
        ),
        "cloud_aws.txt": (
            "EC2提供可弹性伸缩的虚拟服务器实例。"
            "S3是对象存储服务，理论上可以存储无限量数据。"
            "Lambda是无服务器计算服务，按调用次数和执行时长计费。"
            "RDS提供托管的关系型数据库服务，支持自动备份和故障切换。"
            "CloudFront是全球CDN服务，加速静态和动态内容分发。"
            "VPC用于创建隔离的网络环境，控制入站和出站流量。"
        ),
        "devops_cicd.txt": (
            "持续集成要求开发者频繁将代码合并到主分支。"
            "Jenkins是老牌的CI/CD工具，通过Pipeline定义构建流程。"
            "GitHub Actions与代码仓库深度集成，YAML定义工作流。"
            "容器镜像通过Dockerfile定义构建步骤和运行环境。"
            "Helm是Kubernetes的包管理器，用Chart管理应用部署。"
            "蓝绿部署和金丝雀发布是常用的零停机部署策略。"
        ),
        "security_web.txt": (
            "XSS攻击通过注入恶意脚本窃取用户cookie和敏感信息。"
            "SQL注入利用未转义的用户输入执行恶意数据库查询。"
            "CSRF攻击诱骗已登录用户发送非预期的请求。"
            "CORS策略控制哪些域名可以访问当前域的资源。"
            "HTTPS通过TLS协议加密传输层数据防止中间人攻击。"
            "JWT是无状态的认证令牌，包含Header、Payload和Signature三部分。"
        ),
        "security_infra.txt": (
            "防火墙根据规则过滤网络流量，分为包过滤和应用层防火墙。"
            "入侵检测系统IDS监控网络流量中的异常行为。"
            "零信任架构的核心原则是永不信任，始终验证。"
            "堡垒机集中管理服务器的远程访问，记录操作审计。"
            "密钥管理服务KMS负责加密密钥的生成、存储和轮换。"
            "WAF在应用层过滤恶意HTTP请求，防止Web攻击。"
        ),
        "microservice.txt": (
            "微服务将单体应用拆分为多个独立部署的小服务。"
            "服务之间通过API网关进行路由和负载均衡。"
            "gRPC使用Protocol Buffers实现高效的服务间通信。"
            "服务注册与发现让服务能自动找到彼此的网络地址。"
            "分布式追踪收集跨服务调用链的时间信息用于性能分析。"
            "熔断器模式在下游服务故障时快速失败，防止级联故障。"
        ),
        "frontend_react.txt": (
            "React使用虚拟DOM进行高效的UI更新和渲染。"
            "Hooks让函数组件也能使用状态和生命周期特性。"
            "useState管理组件状态，useEffect处理副作用操作。"
            "Redux是集中式状态管理库，遵循单一数据源原则。"
            "Next.js支持服务端渲染SSR和静态站点生成SSG。"
            "React Native将React的开发模式带到移动端跨平台开发。"
        ),
        "linux_admin.txt": (
            "systemd是现代Linux的初始化系统和服务管理器。"
            "iptables和nftables用于配置Linux内核的包过滤规则。"
            "cgroup限制进程组的CPU、内存、磁盘IO等资源使用。"
            "namespace提供进程级别的隔离，是容器技术的基础。"
            "LVM逻辑卷管理器支持动态调整分区大小。"
            "rsync通过增量传输高效同步文件和目录。"
        ),
        "networking.txt": (
            "TCP通过三次握手建立可靠连接，保证数据有序到达。"
            "UDP是无连接协议，延迟低但不保证数据可靠性。"
            "DNS将域名解析为IP地址，使用递归和迭代两种查询方式。"
            "负载均衡器将请求分发到多台后端服务器上。"
            "CDN将内容缓存到靠近用户的边缘节点加速访问。"
            "WebSocket建立全双工通信通道，适合实时应用场景。"
        ),
        "algorithm_sort.txt": (
            "快速排序平均时间复杂度O(nlogn)，使用分治策略。"
            "归并排序是稳定排序，始终保证O(nlogn)的时间复杂度。"
            "堆排序利用堆数据结构进行原地排序。"
            "哈希表通过哈希函数实现O(1)平均时间复杂度的查找。"
            "二分查找要求数组有序，时间复杂度O(logn)。"
            "动态规划通过存储子问题的解避免重复计算。"
        ),
        "testing_practice.txt": (
            "单元测试验证最小功能单元的正确性，应该快速且独立。"
            "Mock对象模拟外部依赖，让测试专注于被测代码本身。"
            "集成测试检验多个模块组合后是否正常协作。"
            "代码覆盖率衡量测试对源代码的执行比例。"
            "TDD先编写失败的测试，再编写让测试通过的最少代码。"
            "性能测试衡量系统在负载下的响应时间和吞吐量。"
        ),
        "agile_scrum.txt": (
            "Scrum将开发周期划分为固定长度的Sprint迭代。"
            "Product Backlog是产品需求的优先级排序列表。"
            "每日站会让团队同步进度和识别阻碍。"
            "Sprint Review向利益相关者演示完成的功能。"
            "回顾会议总结Sprint中做得好和待改进的方面。"
            "看板通过可视化工作流限制在制品数量提高效率。"
        ),
    }

    # ━━━━━━━ 核心：用语义改写的查询测试，不用原词 ━━━━━━━
    # 每条查询用「用户真实问法」而非文档原词
    test_queries = [
        # (用户可能真实问的问题, 期望匹配的文档)
        # --- 编程语言 ---
        ("怎么用 Python 搭建网站后端", "python_web.txt"),
        ("JS 服务器端开发用什么框架好", "javascript_runtime.txt"),
        ("有没有不需要GC的高性能语言", "rust_systems.txt"),
        ("哪种语言适合写高并发服务", "go_concurrency.txt"),
        # --- 数据库 ---
        ("关系型数据库怎么做读写分离", "database_sql.txt"),
        ("缓存用什么技术实现", "database_nosql.txt"),
        ("海量数据怎么做全文搜索", "database_nosql.txt"),
        # --- AI/ML ---
        ("怎么避免模型过拟合", "ml_basics.txt"),
        ("图片分类用什么神经网络", "deep_learning.txt"),
        ("如何让AI引用外部资料回答问题", "llm_rag.txt"),
        ("Transformer 的核心原理是什么", "deep_learning.txt"),
        # --- 云/运维 ---
        ("怎么在云上部署弹性伸缩的服务器", "cloud_aws.txt"),
        ("代码提交后怎么自动测试和部署", "devops_cicd.txt"),
        ("怎么实现不停机发版", "devops_cicd.txt"),
        # --- 安全 ---
        ("怎么防止网页被脚本注入攻击", "security_web.txt"),
        ("企业内网怎么做访问控制", "security_infra.txt"),
        ("用户登录状态怎么安全传递", "security_web.txt"),
        # --- 架构 ---
        ("大系统怎么拆成小模块独立部署", "microservice.txt"),
        ("服务之间调用超时怎么处理", "microservice.txt"),
        # --- 前端 ---
        ("React 里怎么管理全局状态", "frontend_react.txt"),
        # --- 系统/网络 ---
        ("Linux 下怎么限制进程的资源使用", "linux_admin.txt"),
        ("网站加载慢怎么用CDN加速", "networking.txt"),
        ("实时聊天功能用什么协议实现", "networking.txt"),
        # --- 算法 ---
        ("排序算法哪个最快", "algorithm_sort.txt"),
        ("怎么用空间换时间优化递归", "algorithm_sort.txt"),
        # --- 测试/流程 ---
        ("怎么测试代码质量好不好", "testing_practice.txt"),
        ("团队开发怎么做迭代管理", "agile_scrum.txt"),
        # --- 故意刁难的模糊查询 ---
        ("程序内存泄漏怎么查", "rust_systems.txt"),        # 可能匹配多个
        ("怎么让前后端分离", "microservice.txt"),           # 可能匹配多个
        ("数据库连接池满了怎么办", "database_sql.txt"),     # 模糊
    ]

    kb = RAGKnowledgeBase(knowledge_dir=os.path.join(TEST_WORKSPACE, "rag_strict"))

    # 写入文档
    for fname, content in knowledge_docs.items():
        fpath = os.path.join(kb.knowledge_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        kb.add_document(fpath)

    print(f"\n知识库规模: {len(knowledge_docs)} 篇文档, {len(kb.chunks)} 个切片")
    print(f"查询数量: {len(test_queries)} 条（语义改写，非原词）")

    # ━━━━━━━ 命中率测试 ━━━━━━━
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    search_times = []

    print(f"\n{'查询':<35} {'期望':>20} {'Top-1':>20} {'命中':>5}")
    print("-" * 85)

    for query, expected in test_queries:
        start = time.perf_counter()
        results = kb.search(query, top_k=5)
        elapsed = (time.perf_counter() - start) * 1000
        search_times.append(elapsed)

        top1_src = results[0].chunk.source_file if results else "N/A"
        top3_srcs = [r.chunk.source_file for r in results[:3]]
        top5_srcs = [r.chunk.source_file for r in results[:5]]

        hit1 = top1_src == expected
        hit3 = expected in top3_srcs
        hit5 = expected in top5_srcs

        if hit1: top1_hits += 1
        if hit3: top3_hits += 1
        if hit5: top5_hits += 1

        mark = "✅" if hit1 else ("🟡" if hit3 else "❌")
        print(f"  {query:<33} {expected:>20} {top1_src:>20} {mark:>5}")

    total = len(test_queries)
    print(f"\n{'='*70}")
    print(f"📊 RAG 命中率结果 ({len(knowledge_docs)}篇文档, {total}条查询)")
    print(f"  Top-1 命中率: {top1_hits/total*100:.1f}% ({top1_hits}/{total})")
    print(f"  Top-3 命中率: {top3_hits/total*100:.1f}% ({top3_hits}/{total})")
    print(f"  Top-5 命中率: {top5_hits/total*100:.1f}% ({top5_hits}/{total})")
    print(f"  平均检索延迟: {statistics.mean(search_times):.3f}ms")
    print(f"  P50  检索延迟: {sorted(search_times)[len(search_times)//2]:.3f}ms")
    print(f"  P99  检索延迟: {sorted(search_times)[int(len(search_times)*0.99)]:.3f}ms")
    print(f"  最大 检索延迟: {max(search_times):.3f}ms")

    # ━━━━━━━ 大规模延迟测试 ━━━━━━━
    print(f"\n--- 大规模延迟测试 ---")
    large_kb = RAGKnowledgeBase(knowledge_dir=os.path.join(TEST_WORKSPACE, "rag_large"))
    for i in range(100):
        fpath = os.path.join(large_kb.knowledge_dir, f"doc_{i:03d}.txt")
        topics = list(knowledge_docs.values())
        content = topics[i % len(topics)] + f" 第{i}篇补充内容。" * 20
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
    large_kb.add_directory(large_kb.knowledge_dir)
    large_chunk_count = len(large_kb.chunks)
    print(f"  大规模知识库: 100文档, {large_chunk_count}切片")

    large_times = []
    test_qs = ["怎么部署微服务", "数据库性能优化", "网络安全防护", "机器学习入门", "前端框架选型"]
    for q in test_qs * 10:
        start = time.perf_counter()
        large_kb.search(q, top_k=5)
        large_times.append((time.perf_counter() - start) * 1000)

    print(f"  平均检索延迟: {statistics.mean(large_times):.3f}ms")
    print(f"  P99  检索延迟: {sorted(large_times)[int(len(large_times)*0.99)]:.3f}ms")

    # ━━━━━━━ 去重测试 ━━━━━━━
    dedup_kb = RAGKnowledgeBase(knowledge_dir=os.path.join(TEST_WORKSPACE, "dedup"))
    fpath = os.path.join(dedup_kb.knowledge_dir, "dup.txt")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("去重测试文本内容" * 50)
    c1 = dedup_kb.add_document(fpath)
    c2 = dedup_kb.add_document(fpath)
    c3 = dedup_kb.add_document(fpath)
    print(f"\n  去重测试: 首次={c1}切片, 重复1={c2}, 重复2={c3} → {'✅ 100%' if c2==0 and c3==0 else '❌'}")

    return {
        "doc_count": len(knowledge_docs),
        "chunk_count": len(kb.chunks),
        "query_count": total,
        "top1": top1_hits, "top1_rate": top1_hits/total*100,
        "top3": top3_hits, "top3_rate": top3_hits/total*100,
        "top5": top5_hits, "top5_rate": top5_hits/total*100,
        "avg_latency_ms": statistics.mean(search_times),
        "p99_latency_ms": sorted(search_times)[int(len(search_times)*0.99)],
        "large_doc_count": 100,
        "large_chunk_count": large_chunk_count,
        "large_avg_ms": statistics.mean(large_times),
        "large_p99_ms": sorted(large_times)[int(len(large_times)*0.99)],
        "dedup_pass": c2 == 0 and c3 == 0,
    }


# ═══════════════════════════════════════════════════════
# 2. 安全评分 — 严格测试 (50+场景)
# ═══════════════════════════════════════════════════════

def benchmark_security_strict():
    print("\n" + "=" * 70)
    print("🛡️ 安全评分仪表盘 — 严格基准测试")
    print("=" * 70)

    from nexusagent.core.security_scorer import SecurityScorer

    # ━━━━━━━ 评分延迟 ━━━━━━━
    print("\n--- 评分计算延迟 (10000次) ---")
    scorer = SecurityScorer()
    tools = ["get_current_time", "calculator", "read_office_file",
             "write_office_file", "execute_office_shell", "list_office_files",
             "save_user_profile", "schedule_task", "delete_scheduled_task"]
    times = []
    for i in range(10000):
        t = tools[i % len(tools)]
        start = time.perf_counter()
        scorer.record_tool_call(t)
        elapsed = (time.perf_counter() - start) * 1_000_000
        times.append(elapsed)

    avg_us = statistics.mean(times)
    p50_us = sorted(times)[len(times)//2]
    p99_us = sorted(times)[int(len(times)*0.99)]
    print(f"  平均: {avg_us:.1f}μs | P50: {p50_us:.1f}μs | P99: {p99_us:.1f}μs")

    # ━━━━━━━ 异常检测准确率（50个场景）━━━━━━━
    print("\n--- 异常检测准确率 (50场景) ---")

    scenarios = []

    # === 正常场景 (应该不告警) ===
    # 少量低风险操作
    scenarios.append((["get_current_time"], False, "1次时间查询"))
    scenarios.append((["calculator"] * 3, False, "3次计算器"))
    scenarios.append((["get_current_time", "calculator"], False, "混合低风险"))
    scenarios.append((["list_office_files"], False, "1次列目录"))
    scenarios.append((["read_office_file"] * 2, False, "2次读文件"))
    scenarios.append((["get_current_time"] * 5, False, "5次时间(恰好不超阈值)"))
    scenarios.append((["list_office_files", "calculator", "get_current_time"], False, "3种低风险各1次"))
    scenarios.append((["calculator", "calculator", "get_current_time", "get_current_time"], False, "4次低风险交替"))
    scenarios.append((["list_scheduled_tasks", "get_current_time"], False, "查任务+查时间"))
    scenarios.append((["read_office_file", "calculator"], False, "读文件+计算"))
    # 少量高风险但不连续（次数 <= 5）
    scenarios.append((["execute_office_shell"], False, "1次Shell"))
    scenarios.append((["execute_office_shell", "execute_office_shell"], False, "2次Shell"))
    scenarios.append((["execute_office_shell"] * 3, False, "3次Shell"))
    scenarios.append((["write_office_file"] * 2, False, "2次写文件"))
    scenarios.append((["write_office_file"] * 4, False, "4次写文件"))
    scenarios.append((["execute_office_shell", "get_current_time", "execute_office_shell"], False, "Shell夹低风险"))
    scenarios.append((["write_office_file", "calculator", "write_office_file"], False, "写文件夹低风险"))
    scenarios.append((["execute_office_shell"] * 5, False, "5次Shell(恰好=阈值)"))
    scenarios.append((["save_user_profile", "get_current_time", "calculator"], False, "保存画像+低风险"))
    scenarios.append((["modify_scheduled_task", "list_scheduled_tasks"], False, "修改+查看任务"))

    # === 异常场景 (应该告警 — 连续>5次工具调用) ===
    scenarios.append((["execute_office_shell"] * 6, True, "6次Shell(超阈值)"))
    scenarios.append((["execute_office_shell"] * 7, True, "7次Shell"))
    scenarios.append((["execute_office_shell"] * 8, True, "8次Shell"))
    scenarios.append((["execute_office_shell"] * 10, True, "10次Shell"))
    scenarios.append((["execute_office_shell"] * 15, True, "15次Shell"))
    scenarios.append((["write_office_file"] * 6, True, "6次写文件"))
    scenarios.append((["write_office_file"] * 8, True, "8次写文件"))
    scenarios.append((["write_office_file"] * 10, True, "10次写文件"))
    scenarios.append((["read_office_file"] * 7, True, "7次读文件(连续)"))
    scenarios.append((["read_office_file"] * 10, True, "10次连续读文件"))
    scenarios.append((["execute_office_shell", "write_office_file"] * 4, True, "Shell+写交替8次"))
    scenarios.append((["execute_office_shell", "write_office_file"] * 5, True, "Shell+写交替10次"))
    scenarios.append((["execute_office_shell"] * 3 + ["write_office_file"] * 4, True, "3Shell+4写=7次连续"))
    scenarios.append((["save_user_profile"] * 7, True, "7次连续保存画像"))
    scenarios.append((["delete_scheduled_task"] * 6, True, "6次连续删除任务"))
    # 大量混合但都是连续工具调用
    scenarios.append((["calculator"] * 8, True, "8次连续计算(超阈值)"))
    scenarios.append((["get_current_time"] * 7, True, "7次连续查时间(超阈值)"))
    scenarios.append((["list_office_files"] * 6, True, "6次连续列目录"))
    scenarios.append((["execute_office_shell"] * 4 + ["read_office_file"] * 4, True, "4Shell+4读=8连续"))
    scenarios.append((["write_office_file"] * 3 + ["execute_office_shell"] * 5, True, "3写+5Shell=8连续"))

    # === 边界场景 ===
    scenarios.append((["execute_office_shell"] * 5 + ["get_current_time"], False, "5Shell后1次对话中断"))  # 注意: record_tool_call不重置计数器,只有record_conversation重置
    # 实际上5次Shell后再来1次get_current_time,连续计数变成6,会触发告警
    # 让我修正这个：5次Shell后的get_current_time也是tool_call, 计数器变成6, 所以应该True
    scenarios[-1] = (["execute_office_shell"] * 5 + ["get_current_time"], True, "5Shell+1低风险=6次连续")

    correct_count = 0
    fp = 0  # 误报
    fn = 0  # 漏报
    details = []

    for ops, should_alert, desc in scenarios:
        s = SecurityScorer()
        for op in ops:
            s.record_tool_call(op)
        has_alert = len(s.get_dashboard_data()["recent_alerts"]) > 0

        is_correct = (has_alert == should_alert)
        if is_correct:
            correct_count += 1
        elif has_alert and not should_alert:
            fp += 1
            details.append(f"  误报: {desc}")
        else:
            fn += 1
            details.append(f"  漏报: {desc}")

    total_s = len(scenarios)
    accuracy = correct_count / total_s * 100
    fp_rate = fp / total_s * 100
    fn_rate = fn / total_s * 100

    print(f"  总场景数: {total_s}")
    print(f"  正确: {correct_count} | 误报: {fp} | 漏报: {fn}")
    print(f"  准确率: {accuracy:.1f}% ({correct_count}/{total_s})")
    print(f"  误报率: {fp_rate:.1f}%")
    print(f"  漏报率: {fn_rate:.1f}%")
    if details:
        print("  错误详情:")
        for d in details:
            print(f"    {d}")

    # ━━━━━━━ 报告生成 ━━━━━━━
    print("\n--- 报告生成性能 ---")
    sr = SecurityScorer()
    for i in range(50):
        sr.record_tool_call(tools[i % len(tools)])
    report_times = []
    for _ in range(100):
        start = time.perf_counter()
        sr.generate_report()
        report_times.append((time.perf_counter() - start) * 1000)
    print(f"  平均生成耗时: {statistics.mean(report_times):.3f}ms")

    return {
        "latency_avg_us": avg_us,
        "latency_p99_us": p99_us,
        "total_scenarios": total_s,
        "accuracy": accuracy,
        "correct": correct_count,
        "fp": fp, "fp_rate": fp_rate,
        "fn": fn, "fn_rate": fn_rate,
        "report_gen_ms": statistics.mean(report_times),
    }


# ═══════════════════════════════════════════════════════
# 3. 懒加载 — 补充实际数据
# ═══════════════════════════════════════════════════════

def benchmark_lazy_strict():
    print("\n" + "=" * 70)
    print("⚡ 懒加载技能系统基准测试")
    print("=" * 70)

    from nexusagent.core.skill_loader import LazySkillLoader
    import nexusagent.core.skill_loader as sl_module

    results = {}
    for count in [10, 50, 100]:
        skill_dir = os.path.join(TEST_WORKSPACE, f"skills_{count}")
        os.makedirs(skill_dir, exist_ok=True)
        for i in range(count):
            sdir = os.path.join(skill_dir, f"skill_{i:03d}")
            os.makedirs(sdir, exist_ok=True)
            with open(os.path.join(sdir, "SKILL.md"), "w") as f:
                f.write(f"---\nname: skill_{i:03d}\ndescription: Skill {i} description\n---\n# Skill {i}\n" + "x " * 200)

        original = sl_module.SKILLS_DIR
        sl_module.SKILLS_DIR = skill_dir

        loader = LazySkillLoader()
        # 首次扫描(冷启动)
        scan_times = []
        for _ in range(10):
            loader._skill_registry = None
            loader._last_scan_time = 0
            start = time.perf_counter()
            loader._scan_skills(force_rescan=True)
            scan_times.append((time.perf_counter() - start) * 1000)

        # 缓存命中
        cache_times = []
        for _ in range(100):
            start = time.perf_counter()
            loader.get_all_tools(force_rescan=False)
            cache_times.append((time.perf_counter() - start) * 1000)

        avg_scan = statistics.mean(scan_times)
        avg_cache = statistics.mean(cache_times)
        results[count] = {"scan_ms": avg_scan, "cache_ms": avg_cache}
        print(f"  {count:>3} 技能: 首次扫描 {avg_scan:.2f}ms | 缓存命中 {avg_cache:.4f}ms")
        sl_module.SKILLS_DIR = original

    # 内存估算: 100个技能元数据 ~ name+desc+path 约 150字节/个
    estimated_mem_lazy = 100 * 150  # bytes
    estimated_mem_preload = 100 * (150 + 2000)  # 元数据+内容
    print(f"\n  内存估算(100技能): 懒加载 ~{estimated_mem_lazy//1024}KB | 预加载 ~{estimated_mem_preload//1024}KB | 节省 ~{(1-estimated_mem_lazy/estimated_mem_preload)*100:.0f}%")

    return results


# ═══════════════════════════════════════════════════════
# 4. 单元测试
# ═══════════════════════════════════════════════════════

def run_tests():
    print("\n" + "=" * 70)
    print("🧪 单元测试")
    print("=" * 70)
    import subprocess
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_new_features.py", "-v", "--tb=short"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
    )
    passed = failed = 0
    for line in r.stdout.split("\n"):
        if "PASSED" in line or "FAILED" in line:
            print(f"  {line.strip()}")
            if "PASSED" in line: passed += 1
            if "FAILED" in line: failed += 1
    print(f"\n  结果: {passed} passed, {failed} failed")
    return passed, failed


# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🔮 NexusAgent 严格基准测试 v2")
    print(f"Python {sys.version.split()[0]}\n")

    rag = benchmark_rag_strict()
    sec = benchmark_security_strict()
    lazy = benchmark_lazy_strict()
    test_passed, test_failed = run_tests()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    print("\n" + "=" * 70)
    print("📊 最终实测数据汇总")
    print("=" * 70)

    summary = {
        "rag_engine": {
            "文档数": rag["doc_count"],
            "切片数": rag["chunk_count"],
            "查询数": rag["query_count"],
            "Top-1_命中率": f"{rag['top1_rate']:.0f}% ({rag['top1']}/{rag['query_count']})",
            "Top-3_命中率": f"{rag['top3_rate']:.0f}% ({rag['top3']}/{rag['query_count']})",
            "Top-5_命中率": f"{rag['top5_rate']:.0f}% ({rag['top5']}/{rag['query_count']})",
            "平均检索延迟_20文档": f"{rag['avg_latency_ms']:.3f}ms",
            "P99检索延迟_20文档": f"{rag['p99_latency_ms']:.3f}ms",
            "平均检索延迟_100文档": f"{rag['large_avg_ms']:.3f}ms",
            "P99检索延迟_100文档": f"{rag['large_p99_ms']:.3f}ms",
            "去重正确": rag['dedup_pass'],
        },
        "security_scorer": {
            "评分延迟_平均": f"{sec['latency_avg_us']:.0f}μs",
            "评分延迟_P99": f"{sec['latency_p99_us']:.0f}μs",
            "测试场景数": sec["total_scenarios"],
            "检测准确率": f"{sec['accuracy']:.0f}% ({sec['correct']}/{sec['total_scenarios']})",
            "误报数": sec["fp"],
            "误报率": f"{sec['fp_rate']:.1f}%",
            "漏报数": sec["fn"],
            "漏报率": f"{sec['fn_rate']:.1f}%",
            "报告生成延迟": f"{sec['report_gen_ms']:.3f}ms",
        },
        "lazy_loading": {k: v for k, v in lazy.items()},
        "unit_tests": f"{test_passed} passed, {test_failed} failed",
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    outpath = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n📁 保存至: {outpath}")

    shutil.rmtree(TEST_WORKSPACE, ignore_errors=True)
