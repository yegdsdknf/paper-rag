# Paper-RAG 项目优化建议

> 本文档基于当前代码状态重新整理，重点区分“已有基础”和“真实缺口”，并给出可执行的优先级、风险和验收方式。
>
> 生成时间：2026-06-03

---

## 一、当前状态快照

| 维度 | 当前状态 | 判断 |
|---|---|---|
| 包化迁移 | `paper_rag/` 已承载 generation、retrieval、observability、ui、config 等实现；`hybrid_retriever`、`query_expansion`、`reranker`、`context_compression` 已迁入包内，根目录保留兼容薄壳和入口文件 | 已进入最终收口阶段，不是从零开始 |
| 主链路编排 | `rag_pipeline.py` 仍是主要入口，约 500 行，但已委托到 router、context、generation、observability 模块 | 需要继续收敛为 facade，而非大拆大改 |
| 配置管理 | `RagSettings.from_mapping()` 已做必填项和类型转换 | 缺少启动时资源检查和友好诊断 |
| 检索能力 | 已具备混合检索、动态权重、HyDE、Query Expansion、Rerank、Parent Retrieval、Context Compression | 优化重点应转向缓存、阈值校准和回归评估 |
| Web UI | Streamlit 页面可用，已有流式输出、模型切换、上传入库和反馈保存 | 逐 token 刷新和来源展示仍可明显改善 |
| 评估体系 | 已有 benchmark、holdout、baseline runner、eval metrics | 缺少新旧报告自动对比和质量门禁 |
| 文档 | README 与 architecture 已覆盖运行命令、架构边界和迁移历史 | 缺少面向新手的排障 FAQ 和最短上手路径 |

---

## 二、推荐优先级总览

| 优先级 | 优化项 | 核心收益 | 预估风险 | 建议节奏 |
|---|---|---|---|---|
| P0 | 启动诊断与配置资源检查 | 减少“为什么不工作”的时间成本 | 低 | 立即做 |
| P0 | 错误提示友好化 | 降低新手上手门槛 | 低 | 立即做 |
| P0 | Streamlit 流式刷新节流 | 改善页面抖动和交互体验 | 低 | 立即做 |
| P0 | FAQ 与快速开始补齐 | 降低重复排障成本 | 低 | 立即做 |
| P1 | BM25 索引持久化 | 降低检索器初始化成本 | 中 | 先设计失效策略 |
| P1 | Benchmark 回归检测 | 防止检索质量无意退化 | 中 | 和指标报告一起落地 |
| P1 | 来源展示高亮与评分信息 | 提升答案可解释性 | 中 | 注意 Markdown 转义 |
| P2 | Rerank 阈值过滤 | 降低低相关证据噪声 | 中 | 需要 benchmark 校准 |
| P2 | Query Expansion 质量过滤 | 降低发散 query 带来的噪声召回 | 中 | 需要成本收益评估 |
| P2 | Context Compression 打分改进 | 提升送入 LLM 的证据密度 | 中 | 需要防止误删关键上下文 |
| P3 | Embedding 原型预计算 | 缩短检索器冷启动 | 低到中 | 收益较小，后置 |
| P3 | 包化迁移最终收口 | 降低长期维护成本 | 中到高 | 分阶段替换 import |
| P3 | `rag_pipeline.py` 继续瘦身 | 改善可维护性 | 中 | 等核心行为稳定后做 |

---

## 三、P0：立即建议落地

### 3.1 启动诊断与配置资源检查

**现状**

| 已有基础 | 真实缺口 |
|---|---|
| `paper_rag.config.RagSettings` 已能发现必填配置缺失和类型问题 | Chroma 路径、collection、manifest、Ollama 服务、LLM 模型、本地 embedding/reranker 模型等资源问题仍常在运行中暴露 |

**建议方案**

| 项目 | 建议 |
|---|---|
| 新增模块 | `paper_rag/config/diagnostics.py` |
| 新增入口 | `python main.py doctor`，或在 Streamlit 初始化失败时自动调用轻量检查 |
| 检查范围 | `persist_directory` 是否存在、Chroma collection 是否可读、`index_manifest.json` 是否匹配当前配置、embedding 模型是否本地可加载、reranker 路径是否存在、Ollama 是否启动、`llm_model` 是否已拉取 |
| 输出形式 | 按 `OK / WARN / ERROR` 分类，给出下一步命令 |

**已决策边界**

| 主题 | 决策 |
|---|---|
| 目标用户 | 优先服务初学者上手和面试展示，先解决“能否解释性启动”的问题 |
| 第一版深度 | 半动态检查：不跑完整问答链路，但会检查关键组件可用性 |
| CLI 入口 | 使用 `python main.py doctor`，但 `main.py` 只做薄入口，真实逻辑放在 `paper_rag.config.diagnostics` |
| 运行时间预算 | 第一版目标总耗时 10 秒以内；不做硬超时中断，但记录每项检查的 `elapsed_sec`，总耗时超过 10 秒时在 summary 中给 WARN |
| 输出格式 | 默认输出人类可读报告，同时支持 `--json` |
| JSON 输出约束 | `--json` 时 stdout 只能输出纯 JSON；普通日志、第三方噪声和调试信息应走 stderr 或被抑制 |
| 文本报告顺序 | 摘要和阻断项按严重程度在前，详细检查项按执行顺序在后 |
| settings 展示 | JSON 输出完整关键 settings 快照；文本报告只显示 3 到 5 个最关键字段 |
| 数据模型 | 使用 dataclass 表达诊断项，包含技术 `id` 和用户可读 `title`，例如 `DiagnosticCheck(id, title, status, message, suggestion, elapsed_sec, details)` |
| JSON 字段 | 第一版包含 `status`、`exit_code`、`summary`、`checks`、`settings`；每个 check 都包含 `elapsed_sec` |
| settings 快照 | 只输出关键运行字段：`persist_directory`、`collection_name`、`chunk_schema_version`、`embedding_model`、`llm_model`、`enable_rerank`、`reranker_model`、`enable_query_expansion`、`enable_context_compression`、`enable_parent_retrieval` |
| Ollama 检查 | 只请求 `/api/tags`，检查服务连通和 `llm_model` 是否已下载，不触发模型生成 |
| Chroma 检查 | 连接当前 `collection_name`，检查 collection count 是否大于 0 |
| manifest 检查 | 缺失或与当前 `chunk_schema_version` 不一致先设为 WARN，不阻断可用旧库 |
| embedding 检查 | 复用真实设备选择逻辑，初始化本地 embedding，并执行一次短 `embed_query("doctor check")` |
| reranker 检查 | 仅检查 `reranker_model` 路径或本地可定位性；若 `enable_rerank=true` 但不可用，标记 WARN，不实际加载 CrossEncoder |
| 建议文案 | 每个 WARN/ERROR 都必须提供可执行 suggestion |
| 依赖项失败处理 | 上游硬失败时，下游依赖检查保留固定 check id，但标记 WARN 并说明已跳过，避免重复 ERROR |
| Streamlit 集成 | 第一版只做 CLI；后续再复用同一数据模型接入初始化失败页 |
| 单测范围 | 覆盖 dataclass、summary、JSON/text formatter、ERROR/WARN 退出码、checker mock；不把真实 Ollama、Chroma、embedding 放进普通单测 |

**第一版固定检查项**

| id | title | 说明 |
|---|---|---|
| `config.load` | 配置文件 | 读取 `config.yaml` 并构建 `RagSettings` |
| `path.persist_directory` | 向量库目录 | 检查 `persist_directory` 是否存在 |
| `chroma.collection` | Chroma Collection | 检查当前 collection 是否可读且文档数大于 0 |
| `index.manifest` | 索引 Manifest | 检查 manifest 是否存在，并提示 chunk schema 是否一致 |
| `ollama.service` | Ollama 服务 | 检查 `/api/tags` 是否可访问 |
| `ollama.model` | Ollama 模型 | 检查 `llm_model` 是否在 Ollama 模型列表中 |
| `embedding.model` | Embedding 模型 | 初始化本地 embedding 并执行短查询 |
| `reranker.model` | Reranker 模型 | rerank 开启时检查模型可定位；rerank 关闭时显示 OK 并说明跳过 |
| `path.papers` | 论文目录 | 检查 `papers/` 是否存在且包含 PDF；为空时 WARN |
| `diagnostics.elapsed` | 诊断耗时 | 条件项；总耗时超过 10 秒时追加 WARN，提示查看各检查项 `elapsed_sec` |

**ERROR/WARN 划分**

| 等级 | 项目 |
|---|---|
| ERROR | 配置必填项缺失或类型错误、`persist_directory` 不存在、Chroma collection 为空或不可读、Ollama 未连接、`llm_model` 未下载、embedding 本地初始化或短查询失败 |
| WARN | `index_manifest.json` 缺失、manifest 与当前 chunk schema 不一致、`enable_rerank=true` 但 reranker 不可定位、`papers/` 为空 |

**依赖项跳过规则**

| 场景 | 输出 |
|---|---|
| `config.load` 失败 | 只输出 `config.load` ERROR，不继续后续检查 |
| `path.persist_directory` 失败 | `path.persist_directory` 为 ERROR，`chroma.collection` 为 WARN 并说明因目录缺失跳过 |
| `ollama.service` 失败 | `ollama.service` 为 ERROR，`ollama.model` 为 WARN 并说明因服务不可用跳过 |
| `enable_rerank=false` | `reranker.model` 为 OK，说明 Rerank 未启用，无需检查模型 |

**验收方式**

| 场景 | 预期 |
|---|---|
| `persist_directory` 不存在 | 启动前提示需要先运行 `python main.py build` |
| Ollama 未启动 | 提示运行 `ollama serve` |
| LLM 模型未下载 | 提示运行 `ollama pull <model>` |
| reranker 路径不存在且启用 rerank | 提示关闭 rerank 或放置本地模型 |

---

### 3.2 错误提示友好化

**现状**

当前 `_get_llm()` 和 Streamlit 初始化异常会直接暴露底层错误，例如 `Connection refused`、`model not found`、`local model not found`。这些错误对初学者不够直接。

**建议方案**

| 项目 | 建议 |
|---|---|
| 新增函数 | `format_runtime_error(error, settings)` |
| 放置位置 | 优先放在 `paper_rag/ui/errors.py` 或 `paper_rag/config/diagnostics.py` |
| 覆盖错误 | Ollama 未启动、模型未下载、本地 HuggingFace 模型缺失、Chroma collection 为空、向量库未构建、reranker 不可用 |
| UI 使用 | `app.py` 初始化失败、问答失败、上传入库失败统一走格式化文案 |

**已决策边界**

| 主题 | 决策 |
|---|---|
| 第一版覆盖入口 | 只覆盖 Streamlit `app.py` 初始化失败和问答失败；CLI 和 build 入库后续再复用同一 formatter |
| 模块位置 | 放在 `paper_rag/ui/errors.py` |
| 数据模型 | 使用 dataclass，例如 `FriendlyError(title, message, suggestions, details, show_doctor_hint)` |
| 错误模式 | 第一版覆盖 Ollama 连接失败、Ollama 模型缺失、HuggingFace 本地模型缺失、Chroma/向量库未构建或为空、Reranker 不可用、其他兜底异常 |
| Reranker 降级 | 不作为硬失败；UI 显示轻量 warning：“精排模型不可用，已使用原始检索顺序” |
| doctor 引导 | 只在环境类错误和兜底异常中提示运行 `python main.py doctor`；明确可修复错误优先给具体命令 |
| 原始异常展示 | 默认展示简短原始摘要，完整异常放入 Streamlit 折叠详情 |
| 渲染方式 | Streamlit 页面按标题、说明、建议步骤、技术详情分块展示，不直接裸露完整 traceback |

**示例文案**

```text
无法连接到 Ollama 服务。

请检查：
1. 是否已启动 Ollama：ollama serve
2. 是否已下载模型：ollama pull qwen2.5:3b
3. config.yaml 中的 llm_model 是否拼写正确
```

**验收方式**

| 场景 | 预期 |
|---|---|
| Ollama 未启动 | 页面显示可执行排查步骤 |
| 模型未下载 | 页面显示具体 `ollama pull` 命令 |
| embedding 本地模型缺失 | 页面明确说明当前项目默认离线加载模型 |

---

### 3.3 Streamlit 流式刷新节流

**现状**

`app.py` 已使用 `st.empty()`，但仍对每个 token 调用 `placeholder.markdown(answer + "▌")`，长答案时会出现页面抖动和重复渲染成本。

**建议方案**

| 项目 | 建议 |
|---|---|
| 修改位置 | `app.py` 的 token 处理循环 |
| 刷新策略 | 按字符数或时间间隔批量刷新，例如累计 5 到 10 个 token，或距离上次刷新超过 80ms |
| 边界处理 | 流结束时必须 flush 剩余 buffer |
| 测试策略 | 抽出纯函数或 service 聚合逻辑，避免直接测 Streamlit |

**已决策边界**

| 主题 | 决策 |
|---|---|
| 优化目标 | 平衡减少刷新次数和保持实时输出体感 |
| 刷新策略 | 使用双阈值：累计 8 个流式 chunk，或距离上次刷新超过 80ms 即 flush |
| 计数语义 | 按 `ask_stream()` 事件 chunk 数计数，不做真实 tokenizer 计数；参数命名使用 `max_chunks=8` |
| 模块位置 | 抽到 `paper_rag/ui/streaming.py`，避免把节流逻辑写死在 `app.py` |
| 数据结构 | 使用小类 `TokenStreamBuffer` 管理 buffer、chunk 数和上次 flush 时间 |
| 时间注入 | `append(chunk, now=None)`，默认使用 `time.monotonic()`，测试时可传入 `now` |
| 尾部处理 | 流结束后显式调用 `flush()`，确保不丢最后一段 |
| 游标处理 | 每次 flush 渲染 `answer + "▌"`，结束后再渲染最终 `answer` 去掉游标 |
| 异常处理 | 流式过程中异常时，先 flush 已收到的剩余内容，再展示友好错误提示 |
| 配置化 | 第一版不写入 `config.yaml`，避免增加配置负担 |

**验收方式**

| 场景 | 预期 |
|---|---|
| 流式回答很长 | 页面刷新次数显著减少 |
| 回答结束 | 不丢失最后几个 token |
| LLM 断开 | 错误消息仍能完整显示 |

---

### 3.4 FAQ 与快速开始补齐

**现状**

README 已有环境准备和常用命令，但对“第一次运行失败时怎么排查”覆盖不足。

**建议方案**

| 文档 | 内容 |
|---|---|
| `docs/getting-started.md` | 从安装依赖、启动 Ollama、下载模型、构建向量库、启动 Web UI 到第一个问题 |
| `docs/FAQ.md` | Ollama 连接失败、模型缺失、向量库为空、reranker 不可用、回答慢、检索不准、如何添加论文 |
| README | 增加指向 getting-started 和 FAQ 的链接 |

**已决策边界**

| 主题 | 决策 |
|---|---|
| 文档拆分 | 拆成 `docs/getting-started.md` 和 `docs/FAQ.md`，快速开始负责线性路径，FAQ 负责按问题排障 |
| 使用路径 | Web UI 优先，CLI 作为补充 |
| 快速开始第一步 | 先运行 `python main.py doctor`，按报告补齐环境，再进入 build 和 Web UI |
| 第一个演示问题 | 使用 `BERT 和 GPT 的主要区别是什么？` |
| README 接入 | README 顶部增加新手分流链接：第一次使用看 getting-started，出错看 FAQ，理解架构看 architecture |

**FAQ 第一版问题**

| 编号 | 问题 |
|---|---|
| 1 | 为什么 `python main.py doctor` 报 Ollama 未连接？ |
| 2 | 为什么提示 LLM 模型未下载？ |
| 3 | 为什么 embedding 模型不可用？ |
| 4 | 为什么向量库目录不存在或 collection 为空？ |
| 5 | 为什么 Reranker 不可用但系统还能回答？ |
| 6 | 为什么回答很慢？ |
| 7 | 为什么检索结果不准确？ |
| 8 | 如何添加自己的论文并重建知识库？ |

**验收方式**

| 场景 | 预期 |
|---|---|
| 新用户首次运行 | 能在 5 到 10 分钟内完成 build 和 Web 问答 |
| 出现常见错误 | 能从 FAQ 找到直接排查命令 |

---

## 四、P1：高收益工程优化

### 4.1 BM25 索引持久化

**现状**

`HybridRetriever.build_bm25_retriever()` 已有进程内 lazy cache，但每次重新启动进程仍需要从 Chroma 拉取所有文档并构建 BM25。

**建议方案**

| 项目 | 建议 |
|---|---|
| 缓存路径 | `<persist_directory>/bm25_cache/` |
| 缓存内容 | BM25 retriever pickle 文件，加一份 JSON metadata |
| 失效条件 | collection_name、chunk_schema_version、chunk_count、manifest 更新时间、embedding 无关但 index 内容相关 |
| 写入方式 | 先写临时文件，再原子替换，避免中断后留下半成品 |
| 安全边界 | 只加载项目自己生成的缓存，不从用户上传目录加载 pickle |

**已决策边界**

| 主题 | 决策 |
|---|---|
| 优化目标 | 优化 Streamlit 或进程重启后的检索器冷启动等待时间，不承诺每次 query 加速 |
| 失效依据 | 优先使用 `index_manifest.json` 中的 chunk/schema/source 摘要，加上 `collection_name`；manifest 缺失时降级为 Chroma count 校验并给 WARN |
| cache 位置 | 放在 `<persist_directory>/bm25_cache/`，随向量库自然切换 |
| cache 内容 | 第一版 pickle 项目自己生成的 `BM25Retriever`，同时保存 metadata JSON |
| 安全边界 | 只从当前 `persist_directory/bm25_cache/` 加载项目生成的 pickle，不接受用户任意指定 cache 路径 |
| metadata 字段 | `collection_name`、`persist_directory`、`chunk_schema_version`、`chunk_count`、`top_k`、`source_fingerprint`、`created_at`、`cache_format_version`、`langchain_version` 或 `bm25_class` |
| source fingerprint | 从 manifest 的来源文件、页数、chunk 数等摘要排序后 hash；不 hash 全量 Chroma 文本 |
| 构建时机 | `main.py build` 入库结束后预构建；查询时 lazy build 作为兜底 |
| cache 不匹配 | 输出 WARN 后重建，不阻断启动 |
| 写入方式 | pickle 和 metadata 都先写临时文件，再原子替换正式文件 |
| 加载失败 | 输出 WARN 后忽略坏 cache 并重建；第一版可不删除旧文件，直接原子覆盖 |

**推荐 metadata**

```json
{
  "collection_name": "langchain_vision_v1",
  "chunk_schema_version": "v4",
  "chunk_count": 1234,
  "source_manifest_mtime": 1710000000
}
```

**验收方式**

| 场景 | 预期 |
|---|---|
| 首次构建 | 生成 BM25 cache |
| 第二次启动 | 命中 cache，不重新扫描全部文档 |
| 重新入库后 | cache 自动失效并重建 |
| 切换 collection | 不误用旧 cache |

---

### 4.2 Benchmark 回归检测

**现状**

项目已有 `benchmarks/run_baseline.py`、`eval/metrics.py`、`benchmark_v1.jsonl` 和 `holdout_v1.jsonl`，但缺少“新报告 vs 基线报告”的自动比较。

**建议方案**

| 项目 | 建议 |
|---|---|
| 新增脚本 | `benchmarks/regression_check.py` |
| 输入 | baseline `baseline_results*.jsonl`、新 `baseline_results*.jsonl` |
| 指标 | `recall@5`、`mrr`、`source_hit_status`、`answer_completeness`、`evidence_coverage` |
| 容忍度 | 按指标设置不同 tolerance，例如检索指标 0.03，生成指标 0.05 |
| 输出 | 明确列出回退项、旧值、新值、差值 |

**已决策边界**

| 主题 | 决策 |
|---|---|
| 第一版目标 | 检索指标作为硬门禁，生成质量和耗时指标先作为 WARN |
| 输入文件 | 比较两个 `baseline_results*.jsonl`，脚本内部复用 `eval.metrics` 计算指标 |
| 脚本路径 | `benchmarks/regression_check.py` |
| 硬门禁指标 | `recall@5`、`mrr`、`missing_rate` |
| WARN 指标 | `answer_completeness`、`evidence_coverage`、`avg_elapsed_sec` |
| tolerance | `recall@5` 允许下降 0.03，`mrr` 允许下降 0.03，`missing_rate` 允许上升 0.05，生成指标下降超过 0.05 WARN，平均耗时上升超过 20% WARN |
| 样本集合 | baseline 和 new 必须包含完全相同的样本 id；不一致时 ERROR，不取交集比较 |
| 输出格式 | 默认人类可读文本，支持 `--json` |
| exit code | `0` 表示无硬门禁回退，`1` 表示硬门禁回退，`2` 表示输入文件缺失、格式错误或样本 id 不一致 |

**验收方式**

| 场景 | 预期 |
|---|---|
| 指标小幅波动 | 在 tolerance 内通过 |
| Recall 或 MRR 明显下降 | 脚本 exit code 非 0 |
| 新报告字段缺失 | 明确报错，不静默跳过 |

---

### 4.3 来源展示高亮与评分信息

**现状**

Streamlit 来源展示当前主要显示文件名、页码和 `st.text(d.page_content[:400])`。用户能看到来源，但不容易判断“为什么这段被选中”。

**实施状态**

| 状态 | 实际改动 | 验证命令 |
|---|---|---|
| 已完成 | 新增 `paper_rag/ui/sources.py` 生成来源 view model；Streamlit 来源展示改为文件名、页码、rerank 分数、命中高亮片段和原始片段折叠详情 | `.\.venv\Scripts\python.exe -m unittest discover -s tests` |

**建议方案**

| 项目 | 建议 |
|---|---|
| 展示字段 | 文件名、页码、rerank_score、是否 compressed、片段预览 |
| 高亮方式 | 用 query terms 高亮片段中的匹配词 |
| 渲染方式 | 从 `st.text` 改为经过转义后的 `st.markdown` |
| 安全处理 | 先 HTML/Markdown 转义文档文本，再插入高亮 |

**已决策边界**

| 主题 | 决策 |
|---|---|
| 优化目标 | 让用户快速判断来源为什么相关，而不是单纯美化页面 |
| 展示字段 | 文件名、页码、可选 `rerank_score`、命中高亮片段；原始片段全文放折叠详情 |
| 高亮关键词来源 | 合并用户原始 query 和多轮改写后的 standalone question；第一版不纳入 query expansion 变体 |
| 高亮粒度 | 英文按词和术语，中文按 2 字以上片段，过滤停用词 |
| 片段策略 | 展示最高命中句子，最多 2 到 3 句；不再固定展示 chunk 前 400 字符 |
| 渲染方式 | 使用 HTML `<mark>`；必须先 `html.escape()` 原文，再插入高亮标签 |
| 模块位置 | 放在 `paper_rag/ui/sources.py`，返回可渲染 view model |
| view model 字段 | `title`、`source`、`page`、`score_label`、`highlight_html`、`raw_preview`、可选精简 `metadata` |

**验收方式**

| 场景 | 预期 |
|---|---|
| 用户问英文术语 | 来源片段中相关英文词被高亮 |
| 用户问中文关键词 | 中文关键词可高亮 |
| 文档含 Markdown 特殊字符 | 不破坏页面结构 |

---

## 五、P2：质量优化，必须用评估约束

### 5.1 Rerank 阈值过滤

**现状**

`reranker.py` 已将 `rerank_score` 写入 metadata，但目前主要按 `top_k` 截断，不按分数阈值过滤。

**实施状态**

| 状态 | 实际改动 | 验证命令 |
|---|---|---|
| 已完成 | 新增 `rerank_score_threshold`、`rerank_min_docs` 配置；默认关闭阈值；仅普通 mixed 路由按 `rerank_score` 过滤，并保留最小文档数；evidence/source anchor 路径不应用阈值 | `.\.venv\Scripts\python.exe -m unittest tests.test_config_settings tests.test_retrieval_router` |

**风险**

| 风险 | 说明 |
|---|---|
| 分数不可直接跨 query 比较 | Cross-Encoder 分数在不同问题上的绝对尺度可能不同 |
| 过滤过狠 | 可能导致 evidence 问题无来源，反而让答案质量下降 |
| 对特殊路由影响不同 | evidence、comparison、source-specific 路由都有 anchor docs，不能统一粗暴过滤 |

**建议方案**

| 项目 | 建议 |
|---|---|
| 配置项 | `rerank_score_threshold`、`rerank_min_docs` |
| 默认策略 | 默认关闭阈值，仅通过实验开启 |
| 保护机制 | 至少保留 `rerank_min_docs` 个文档 |
| 适用范围 | 先只作用于普通 mixed 路由，不作用于 evidence anchor docs |

**验收方式**

| 指标 | 要求 |
|---|---|
| Recall@5 | 不低于 baseline tolerance |
| MRR | 不低于 baseline tolerance |
| Answer Completeness | 应有提升或持平 |
| 失败样本 | 手动检查被过滤文档是否确实低相关 |

---

### 5.2 Query Expansion 质量过滤

**现状**

`query_expansion.py` 已有改写 prompt 和解析逻辑，能避免标签行、重复行等低级问题。当前缺口不是“没有 Query Expansion”，而是缺少变体质量控制。

**实施状态**

| 状态 | 实际改动 | 验证命令 |
|---|---|---|
| 已完成 | 新增 Query Expansion 相似度过滤器；默认关闭；启用后通过现有 embedding 过滤过近或过远变体，并在 query log 中记录保留变体和过滤原因 | `.\.venv\Scripts\python.exe -m unittest tests.test_query_expansion tests.test_query_logger tests.test_config_settings` |

**建议方案**

| 项目 | 建议 |
|---|---|
| Prompt | 加入 2 到 3 个 few-shot 示例，强调“互补但不发散” |
| 过滤 | 使用 embedding 相似度过滤过近或过远的变体 |
| 成本控制 | 只在 `enable_query_expansion` 开启时额外计算相似度 |
| 日志 | query log 记录原始 query、变体、过滤原因 |

**阈值建议**

| 类型 | 建议处理 |
|---|---|
| 与原问题几乎相同 | 过滤，避免重复召回 |
| 与原问题语义太远 | 过滤，避免噪声召回 |
| 包含关键术语翻译或缩写全称 | 保留 |

**验收方式**

| 指标 | 要求 |
|---|---|
| Recall@5 | 应提升或持平 |
| MRR | 不应明显下降 |
| 检索耗时 | 记录额外 embedding 成本 |
| 失败样本 | 检查变体是否引入无关论文 |

---

### 5.3 Context Compression 打分改进

**现状**

`context_compression.py` 当前按 query terms 做简单词频命中，优点是可控、速度快，缺点是不能识别句子位置、术语覆盖密度、标题附近上下文等信号。

**实施状态**

| 状态 | 实际改动 | 验证命令 |
|---|---|---|
| 已完成 | 将句子评分升级为 term 命中、覆盖率、长度因子和位置微加分的组合；对 vision summary、abstract、table/equation/formula 等结构化证据保守不压缩 | `.\.venv\Scripts\python.exe -m unittest tests.test_context_compression` |

**建议方案**

| 项目 | 建议 |
|---|---|
| 位置权重 | 越靠前句子略加分，但不能用固定常量，应随句子 index 递减 |
| 覆盖密度 | 匹配不同 query terms 越多，分数越高 |
| 长度惩罚 | 过短句子和过长句子适度降权 |
| 保护策略 | 对 vision summary、abstract、含公式/表格说明的 chunk 保持保守 |

**示意公式**

```text
score = term_score * coverage_bonus * length_factor + position_bonus
```

**验收方式**

| 场景 | 预期 |
|---|---|
| query terms 明确 | 压缩后保留核心证据句 |
| query terms 很少 | 不强行压缩，保留原 chunk |
| parent retrieval 后上下文较长 | 压缩比例提升但答案完整度不下降 |

---

## 六、P3：长期收口与维护性优化

### 6.1 Embedding 原型预计算

**现状**

`SemanticWeightDecider` 每次初始化会对 precise 和 semantic anchors 做 embedding，用于动态权重微调。这个成本只发生在检索器初始化阶段，不是每次查询成本。

**实施状态**

| 状态 | 实际改动 | 验证命令 |
|---|---|---|
| 已完成 | 新增 `data/prototypes/<hash>.npz` 原型缓存；按 embedding model、anchor version、cache format 生成缓存 key；缓存损坏或不匹配时自动重算并覆盖 | `.\.venv\Scripts\python.exe -m unittest tests.test_semantic_prototype_cache` |

**建议方案**

| 项目 | 建议 |
|---|---|
| 缓存路径 | `data/prototypes/<embedding_model_hash>.npz` |
| 版本字段 | embedding model、anchor version、embedding dimension |
| 失效条件 | anchor 文本变化、embedding 模型变化、向量维度变化 |
| 降级策略 | 缓存不存在时现场计算并写入 |

**优先级说明**

该项可行，但收益小于 BM25 持久化。建议在启动诊断和 BM25 cache 完成后再做。

---

### 6.2 包化迁移最终收口

**现状**

当前根目录存在两类文件：

| 类型 | 示例 | 建议 |
|---|---|---|
| 真实入口 | `app.py`、`main.py`、`query.py`、`build_knowledge.py`、`rag_pipeline.py` | 暂时保留 |
| 兼容薄壳 | `generation_service.py`、`context_builder.py`、`retrieval_router.py`、`source_utils.py`、`query_expansion.py`、`reranker.py`、`context_compression.py` | 分阶段移除 |
| 仍有真实实现 | 暂无明确必须迁移的检索/生成 helper | 下一步聚焦 `rag_pipeline.py` 继续瘦身和兼容壳退场策略 |

**已完成进展**

| 状态 | 内容 | 验证 |
|---|---|---|
| 已完成 | 新增 `paper_rag.generation.context_compression`、`paper_rag.retrieval.query_expansion`、`paper_rag.retrieval.reranker`，根目录同名文件降级为兼容薄壳 | `.\.venv\Scripts\python.exe -m unittest tests.test_package_module_migration` |
| 已完成 | `paper_rag.generation.context`、`paper_rag.retrieval.router`、`rag_pipeline.py` 的内部导入切换到 `paper_rag.*` 路径 | `.\.venv\Scripts\python.exe -m unittest tests.test_query_expansion tests.test_context_compression tests.test_reranker tests.test_retrieval_router tests.test_context_builder` |
| 已完成 | 新增 `paper_rag.retrieval.hybrid`，根目录 `hybrid_retriever.py` 降级为兼容薄壳；`rag_pipeline.py` 与 `build_knowledge.py` 改用包内路径 | `.\.venv\Scripts\python.exe -m unittest tests.test_package_module_migration tests.test_hybrid_retriever_bm25_cache tests.test_semantic_prototype_cache tests.test_build_experiment` |

**建议阶段**

| 阶段 | 动作 | 验收 |
|---|---|---|
| 1 | 全局替换内部 import 为 `paper_rag.*` | 已覆盖 generation context、retrieval router 和 `rag_pipeline.py` 的核心辅助导入 |
| 2 | 对薄壳增加 deprecation 注释或测试保护 | 旧入口仍可用；保留 `reranker.get_reranker` patch 兼容语义 |
| 3 | 将剩余真实实现迁入 `paper_rag.retrieval`、`paper_rag.generation` | 已迁入 `hybrid_retriever`、`query_expansion`、`reranker`、`context_compression` |
| 4 | 在确认无外部调用后删除薄壳 | README 和 docs 同步更新 |

**风险**

| 风险 | 规避方式 |
|---|---|
| Streamlit、CLI、benchmark import 被破坏 | 先用 `rg` 建立 import 清单，再分批替换 |
| 测试 patch 路径变化 | 保留兼容 wrapper，逐步迁移测试 |
| 循环导入 | 包内 `__init__` 保持轻量，必要时使用懒加载 |

---

### 6.3 `rag_pipeline.py` 继续瘦身

**现状**

`rag_pipeline.py` 已经不再承载所有实现细节，但仍混合了：

| 职责 | 当前内容 |
|---|---|
| 运行时环境 | HuggingFace offline 环境变量、device 选择 |
| 生命周期 | LLM cache、embedding、Chroma、HybridRetriever 构建 |
| 检索编排 | retrieve、multi-query、HyDE、route wrapper |
| 生成编排 | ask、ask_stream、日志写入 |
| 演示入口 | `main()` 测试问答 |

**已完成进展**

| 状态 | 内容 | 验证 |
|---|---|---|
| 已完成 | 新增 `paper_rag.runtime.models`，将 embedding device 选择和 LLM 连接缓存从 `rag_pipeline.py` 抽出；`rag_pipeline._get_embedding_device` 与 `_get_llm` 保留为兼容入口 | `.\.venv\Scripts\python.exe -m unittest tests.test_runtime_models tests.test_query_logger` |
| 已完成 | 将 embedding、Chroma、HybridRetriever 的构建逻辑迁入 `paper_rag.runtime.models.build_hybrid_retriever`；`rag_pipeline.build_hybrid_retriever` 保留为兼容入口 | `.\.venv\Scripts\python.exe -m unittest tests.test_runtime_models tests.test_query_logger tests.test_query_expansion tests.test_context_compression` |
| 已完成 | 新增 `paper_rag.pipeline.retrieval`，迁出基础检索和 HyDE 检索编排；`rag_pipeline._retrieve` 与 `_retrieve_with_hyde` 保留为兼容入口 | `.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_retrieval tests.test_query_expansion tests.test_rag_rerank_integration tests.test_query_logger` |
| 已完成 | 将 multi-query 扩展、变体过滤、合并去重和结果截断迁入 `paper_rag.pipeline.retrieval.retrieve_multi_query`；`rag_pipeline._retrieve_multi_query` 保留 trace 兼容入口 | `.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_retrieval tests.test_query_expansion tests.test_rag_rerank_integration tests.test_query_logger` |
| 已完成 | 将 `RetrievalRouter` 依赖注入和路由调用迁入 `paper_rag.pipeline.retrieval.route_retrieve`；`rag_pipeline._route_retrieve` 保留为兼容入口 | `.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_retrieval tests.test_query_expansion tests.test_rag_rerank_integration tests.test_query_logger` |
| 已完成 | 将检索片段格式化迁入 `paper_rag.generation.service.format_docs`；`generation_service` 与 `rag_pipeline._format_docs` 保留兼容导出 | `.\.venv\Scripts\python.exe -m unittest tests.test_generation_service tests.test_context_compression tests.test_query_logger` |
| 已完成 | 将非流式答案生成编排迁入 `paper_rag.generation.service.generate_answer_from_docs`；`rag_pipeline._generate_answer` 保留为注入当前配置的兼容入口 | `.\.venv\Scripts\python.exe -m unittest tests.test_generation_service tests.test_context_compression tests.test_parent_retrieval tests.test_query_logger tests.test_package_imports` |
| 已完成 | 将流式 token 生成编排迁入 `paper_rag.generation.service.stream_answer_from_docs`；`rag_pipeline.ask_stream` 继续负责事件顺序、日志和 LLM 不可用错误记录 | `.\.venv\Scripts\python.exe -m unittest tests.test_generation_service tests.test_query_logger tests.test_package_imports` |
| 已完成 | 新增 `paper_rag.pipeline.service.reformulate_question`，统一非流式和流式入口的多轮追问改写决策；`rag_pipeline` 继续负责打印、事件和日志 | `.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_service tests.test_query_logger tests.test_app_services` |
| 已完成 | 新增 `paper_rag.pipeline.service.write_pipeline_query_log`，将 mixed route 的 query expansion trace 选择逻辑从 `rag_pipeline._write_query_log` 迁入包内 | `.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_service tests.test_query_logger` |
| 已完成 | 新增 `paper_rag.pipeline.service.handle_no_docs_response`，统一多轮与流式入口无检索结果时的空 docs 日志和返回消息 | `.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_service tests.test_query_logger tests.test_app_services` |
| 已完成 | 新增 `paper_rag.pipeline.service.handle_llm_unavailable_response`，统一流式生成中 LLM 不可用时的错误日志和 token 事件 | `.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_service tests.test_query_logger tests.test_app_services` |
| 已完成 | 新增 `paper_rag.pipeline.service.prepare_pipeline_context`，统一生成前上下文准备与 context stats 计算，供非流式和流式入口复用 | `.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_service tests.test_query_logger tests.test_context_builder tests.test_context_compression tests.test_parent_retrieval` |
| 已完成 | 新增 `paper_rag.pipeline.service.stream_token_events`，将正常流式生成 token 的事件包装从 `rag_pipeline.ask_stream` 迁入包内 | `.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_service tests.test_query_logger tests.test_app_services tests.test_ui_streaming` |
| 已完成 | 新增 `paper_rag.pipeline.service.stream_retrieval_events`，将流式入口的 route/docs 前置事件包装从 `rag_pipeline.ask_stream` 迁入包内 | `.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_service tests.test_query_logger tests.test_app_services tests.test_ui_streaming` |
| 已完成 | 新增 `paper_rag.pipeline.service.write_successful_response_log`，将非流式与流式成功生成后的 query log 字段拼装迁入包内 | `.\.venv\Scripts\python.exe -m unittest tests.test_pipeline_service tests.test_query_logger tests.test_app_services tests.test_ui_streaming` |

**建议拆分**

| 目标模块 | 职责 |
|---|---|
| `paper_rag.runtime.models` | LLM cache、embedding device、模型构建 |
| `paper_rag.pipeline.service` | `ask_with_context`、`ask_stream` |
| `paper_rag.pipeline.retrieval` | `_retrieve`、multi-query、HyDE wrapper |
| `rag_pipeline.py` | 仅保留兼容导出 |

**验收方式**

| 项目 | 要求 |
|---|---|
| 公开函数 | `build_hybrid_retriever`、`ask_with_context`、`ask_stream` 调用形状不变 |
| 行为 | 现有测试通过 |
| 日志 | query log 字段不变 |
| Benchmark | `run_baseline.py --no-generate` 通过 |

---

## 七、建议执行路线

| 周期 | 任务 | 验收命令 |
|---|---|---|
| 第 1 批 | 启动诊断、错误提示、Streamlit 刷新节流、FAQ | `python -m unittest tests.test_config_settings tests.test_app_services` |
| 第 2 批 | BM25 持久化、来源展示优化 | `python -m unittest tests.test_rag_rerank_integration tests.test_source_utils` |
| 第 3 批 | Benchmark 回归检测 | `python benchmarks/run_baseline.py --no-generate`，再运行新增 regression check 单测 |
| 第 4 批 | Rerank 阈值、Query Expansion 过滤、Compression 改进 | 完整 benchmark，对比 baseline |
| 第 5 批 | 原型预计算、包化迁移收口、`rag_pipeline.py` 瘦身 | `python -m unittest discover -s tests` |

---

## 八、不建议优先做的事项

| 事项 | 原因 |
|---|---|
| 直接删除所有根目录薄壳 | 当前入口、测试和 benchmark 仍可能依赖旧路径，容易造成非必要破坏 |
| 一次性大拆 `rag_pipeline.py` | 行为面广，收益不如 P0/P1 明确 |
| 无评估地开启 rerank 阈值 | 可能提升个别问题表现，但降低整体召回 |
| 对所有 query 做结果缓存 | 容易被 index version、route、rerank、query expansion、source filter 等变量污染 |
| 为小项目引入大型服务化组件 | 当前定位是本地演示和学习项目，优先保持轻量 |

---

## 九、推荐先实施的最小闭环

如果只选择一个最小闭环，建议按下面顺序：

| 顺序 | 任务 | 为什么 |
|---|---|---|
| 1 | 新增 `doctor` 启动诊断 | 先让环境问题可解释 |
| 2 | Streamlit 初始化和问答异常使用友好错误文案 | 让用户看到可执行解决方案 |
| 3 | 流式刷新节流 | 立刻改善 Web 体验 |
| 4 | 写 FAQ | 把诊断和错误文案沉淀成文档 |
| 5 | 加 benchmark regression check | 后续质量优化才有保护栏 |

这个闭环的特点是风险低、收益清晰，并为后续 BM25 持久化、rerank 阈值、query expansion 过滤等质量相关改动提供验证基础。

---

## 十、维护说明

每完成一项优化，建议在本文档中补充：

| 字段 | 说明 |
|---|---|
| 状态 | 未开始、进行中、已完成、暂缓 |
| 实际收益 | 启动耗时、刷新次数、benchmark 指标、用户排障耗时等 |
| 验证命令 | 本次改动实际跑过的测试或 benchmark |
| 后续影响 | 是否改变配置、入口命令、文档或测试策略 |

本文档应作为项目 backlog 使用。涉及检索质量的优化必须配合 benchmark 或失败样本验证，避免只凭主观感受判断效果。
