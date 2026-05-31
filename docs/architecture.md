# 架构说明

本文记录当前项目结构、第一轮工程化边界，以及后续渐进包化目标。当前重构原则是先稳定公共接口，再移动核心模块。

## 当前运行链路

| 层级 | 当前模块 | 职责 |
|---|---|---|
| 入口层 | `app.py`、`main.py`、`query.py` | Web、CLI 和命令分发入口。 |
| 入库层 | `build_knowledge.py` | 读取 PDF、切分文本、写入 Chroma。 |
| 编排层 | `rag_pipeline.py` | 构建检索器、选择检索路线、准备上下文、生成答案、写查询日志。 |
| 检索层 | `hybrid_retriever.py`、`query_expansion.py`、`reranker.py`、`hyde_retriever.py` | 混合检索、多 query、精排和 HyDE 相关能力。 |
| 上下文层 | `parent_retrieval.py`、`context_compression.py` | 生成 prompt 前的 parent 回溯与上下文压缩。 |
| 对话层 | `conversation.py` | 多轮历史、追问改写和模型配置。 |
| 观测层 | `query_logger.py`、`feedback.py`、`source_utils.py` | 查询日志、失败样本回流和来源字段归一化。 |
| 评估层 | `benchmarks/`、`eval/` | 基准运行、结果输出和离线指标计算。 |

## 第一轮重构边界

第一轮只做低风险工程化，不改变问答行为：

| 改动 | 说明 |
|---|---|
| 新增 `README.md` | 集中说明运行、测试、日志和评估方式。 |
| 新增 `docs/architecture.md` | 固化当前边界和后续拆分方向。 |
| 新增 `source_utils.py` | 统一日志、反馈和 baseline 的来源序列化字段。 |
| 复用来源序列化 | `query_logger.py`、`feedback.py`、`benchmarks/run_baseline.py` 共享同一来源结构。 |

## 第二轮重构边界

第二轮继续保持公开入口不变，只拆出生成前上下文构建：

| 改动 | 说明 |
|---|---|
| 新增 `context_builder.py` | 承载 Parent Retrieval、Context Compression 和上下文统计。 |
| 简化 `rag_pipeline.py` | 移除上下文增强的实现细节，只在生成前调用 `prepare_docs_for_context`。 |
| 新增 `tests/test_context_builder.py` | 独立验证开关关闭、parent 先于 compression、统计字段稳定。 |
| 保持来源语义 | 检索返回给 UI 和日志来源的原始 `docs` 不变，压缩只作用于 prompt context。 |

## 第三轮重构边界

第三轮拆出检索路由，继续保留 `rag_pipeline._route_retrieve` 作为兼容入口：

| 改动 | 说明 |
|---|---|
| 新增 `retrieval_router.py` | 承载 mixed、multi-query、evidence、compare anchors 和 HyDE 路由选择。 |
| 简化 `rag_pipeline.py` | `_route_retrieve` 改为委托 `RetrievalRouter`，保留原有公开调用形状。 |
| 新增 `tests/test_retrieval_router.py` | 独立验证分类器、mixed rerank、HyDE 路由、evidence source 过滤。 |
| 保持 patch 兼容 | 旧测试对 `rag_pipeline._retrieve_multi_query`、`_retrieve_with_hyde`、`apply_rerank` 的 patch 仍有效。 |

## 第四轮重构边界

第四轮拆出答案生成服务，继续保留 `rag_pipeline._generate_answer` 作为兼容入口：

| 改动 | 说明 |
|---|---|
| 新增 `generation_service.py` | 承载 RAG prompt 构造、非流式 LLM 调用、流式 token 输出和 LLM 不可用文案。 |
| 简化 `rag_pipeline.py` | `_generate_answer` 委托 `generate_answer`，`ask_stream` 委托 `stream_answer_tokens`。 |
| 新增 `tests/test_generation_service.py` | 独立验证 prompt 拼接、响应 content 解析、字符串 chunk 和 LLM 不可用返回。 |
| 保持日志语义 | `ask_stream` 仍在 LLM 不可用时记录 `error`，成功流式生成后记录 context 和耗时。 |

## 第五轮重构边界

第五轮拆出 Web UI service，继续保持 Streamlit 页面行为不变：

| 改动 | 说明 |
|---|---|
| 新增 `app_services.py` | 承载上传 PDF 落盘、流式事件收敛、反馈 payload 构造和反馈保存。 |
| 简化 `app.py` | 页面侧继续负责渲染和逐 token 展示，非渲染逻辑委托 service。 |
| 新增 `tests/test_app_services.py` | 独立验证上传保存、流式事件聚合、反馈 payload 和空备注校验。 |
| 保持状态语义 | 清空对话仍由 `app_state.clear_conversation_state` 处理，保留模型选择和数据库版本。 |

## 来源结构

所有观测和评估输出中的来源字段应优先复用 `source_utils.sources_from_docs`。

| 字段 | 说明 |
|---|---|
| `file` | 来源文件名，去掉目录路径。 |
| `page` | 页码，缺失时为 `-1`。 |
| `content_preview` | 来源片段预览。 |
| `rerank_score` | 可选字段，仅当 metadata 中存在时输出。 |

## 目标包结构

后续迁移目标如下。迁移过程中应保留旧入口，避免破坏 Streamlit、CLI 和 benchmark。

```text
paper_rag/
  config/
  ingestion/
  retrieval/
  generation/
  conversation/
  observability/
  ui/
  cli/
  evaluation/
```

## 第六轮重构边界

第六轮进入包化迁移前准备，只建立兼容 facade，不移动旧文件：

| 改动 | 说明 |
|---|---|
| 新增 `paper_rag/` 包 | 提供 `generation`、`retrieval`、`observability`、`ui` 子包。 |
| 采用 re-export | 子包从当前根目录稳定模块导出能力，避免一次性改全项目 import。 |
| 新增 `tests/test_package_imports.py` | 验证目标包路径可导入当前服务。 |
| 新增 `pyproject.toml` | 记录项目基础元数据和常用命令说明，不改变运行方式。 |

当前 facade 只是迁移过渡层。后续可以按模块逐步把实现文件移动到 `paper_rag/` 内，再让根目录旧文件变成薄壳。

## 第七轮重构边界

第七轮迁移第一批低耦合实现文件，根目录继续保留兼容薄壳：

| 包内实现 | 根目录薄壳 | 说明 |
|---|---|---|
| `paper_rag.observability.sources` | `source_utils.py` | 来源序列化实现迁入 observability 子包。 |
| `paper_rag.generation.service` | `generation_service.py` | Prompt 构造和 LLM 生成服务迁入 generation 子包。 |
| `paper_rag.ui.services` | `app_services.py` | 上传、流式聚合和反馈 service 迁入 ui 子包。 |

为避免 `feedback -> source_utils -> paper_rag.observability` 的循环导入，`paper_rag.observability.__init__` 对反馈和查询日志函数使用懒加载。

## 第八轮重构边界

第八轮迁移中等耦合模块，继续保留根目录兼容薄壳：

| 包内实现 | 根目录薄壳 | 说明 |
|---|---|---|
| `paper_rag.generation.context` | `context_builder.py` | Parent Retrieval、Context Compression 编排和 context stats。 |
| `paper_rag.retrieval.router` | `retrieval_router.py` | mixed / multi-query / evidence / compare anchors / HyDE 路由选择。 |

`tests/test_package_module_migration.py` 已覆盖这些包内目标路径，旧测试仍覆盖根目录薄壳路径，确保迁移期间新旧 import 都稳定。

## 第九轮重构边界

第九轮继续迁移剩余辅助模块，进一步减少根目录承载的真实实现：

| 包内实现 | 根目录薄壳 | 说明 |
|---|---|---|
| `paper_rag.observability.feedback` | `feedback.py` | 失败样本回流记录构造与写入。 |
| `paper_rag.observability.query_logger` | `query_logger.py` | 查询运行日志记录构造与写入。 |
| `paper_rag.generation.parent_retrieval` | `parent_retrieval.py` | 生成阶段按页回溯 parent context。 |

`paper_rag.observability` 的懒加载现在指向包内模块，避免 facade 反向依赖根目录兼容壳。

## 第十轮重构边界

第十轮开始收口配置访问，不改变 YAML 文件格式，也不移除旧 `utils.config_loader.load_config`：

| 改动 | 说明 |
|---|---|
| 新增 `paper_rag.config.RagSettings` | 将核心配置项转换为 typed settings，并为可选项提供稳定默认值。 |
| 新增 `ConfigError` | 缺少 `persist_directory`、`embedding_model` 等必要字段时，在启动阶段给出明确错误。 |
| 替换模块级常量读取 | `rag_pipeline.py`、`build_knowledge.py` 的核心常量改由 `RagSettings` 读取。 |
| 保留原始 `config` 字典 | 检索路由、上下文增强等旧函数仍接收 dict，避免一次性牵动整条链路。 |

## 第十一轮重构边界

第十一轮让已包化模块支持 typed settings，同时继续兼容旧 dict：

| 改动 | 说明 |
|---|---|
| 新增 `paper_rag.config.get_setting` | 统一读取 dict 与 `RagSettings`，避免包内模块直接假设 `.get`。 |
| 更新 `paper_rag.generation.context` | `prepare_docs_for_context` 可直接接收 `RagSettings`。 |
| 更新 `paper_rag.retrieval.router` | `RetrievalRouter` 可直接接收 `RagSettings`。 |
| 更新 `rag_pipeline.py` | 调用已包化上下文和路由模块时传递 typed `settings`。 |

## 第十二轮重构边界

第十二轮抽出查询日志写入策略，减少 `rag_pipeline.py` 的观测细节：

| 改动 | 说明 |
|---|---|
| 新增 `paper_rag.observability.service.write_query_log` | 负责读取日志开关、构造 query log record 并写入 JSONL。 |
| 保留 `_write_query_log` | `rag_pipeline.py` 中旧函数变成兼容包装，调用方不需要变化。 |
| 复用 typed settings | 日志开关和路径通过 `RagSettings` / `get_setting` 读取，继续兼容旧 dict。 |

## 第十三轮重构边界

第十三轮抽出链路阶段耗时工具，让主编排更聚焦业务步骤：

| 改动 | 说明 |
|---|---|
| 新增 `paper_rag.observability.trace.TraceTimer` | 集中生成 rewrite、retrieve、generate、total 耗时。 |
| 更新 `rag_pipeline.py` | `ask_with_context` 与 `ask_stream` 不再直接散落 `time.perf_counter()` 计算。 |
| 保持日志结构 | 输出的 `elapsed` 字段名和语义不变，仍由 query log record 做最终 rounding。 |

## 第十四轮重构边界

第十四轮完成主链路配置读取收口：

| 改动 | 说明 |
|---|---|
| 收口 `rag_pipeline.py` 配置读取 | 检索器构建和 query expansion 改为通过动态 `RagSettings` 读取。 |
| 收口 `build_knowledge.py` 配置读取 | 文本切分 separators 改为通过 `settings.separators` 读取。 |
| 保留 patch 兼容 | `rag_pipeline._get_settings()` 仍从当前 `config` 构建，测试和运行期 patch 语义不变。 |

## 后续拆分顺序

| 阶段 | 目标 | 说明 |
|---|---|---|
| 1 | 主编排 facade | 如后续功能继续变多，再拆 `rag_pipeline.py` 的生命周期和异常边界。 |
| 2 | 评估包化 | 将 `benchmarks/`、`eval/` 迁入 `paper_rag.evaluation`，当前小项目阶段可暂缓。 |

## 验收清单

每个重构阶段至少运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe benchmarks\run_baseline.py --no-generate
```

涉及真实问答链路时，再补充：

```powershell
streamlit run app.py
```

并检查 `logs/query_runs.jsonl` 是否新增记录。
