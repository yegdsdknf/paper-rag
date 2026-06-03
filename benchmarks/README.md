# 人工标注基准集

本目录用于保存 RAG 系统的人工标注基准样本。基准集的目标是先定义「什么叫答得对」，再用离线评估脚本比较不同检索和生成策略。

## 当前文件

| 文件 | 说明 |
|---|---|
| `benchmark_v1.jsonl` | 任务文档约定的基准集入口文件，覆盖 29 条样本，包含 evidence 与 figure 类型题。 |
| `holdout_v1.jsonl` | 独立留出评估集，问题文本不与 `benchmark_v1.jsonl` 重复，用于检查检索规则是否过拟合。 |
| `run_baseline.py` | 基线跑分脚本，读取基准集并输出当前 RAG 系统的回答结果。 |
| `baseline_results.jsonl` | 基线跑分输出文件，由 `run_baseline.py` 生成。 |
| `feedback/feedback.jsonl` | 真实使用反馈暂存文件，用于后续人工筛选并回流到基准集。 |
| `../logs/query_runs.jsonl` | 结构化查询日志，记录每次问答的路由、来源、上下文长度和耗时。 |
| `../eval/run_eval.py` | 离线评估脚本，读取 baseline 结果并生成 Recall@k / MRR 报告。 |

## 字段说明

| 字段 | 含义 |
|---|---|
| `id` | 样本编号。 |
| `question` | 用户问题。 |
| `history` | 可选字段，用于追问类样本的多轮上下文。 |
| `gold_answer` | 人工标注的标准答案。 |
| `gold_sources` | 标准证据来源，包含文件名和页码。 |
| `gold_evidence` | 支撑答案的关键原文片段或关键词。 |
| `task_type` | 任务类型，包括 `definition`、`compare`、`method`、`experiment`、`detail`、`summary`、`followup`、`evidence`、`figure`。 |
| `difficulty` | 难度，当前包含 `easy`、`medium`、`hard`。 |
| `notes` | 标注备注，说明该样本主要测试什么能力。 |

## 使用建议

评估时至少检查两层结果：检索层看 `gold_sources` 是否命中，生成层看回答是否覆盖 `gold_answer` 和 `gold_evidence`。

第一页页码沿用当前 PDF loader 的页码习惯，从 `0` 开始计数。

## 基线跑分

快速检查基准集加载和输出格式：

```powershell
python benchmarks/run_baseline.py --no-generate
```

运行完整 RAG 基线：

```powershell
.\.venv\Scripts\python.exe benchmarks/run_baseline.py
```

运行 Agentic RAG 基线：

```powershell
.\.venv\Scripts\python.exe benchmarks\run_baseline.py --agent --output benchmarks\agentic_results_qwen2.5_3b.jsonl
```

Agentic 路径默认关闭，只有传入 `--agent` 或在系统配置/调用入口显式启用时才会使用。结果文件会额外包含 `agent_trace`，用于检查规划目标、证据验证、repair 轮次和最终来源。

运行 holdout 留出集：

```powershell
.\.venv\Scripts\python.exe benchmarks/run_baseline.py --benchmark benchmarks/holdout_v1.jsonl --output benchmarks/holdout_results_qwen2.5_3b.jsonl
```

holdout 只用于最终验证，不应用来继续调具体规则。若必须根据 holdout 暴露的问题修改系统，应先把该问题移入下一版训练/开发集，再重新创建新的 holdout。

## 离线自动评估

任务 3 的评估脚本只读取已有 baseline 结果，不重新调用模型，避免把模型输出波动混入评估脚本验证。

```powershell
python eval/run_eval.py --input benchmarks/baseline_results_qwen2.5_3b.jsonl --label qwen2_5_3b
python eval/run_eval.py --input benchmarks/baseline_results_deepseek_r1_7b.jsonl --label deepseek_r1_7b
python eval/run_eval.py --input benchmarks/holdout_results_qwen2.5_3b.jsonl --label holdout_qwen2_5_3b
python eval/run_eval.py --input benchmarks/agentic_results_qwen2.5_3b.jsonl --label agentic_qwen2_5_3b
```

报告会写入 `eval/reports/`，包含样本数、平均 Recall@5、平均 MRR、来源命中统计、低召回样本清单，以及按题型和难度聚合的结果。

第二阶段后，报告会额外输出 `layers` 和 `error_bucket_counts`：

| 层级 / 字段 | 说明 |
|---|---|
| `layers.retrieval` | 检索层指标，包括 Recall@K、MRR、来源命中统计。 |
| `layers.context` | 上下文层指标，包括输入/输出字符数、压缩率、parent 命中数量。 |
| `layers.answer` | 答案层指标，包括证据覆盖率、答案完整度、引用准确度和未支撑声明数量。 |
| `layers.experience` | 体验层指标，包括平均耗时、错误数、跳过数和错误桶分布。 |
| `error_bucket_counts` | 将样本归入 `retrieval_miss`、`partial_retrieval`、`answer_incomplete`、`runtime_error`、`skipped` 或 `ok`，用于定位下一轮优化方向。 |

## 抗过拟合评估约定

| 集合 | 用途 | 允许行为 |
|---|---|---|
| `benchmark_v1.jsonl` | 开发/回归集，用于定位和修复已知失败模式。 | 可以基于错误分析修改检索、上下文和生成策略。 |
| `holdout_v1.jsonl` | 留出集，用于验证修复是否泛化到新问法。 | 只读评估；不要直接按单条 holdout 失败样本调规则。 |

对比报告时优先看两组指标的差距：如果开发集 Recall@5 明显高于 holdout，说明当前策略可能仍有过拟合或来源解析覆盖不足。

## 样本回流

Streamlit 问答完成后，可以在“记录失败样本 / 反馈”中填写备注并保存。反馈记录会追加到：

```powershell
benchmarks/feedback/feedback.jsonl
```

该文件只作为暂存区。后续需要人工补齐 `gold_answer`、`gold_sources`、`gold_evidence`、`task_type` 和 `difficulty` 后，再追加到 `benchmark_v1.jsonl`。

## 结构化查询日志

任务 9 已在 `rag_pipeline.ask_with_context` 和 `rag_pipeline.ask_stream` 中统一写入结构化日志。开启项位于 `config.yaml`：

```yaml
enable_query_logging: true
query_log_path: logs/query_runs.jsonl
```

日志为 JSONL，每行对应一次问答，核心字段如下：

| 字段 | 说明 |
|---|---|
| `question` / `standalone_question` | 原始问题和多轮改写后的独立问题。 |
| `route` | 本次使用的检索路线，例如 `mixed`、`mixed_multi_query`、`hyde`。 |
| `llm_model` / `embedding_device` | 生成模型和向量编码设备，用于区分不同实验环境。 |
| `index_version` | 当前查询使用的索引版本，优先来自向量库目录下的 `index_manifest.json`。 |
| `feature_flags` | 当前查询启用的核心增强开关，包括 rerank、query expansion、context compression、parent retrieval 和 `agentic_query`。 |
| `query_variants` | 预留的 Query Expansion 变体字段，当前日志结构已支持。 |
| `retrieved_sources` | 命中的来源文件、页码、预览文本和可选 rerank 分数。 |
| `agent_trace` | Agentic RAG 启用时记录规划、证据验证和 repair 结果；未启用时为空。 |
| `context` | 生成阶段上下文统计，包括召回片段数、上下文片段数和字符数变化。 |
| `elapsed` | `rewrite`、`retrieve`、`generate`、`total` 分阶段耗时。 |
| `error` | LLM 不可用等异常状态；正常问答为 `null`。 |

基线脚本会调用 `ask_with_context`，因此开启日志后，人工问答和批量基线运行会复用同一套字段，便于按题型、路线、模型和耗时瓶颈做后续分析。
