# 论文知识库问答系统

本项目是一个面向论文 PDF 的本地 RAG 问答系统，支持 PDF 入库、混合检索、HyDE、Query Expansion、Rerank、Parent Retrieval、Context Compression、多轮对话、Streamlit Web 界面、基准评估和结构化查询日志。

## 新手入口

| 你想做什么 | 推荐阅读 |
|---|---|
| 第一次运行项目 | [快速开始](docs/getting-started.md) |
| 遇到启动、模型或检索问题 | [FAQ](docs/FAQ.md) |
| 理解项目结构和重构路线 | [架构说明](docs/architecture.md) |

## 目录概览

| 路径 | 说明 |
|---|---|
| `app.py` / `app_services.py` / `app_state.py` | Streamlit Web 入口、UI service 和会话状态工具。 |
| `main.py` / `query.py` | CLI 入口和交互式问答。 |
| `build_knowledge.py` | PDF 入库与向量库构建入口。 |
| `rag_pipeline.py` | 当前 RAG 主编排入口，后续会逐步拆分为更小模块。 |
| `hybrid_retriever.py` / `retrieval_router.py` | 混合检索实现和检索路线选择。 |
| `generation_service.py` / `context_builder.py` | 答案生成、prompt 构造、流式 token 输出和生成阶段上下文构建。 |
| `context_compression.py` / `parent_retrieval.py` | 上下文压缩和 parent 回溯。 |
| `query_logger.py` / `feedback.py` / `source_utils.py` | 结构化日志、失败样本回流和来源序列化。 |
| `benchmarks/` | 人工标注基准集和 baseline 脚本。 |
| `eval/` | 离线评估脚本和指标。 |
| `tests/` | 单元测试与集成边界测试。 |
| `prompts/` | RAG 和 HyDE prompt 模板。 |
| `paper_rag/` | 渐进包化 facade，当前 re-export 根目录稳定模块。 |

## 环境准备

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本项目默认使用本地模型与本地向量库。运行完整问答前，请确认：

| 依赖 | 检查点 |
|---|---|
| Ollama | 已启动服务，并拉取 `config.yaml` 中的 `llm_model`。 |
| Embedding / Reranker 模型 | `config.yaml` 中的本地路径或模型名可用。 |
| PDF 语料 | `papers/` 下存在待入库论文。 |

## 常用命令

| 场景 | 命令 |
|---|---|
| 构建或刷新知识库 | `.\.venv\Scripts\python.exe main.py build` |
| CLI 问答 | `.\.venv\Scripts\python.exe main.py query` |
| Web 问答 | `streamlit run app.py` |
| 基准集格式检查 | `.\.venv\Scripts\python.exe benchmarks\run_baseline.py --no-generate` |
| 完整 baseline | `.\.venv\Scripts\python.exe benchmarks\run_baseline.py` |
| 离线评估 | `.\.venv\Scripts\python.exe eval\run_eval.py --input benchmarks\baseline_results_qwen2.5_3b.jsonl --label qwen2_5_3b` |

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

常用定向测试：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_query_logger
.\.venv\Scripts\python.exe -m unittest tests.test_feedback
.\.venv\Scripts\python.exe -m unittest tests.test_source_utils
```

`pyproject.toml` 中记录了项目元数据和常用命令说明；当前测试仍以 `unittest` 为准，不额外引入测试框架。

## 日志与反馈

| 文件 | 用途 |
|---|---|
| `logs/query_runs.jsonl` | 每次问答的 route、模型、来源、上下文统计和耗时。 |
| `benchmarks/feedback/feedback.jsonl` | 用户手动记录的失败样本暂存区。 |

格式化查看最后一条查询日志：

```powershell
Get-Content logs\query_runs.jsonl -Tail 1 | ConvertFrom-Json | ConvertTo-Json -Depth 20
```

## 工程化重构路线

当前阶段保留根目录入口，优先通过小步重构建立模块边界。后续目标是逐步迁移到 `paper_rag/` 包结构，并让 `rag_pipeline.py` 退化为兼容 facade。

目前 `paper_rag/` 已提供兼容 facade，例如：

```python
from paper_rag.retrieval import RetrievalRouter
from paper_rag.generation import build_rag_prompt
from paper_rag.observability import TraceTimer, source_from_doc, write_query_log
from paper_rag.config import RagSettings, get_setting
from paper_rag.ui import build_feedback_payload
```

第一批低耦合实现已迁入包内：

| 包内模块 | 根目录兼容薄壳 |
|---|---|
| `paper_rag.config` | `utils/config_loader.py` 仍负责 YAML 读取 |
| `paper_rag.observability.sources` | `source_utils.py` |
| `paper_rag.observability.feedback` | `feedback.py` |
| `paper_rag.observability.query_logger` | `query_logger.py` |
| `paper_rag.observability.service` | `rag_pipeline._write_query_log` 兼容包装 |
| `paper_rag.observability.trace` | `rag_pipeline.py` 阶段耗时统计 |
| `paper_rag.generation.service` | `generation_service.py` |
| `paper_rag.generation.context` | `context_builder.py` |
| `paper_rag.generation.parent_retrieval` | `parent_retrieval.py` |
| `paper_rag.retrieval.router` | `retrieval_router.py` |
| `paper_rag.ui.services` | `app_services.py` |

详见 [docs/architecture.md](docs/architecture.md)。
