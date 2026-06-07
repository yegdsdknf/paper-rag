# Paper-RAG FAQ

> 常见问题优先从 `python main.py doctor` 的诊断结果入手。`ERROR` 先修，`WARN` 视情况处理。

---

## 1. 为什么 `python main.py doctor` 报 Ollama 未连接？

常见原因：

| 原因 | 处理 |
|---|---|
| Ollama 服务未启动 | 运行 `ollama serve` |
| 端口 `11434` 不可访问 | 检查防火墙、代理或端口占用 |
| Ollama 安装异常 | 重新安装或重启 Ollama |

验证：

```powershell
.\.venv\Scripts\python.exe main.py doctor
```

---

## 2. 为什么提示 LLM 模型未下载？

`config.yaml` 中的 `llm_model` 必须存在于 Ollama 模型列表中。

查看当前配置：

```powershell
Select-String -Path config.yaml -Pattern "llm_model"
```

下载默认模型：

```powershell
ollama pull qwen2.5:3b
```

如果你修改了 `llm_model`，下载对应模型：

```powershell
ollama pull <model>
```

---

## 3. 为什么 embedding 模型不可用？

项目默认按离线方式加载 embedding 模型。常见原因：

| 原因 | 处理 |
|---|---|
| HuggingFace 模型未提前下载到本机缓存 | 先准备本地模型缓存 |
| `embedding_model` 指向的本地路径不存在 | 修改 `config.yaml` 为真实路径 |
| GPU/驱动环境异常 | 先确认 PyTorch 能正常运行，必要时切到 CPU 环境验证 |

诊断命令：

```powershell
.\.venv\Scripts\python.exe main.py doctor
```

如果 `embedding.model` 是 `ERROR`，问答链路通常无法正常检索。

---

## 4. 为什么向量库目录不存在或 collection 为空？

这通常说明知识库还没有构建，或 `config.yaml` 指向了错误的 `persist_directory` / `collection_name`。

处理步骤：

```powershell
.\.venv\Scripts\python.exe main.py build
.\.venv\Scripts\python.exe main.py doctor
```

如果仍然为空，检查：

| 检查项 | 说明 |
|---|---|
| `papers/` | 是否有 PDF |
| `persist_directory` | 是否指向你刚构建的向量库 |
| `collection_name` | 是否和构建时一致 |
| build 日志 | 是否有 PDF 解析或模型错误 |

---

## 5. 为什么 Reranker 不可用但系统还能回答？

Reranker 是精排增强能力，不是问答链路的硬依赖。

| 状态 | 含义 |
|---|---|
| Reranker 可用 | 检索结果会经过二阶段精排 |
| Reranker 不可用 | 系统降级使用原始检索顺序，仍可回答 |

处理方式：

| 目标 | 处理 |
|---|---|
| 想启用精排 | 确认 `reranker_model` 路径存在 |
| 暂时不需要精排 | 在 `config.yaml` 中关闭 `enable_rerank` |

---

## 6. 为什么回答很慢？

常见瓶颈：

| 阶段 | 可能原因 |
|---|---|
| 启动 | embedding、Chroma、BM25 或 reranker 冷启动 |
| 检索 | Query Expansion 会额外调用 LLM 改写 query |
| 生成 | LLM 模型较大、上下文较长、CPU 推理 |
| 视觉索引 | build 阶段启用 vision analysis 会更慢 |

排查方式：

```powershell
.\.venv\Scripts\python.exe main.py doctor
```

如果 `diagnostics.elapsed` 是 `WARN`，查看每个检查项的 `elapsed_sec`。问答运行后也可以查看结构化日志：

```powershell
Get-Content logs\query_runs.jsonl -Tail 1 | ConvertFrom-Json | ConvertTo-Json -Depth 20
```

---

## 7. 为什么检索结果不准确？

常见原因：

| 原因 | 处理 |
|---|---|
| 论文没有入库 | 把 PDF 放入 `papers/`，运行 `python main.py build` |
| 问题太泛 | 尝试点名论文、方法、页码或术语 |
| collection 指错 | 运行 `python main.py doctor` 检查 Chroma collection |
| Query Expansion 引入噪声 | 暂时关闭 `enable_query_expansion` 对比 |
| Reranker 不可用 | 检查 `reranker_model` 或关闭 rerank 做对照 |

建议提问方式：

```text
BERT 论文中的预训练任务有哪些？
ViT 如何把图像切成 patch 后输入 Transformer？
GPT-3 论文如何描述 few-shot learning？
```

---

## 8. 如何添加自己的论文并重建知识库？

1. 将 PDF 放入 `papers/`。
2. 运行：

```powershell
.\.venv\Scripts\python.exe main.py build
```

3. 运行：

```powershell
.\.venv\Scripts\python.exe main.py doctor
```

4. 启动 Web UI：

```powershell
streamlit run app.py
```

如果 Web UI 已经打开，可以使用侧边栏上传 PDF 并点击“一键入库”。入库完成后，页面会刷新检索器。

---

## 9. 旧的根目录 import 还能继续用吗？

可以继续用，但新代码推荐改成 `paper_rag.*` 包路径。

当前根目录中的 `generation_service.py`、`hybrid_retriever.py`、`query_expansion.py`、`reranker.py`、`context_builder.py`、`app_services.py` 等文件主要是兼容薄壳，用于保护旧 import、测试 patch 和潜在外部调用。真实实现已迁入 `paper_rag.generation`、`paper_rag.retrieval`、`paper_rag.observability`、`paper_rag.ui` 等包内模块。

| 场景 | 建议 |
|---|---|
| 新增代码 | 优先使用 `paper_rag.*` 包路径 |
| 旧脚本还在 `from hybrid_retriever import HybridRetriever` | 暂时可继续运行 |
| 想确认替代路径 | 查看 `paper_rag.compat.COMPAT_WRAPPER_REPLACEMENTS` |
| 担心薄壳被删除 | 当前策略是 `keep_compat_wrapper`，不会一次性删除 |

如果后续需要退场某个兼容薄壳，会先调整 `paper_rag.compat.COMPAT_WRAPPER_RETIREMENT_POLICY`，再同步 README、FAQ 和迁移测试。
