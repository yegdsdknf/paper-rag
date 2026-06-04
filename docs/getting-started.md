# Paper-RAG 快速开始

> 目标：先用 `doctor` 检查环境，再构建知识库，最后启动 Web UI 完成第一次论文问答。

---

## 1. 安装依赖

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果还没有虚拟环境，先创建并激活：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

---

## 2. 先运行启动诊断

```powershell
.\.venv\Scripts\python.exe main.py doctor
```

诊断报告会检查：

| 检查项 | 作用 |
|---|---|
| 配置文件 | 确认 `config.yaml` 能被读取并转换为运行配置 |
| 向量库目录与 Chroma collection | 确认知识库已经构建且 collection 不为空 |
| 索引 manifest | 确认索引元信息是否存在、chunk schema 是否一致 |
| Ollama 服务和模型 | 确认本地 LLM 服务可访问，`llm_model` 已下载 |
| Embedding 模型 | 确认本地 embedding 能完成一次短查询 |
| Reranker 模型 | 如果启用 rerank，确认精排模型可定位 |
| 论文目录 | 确认 `papers/` 中有 PDF |

状态含义：

| 状态 | 含义 |
|---|---|
| `OK` | 当前检查通过 |
| `WARN` | 不一定阻断运行，但建议处理 |
| `ERROR` | 会影响启动或问答，需要先修复 |

机器可读输出：

```powershell
.\.venv\Scripts\python.exe main.py doctor --json
```

---

## 3. 准备 Ollama

如果 `doctor` 提示 Ollama 未连接，先启动服务：

```powershell
ollama serve
```

如果提示 LLM 模型未下载，根据 `config.yaml` 中的 `llm_model` 拉取模型，例如：

```powershell
ollama pull qwen2.5:3b
```

如果需要使用 reasoning 对照模型：

```powershell
ollama pull deepseek-r1:7b
```

---

## 4. 构建知识库

项目自带示例论文在 `papers/` 目录。首次运行或新增论文后，执行：

```powershell
.\.venv\Scripts\python.exe main.py build
```

构建完成后，再运行一次诊断：

```powershell
.\.venv\Scripts\python.exe main.py doctor
```

确认向量库目录、Chroma collection、Embedding 模型和 Ollama 模型都不是 `ERROR`。

---

## 5. 启动 Web UI

```powershell
streamlit run app.py
```

打开页面后，试试第一个问题：

```text
BERT 和 GPT 的主要区别是什么？
```

你应该能看到：

| 页面区域 | 说明 |
|---|---|
| 回答正文 | LLM 基于检索上下文生成的回答 |
| 检索策略 | mixed 或 HyDE 等路由信息 |
| 参考来源 | 检索到的论文片段、文件名和页码 |
| 反馈区 | 可记录答非所问、证据不足等失败样本 |

---

## 6. CLI 问答

如果只想在终端里问答：

```powershell
.\.venv\Scripts\python.exe main.py query
```

Web UI 更适合第一次体验和面试展示；CLI 更适合快速验证链路。

---

## 7. 添加自己的论文

1. 将 PDF 放入 `papers/` 目录。
2. 运行：

```powershell
.\.venv\Scripts\python.exe main.py build
```

3. 运行：

```powershell
.\.venv\Scripts\python.exe main.py doctor
```

4. 启动或刷新 Web UI。

---

## 8. 出错怎么办

先运行：

```powershell
.\.venv\Scripts\python.exe main.py doctor
```

再查看 [FAQ](FAQ.md)。常见问题包括 Ollama 未启动、模型未下载、embedding 模型不可用、向量库为空、reranker 不可用、回答慢和检索不准。
