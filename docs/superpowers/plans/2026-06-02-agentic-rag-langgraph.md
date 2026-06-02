# Agentic RAG LangGraph 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 Paper RAG 增加基于 LangGraph 的可控 Agentic RAG 分支，用于复杂论文问答的证据规划、检索、校验、补查和上下文组装。

**架构：** 第一阶段保留 `rag_pipeline.py` 为兼容主入口，新增 `paper_rag.agentic` 包承载 LangGraph workflow。Agent 分支只在配置或命令开关启用时运行，对复杂题执行 plan -> collect -> verify -> repair -> assemble -> generate，并把 `agent_trace` 写入日志和评估报告。

**技术栈：** Python 3.11、unittest、LangGraph、LangChain Document、Chroma 只读检索、现有 Ollama LLM factory、现有 benchmark/eval/logging 工具。

---

## 设计依据

本计划实现已批准规格：[2026-06-02-agentic-rag-langgraph-design.md](C:/Users/Admin/projects/paper-rag/docs/superpowers/specs/2026-06-02-agentic-rag-langgraph-design.md)。

## 文件结构

| 文件 | 操作 | 职责 |
|---|---|---|
| `requirements.txt` | 修改 | 增加 `langgraph>=0.2`。 |
| `config.yaml` | 修改 | 增加 agentic 配置项，默认关闭。 |
| `paper_rag/config/settings.py` | 修改 | 为 agentic 配置添加 typed settings。 |
| `benchmarks/benchmark_v1.jsonl` | 修改 | 增加 4-6 条 `figure` 样本。 |
| `benchmarks/run_baseline.py` | 修改 | 增加 `--agent` / `--no-agent`、独立输出文件和结果中的 `agent_trace`。 |
| `eval/run_eval.py` | 修改 | 汇总 agent 指标并写入报告。 |
| `paper_rag/agentic/__init__.py` | 创建 | 导出 agentic workflow 公开入口。 |
| `paper_rag/agentic/schema.py` | 创建 | TypedDict、状态结构、日志序列化工具。 |
| `paper_rag/agentic/json_utils.py` | 创建 | 容错 JSON 解析与 schema 清洗。 |
| `paper_rag/agentic/planner.py` | 创建 | 规则骨架 + LLM 补全 evidence goals。 |
| `paper_rag/agentic/verifier.py` | 创建 | 关键词预筛 + LLM 支撑判断 + fallback。 |
| `paper_rag/agentic/collector.py` | 创建 | 按 goal 类型调用现有检索能力。 |
| `paper_rag/agentic/context.py` | 创建 | 重排、过滤、标注 docs，构造 verified evidence summary。 |
| `paper_rag/agentic/graph.py` | 创建 | LangGraph 状态机与 `run_agentic_rag`。 |
| `prompts/agent_planner_prompt.txt` | 创建 | Planner JSON 输出提示词。 |
| `prompts/agent_verifier_prompt.txt` | 创建 | Verifier JSON 输出提示词。 |
| `prompts/agent_context_prompt.txt` | 创建 | 生成阶段的 agentic 上下文约束。 |
| `generation_service.py` / `paper_rag/generation/service.py` | 修改 | 支持把 verified evidence summary 前置到最终 prompt。 |
| `rag_pipeline.py` | 修改 | 接入 agentic 非流式与流式路径，写入 `agent_trace`。 |
| `main.py` / `query.py` | 修改 | 支持 CLI agent 开关。 |
| `app.py` / `paper_rag/ui/services.py` | 修改 | 支持 Streamlit agent 开关和 agent status 事件。 |
| `tests/test_agentic_eval_metrics.py` | 创建 | agent 指标计算测试。 |
| `tests/test_agentic_schema.py` | 创建 | schema 序列化和 JSON 容错解析测试。 |
| `tests/test_agentic_planner.py` | 创建 | planner 规则骨架、LLM 补全、fallback 测试。 |
| `tests/test_agentic_verifier.py` | 创建 | verifier 状态解析和 fallback 测试。 |
| `tests/test_agentic_collector.py` | 创建 | goal 类型到检索策略映射测试。 |
| `tests/test_agentic_context.py` | 创建 | 上下文重排、过滤、summary 构造测试。 |
| `tests/test_agentic_graph.py` | 创建 | LangGraph 状态流转、补查一轮限制测试。 |
| `tests/test_agentic_rag_pipeline.py` | 创建 | `rag_pipeline` 开关、route、日志字段测试。 |

## 任务 1：补充 figure benchmark 和 agent 指标评估

**文件：**
- 修改：`benchmarks/benchmark_v1.jsonl`
- 修改：`eval/run_eval.py`
- 创建：`tests/test_agentic_eval_metrics.py`

- [ ] **步骤 1：编写失败的 agent 指标测试**

在 `tests/test_agentic_eval_metrics.py` 创建测试，验证 `eval.run_eval.evaluate_rows` 能聚合 `agent_trace`。

```python
import unittest

from eval.run_eval import evaluate_rows


class AgenticEvalMetricsTest(unittest.TestCase):
    def test_evaluate_rows_summarizes_agent_trace_metrics(self):
        rows = [
            {
                "id": "a1",
                "question": "GPT-3 使用 Transformer 结构的证据在哪一页？",
                "gold_sources": [{"file": "gpt3.pdf", "page": 7}],
                "gold_evidence": ["same model and architecture as GPT-2"],
                "predicted_answer": "证据在 gpt3.pdf 第 7 页。",
                "retrieved_sources": [{"file": "gpt3.pdf", "page": 7}],
                "elapsed_sec": 2.0,
                "agent_trace": {
                    "enabled": True,
                    "plan": [{"id": "g1"}, {"id": "g2"}],
                    "verification": [
                        {"goal_id": "g1", "status": "supported"},
                        {"goal_id": "g2", "status": "partial"},
                    ],
                    "repair_rounds": 1,
                    "repair_success": True,
                    "agent_elapsed_sec": 0.8,
                },
            },
            {
                "id": "a2",
                "question": "BERT 的全称是什么？",
                "gold_sources": [{"file": "bert.pdf", "page": 0}],
                "gold_evidence": ["Bidirectional Encoder Representations"],
                "predicted_answer": "BERT 的全称是 Bidirectional Encoder Representations from Transformers。",
                "retrieved_sources": [{"file": "bert.pdf", "page": 0}],
                "elapsed_sec": 1.0,
            },
        ]

        report = evaluate_rows(rows, label="agentic_demo", k=5)

        self.assertEqual(report["agent"]["enabled_count"], 1)
        self.assertEqual(report["agent"]["avg_evidence_goal_count"], 2.0)
        self.assertEqual(report["agent"]["goal_support_rate"], 0.5)
        self.assertEqual(report["agent"]["repair_trigger_rate"], 1.0)
        self.assertEqual(report["agent"]["repair_success_rate"], 1.0)
        self.assertEqual(report["agent"]["avg_agent_elapsed_sec"], 0.8)
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_eval_metrics
```

预期：FAIL，报错包含 `KeyError: 'agent'` 或断言找不到 agent 指标。

- [ ] **步骤 3：实现 agent 指标汇总**

在 `eval/run_eval.py` 增加 `_agent_summary`，并在 `evaluate_rows` 的返回 dict 中加入 `"agent": _agent_summary(evaluated)`。

```python
def _agent_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    traces = [
        item.get("agent_trace") or {}
        for item in items
        if (item.get("agent_trace") or {}).get("enabled")
    ]
    enabled_count = len(traces)
    if not traces:
        return {
            "enabled_count": 0,
            "avg_evidence_goal_count": 0.0,
            "goal_support_rate": 0.0,
            "repair_trigger_rate": 0.0,
            "repair_success_rate": 0.0,
            "avg_agent_elapsed_sec": 0.0,
        }

    goal_counts = [len(trace.get("plan") or []) for trace in traces]
    verification_rows = [
        row
        for trace in traces
        for row in (trace.get("verification") or [])
    ]
    supported_count = sum(1 for row in verification_rows if row.get("status") == "supported")
    repair_traces = [trace for trace in traces if int(trace.get("repair_rounds") or 0) > 0]
    successful_repairs = [trace for trace in repair_traces if trace.get("repair_success")]
    elapsed_values = [float(trace.get("agent_elapsed_sec") or 0.0) for trace in traces]

    return {
        "enabled_count": enabled_count,
        "avg_evidence_goal_count": _avg(goal_counts),
        "goal_support_rate": round(supported_count / len(verification_rows), 4) if verification_rows else 0.0,
        "repair_trigger_rate": round(len(repair_traces) / enabled_count, 4),
        "repair_success_rate": round(len(successful_repairs) / len(repair_traces), 4) if repair_traces else 0.0,
        "avg_agent_elapsed_sec": _avg(elapsed_values),
    }
```

在 `evaluate_rows` 构造每个 `item` 时透传输入行里的 `agent_trace`：

```python
"agent_trace": row.get("agent_trace") or {},
```

- [ ] **步骤 4：新增 figure benchmark 样本**

在 `benchmarks/benchmark_v1.jsonl` 末尾追加 4-6 条 `task_type="figure"` 样本，优先围绕 `deepseekr1.pdf` Figure 14。使用现有页码习惯，DeepSeek-R1 的视觉摘要在 page 51，正文说明可在 page 52 附近。

示例追加记录：

```json
{"id":"q026","question":"DeepSeek-R1 的 multilingual safety performance 图在哪一页？","gold_answer":"该图表证据位于 deepseekr1.pdf 第 51 页，图题为 Figure 14 | Multilingual safety performance。","gold_sources":[{"file":"deepseekr1.pdf","page":51}],"gold_evidence":["Figure 14","Multilingual safety performance"],"task_type":"figure","difficulty":"medium","notes":"figure 定位题，验证 vision summary 页能被召回。"}
{"id":"q027","question":"DeepSeek-R1 在 multilingual safety performance 图里整体表现如何？","gold_answer":"图表用于比较 DeepSeek-R1 与 DeepSeek-V3、o1 等模型在多语言安全测试中的表现，并区分是否使用风险控制系统。","gold_sources":[{"file":"deepseekr1.pdf","page":51},{"file":"deepseekr1.pdf","page":52}],"gold_evidence":["Multilingual safety performance","risk control system","DeepSeek-R1"],"task_type":"figure","difficulty":"hard","notes":"figure 解释题，验证 vision summary 与正文说明联合使用。"}
{"id":"q028","question":"Figure 14 中 R1-check 和 V3-check 代表什么？","gold_answer":"R1-check 和 V3-check 表示分别使用 DeepSeek-R1 与 DeepSeek-V3 风险控制系统评估后的结果。","gold_sources":[{"file":"deepseekr1.pdf","page":51},{"file":"deepseekr1.pdf","page":52}],"gold_evidence":["V3-check and R1-check represent the risk control system evaluation results"],"task_type":"figure","difficulty":"medium","notes":"figure 图例解释题，验证图表摘要和同页/相邻页文本。"}
{"id":"q029","question":"结合 Figure 14 和正文，风险控制系统对 DeepSeek-R1 安全评估有什么作用？","gold_answer":"论文用 Figure 14 比较带与不带风险控制系统时的安全分数，说明 R1 类推理模型更依赖风险控制系统进行安全检查。","gold_sources":[{"file":"deepseekr1.pdf","page":51},{"file":"deepseekr1.pdf","page":53}],"gold_evidence":["with and without the risk control system","rely more heavily on the risk control system"],"task_type":"figure","difficulty":"hard","notes":"图文联合证据题，验证 agentic figure 分支。"}
```

- [ ] **步骤 5：运行测试验证通过**

运行：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_eval_metrics
.\.venv\Scripts\python.exe benchmarks\run_baseline.py --no-generate
```

预期：测试 PASS；benchmark schema 检查 PASS。

- [ ] **步骤 6：Commit**

```powershell
git add benchmarks/benchmark_v1.jsonl eval/run_eval.py tests/test_agentic_eval_metrics.py
git commit -m "test: add agentic eval metrics and figure benchmark"
```

## 任务 2：建立 agentic schema 与 JSON 解析工具

**文件：**
- 创建：`paper_rag/agentic/__init__.py`
- 创建：`paper_rag/agentic/schema.py`
- 创建：`paper_rag/agentic/json_utils.py`
- 创建：`tests/test_agentic_schema.py`

- [ ] **步骤 1：编写失败的 schema 测试**

创建 `tests/test_agentic_schema.py`。

```python
import unittest

from langchain_core.documents import Document

from paper_rag.agentic.json_utils import parse_json_object
from paper_rag.agentic.schema import docs_to_agent_sources, normalize_goal


class AgenticSchemaTest(unittest.TestCase):
    def test_parse_json_object_extracts_json_from_noisy_model_output(self):
        text = "结果如下：\n```json\n{\"goals\": [{\"id\": \"g1\"}]}\n```"

        parsed = parse_json_object(text)

        self.assertEqual(parsed["goals"][0]["id"], "g1")

    def test_normalize_goal_limits_status_and_source_hint(self):
        goal = normalize_goal(
            {
                "id": "",
                "goal_type": "weird",
                "claim": "GPT-3 uses Transformer",
                "query": "",
                "source_hint": "not-in-index.pdf",
                "page_hint": "7",
            },
            index=0,
            allowed_sources={"gpt3.pdf"},
        )

        self.assertEqual(goal["id"], "g1")
        self.assertEqual(goal["goal_type"], "method_overview")
        self.assertEqual(goal["query"], "GPT-3 uses Transformer")
        self.assertEqual(goal["source_hint"], "")
        self.assertEqual(goal["page_hint"], 7)

    def test_docs_to_agent_sources_serializes_documents(self):
        docs = [
            Document(
                page_content="Figure 14 | Multilingual safety performance",
                metadata={"source": "./papers/deepseekr1.pdf", "page": 51},
            )
        ]

        sources = docs_to_agent_sources(docs)

        self.assertEqual(sources[0]["file"], "deepseekr1.pdf")
        self.assertEqual(sources[0]["page"], 51)
        self.assertIn("Figure 14", sources[0]["content_preview"])
```

- [ ] **步骤 2：运行测试验证失败**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_schema
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'paper_rag.agentic'`。

- [ ] **步骤 3：实现 schema 和 JSON 解析**

创建 `paper_rag/agentic/schema.py`，定义 `TypedDict` 和清洗函数。

```python
from __future__ import annotations

import os
from typing import Any, Literal, TypedDict

from langchain_core.documents import Document


GoalType = Literal["page_evidence", "compare_dimension", "method_overview", "figure_evidence"]
VerificationStatus = Literal["supported", "partial", "unsupported"]


class EvidenceGoal(TypedDict, total=False):
    id: str
    goal_type: GoalType
    claim: str
    query: str
    source_hint: str
    page_hint: int | None


class VerifiedEvidence(TypedDict, total=False):
    goal_id: str
    claim: str
    status: VerificationStatus
    supporting_sources: list[dict[str, Any]]
    missing_terms: list[str]


class AgenticRagState(TypedDict, total=False):
    question: str
    standalone_question: str
    task_type: str
    route: str
    source_hints: list[str]
    goals: list[EvidenceGoal]
    collected_docs: list[Document]
    verified_evidence: list[VerifiedEvidence]
    final_docs: list[Document]
    repair_rounds: int
    missing_goal_ids: list[str]
    fallback_reason: str | None
    agent_trace: dict[str, Any]
    answer: str
    sources: list[dict[str, Any]]
    elapsed: dict[str, float]


VALID_GOAL_TYPES = {"page_evidence", "compare_dimension", "method_overview", "figure_evidence"}
VALID_STATUSES = {"supported", "partial", "unsupported"}


def _source_file(source: object) -> str:
    return os.path.basename(str(source).replace("\\", "/"))


def normalize_goal(raw: dict[str, Any], index: int, allowed_sources: set[str] | None = None) -> EvidenceGoal:
    claim = str(raw.get("claim") or "").strip()
    query = str(raw.get("query") or claim).strip()
    goal_type = str(raw.get("goal_type") or "method_overview").strip()
    if goal_type not in VALID_GOAL_TYPES:
        goal_type = "method_overview"

    source_hint = _source_file(raw.get("source_hint") or "")
    if allowed_sources is not None and source_hint not in allowed_sources:
        source_hint = ""

    page_hint: int | None
    try:
        page_hint = int(raw.get("page_hint"))
    except (TypeError, ValueError):
        page_hint = None

    return {
        "id": str(raw.get("id") or f"g{index + 1}").strip() or f"g{index + 1}",
        "goal_type": goal_type,  # type: ignore[typeddict-item]
        "claim": claim,
        "query": query,
        "source_hint": source_hint,
        "page_hint": page_hint,
    }


def docs_to_agent_sources(docs: list[Document]) -> list[dict[str, Any]]:
    sources = []
    for doc in docs:
        metadata = dict(doc.metadata)
        sources.append(
            {
                "file": _source_file(metadata.get("source") or metadata.get("source_file") or ""),
                "page": int(metadata.get("page") if metadata.get("page") is not None else -1),
                "content_preview": str(doc.page_content).replace("\n", " ")[:180],
            }
        )
    return sources
```

创建 `paper_rag/agentic/json_utils.py`。

```python
from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object")
    return parsed
```

创建 `paper_rag/agentic/__init__.py`。

```python
from paper_rag.agentic.schema import AgenticRagState, EvidenceGoal, VerifiedEvidence

__all__ = ["AgenticRagState", "EvidenceGoal", "VerifiedEvidence"]
```

- [ ] **步骤 4：运行测试验证通过**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_schema
```

预期：PASS。

- [ ] **步骤 5：Commit**

```powershell
git add paper_rag/agentic/__init__.py paper_rag/agentic/schema.py paper_rag/agentic/json_utils.py tests/test_agentic_schema.py
git commit -m "feat: add agentic schema utilities"
```

## 任务 3：实现 Planner 的规则骨架与 LLM 补全

**文件：**
- 创建：`paper_rag/agentic/planner.py`
- 创建：`prompts/agent_planner_prompt.txt`
- 创建：`tests/test_agentic_planner.py`

- [ ] **步骤 1：编写失败的 Planner 测试**

创建 `tests/test_agentic_planner.py`。

```python
import unittest

from paper_rag.agentic.planner import plan_evidence_goals


class FakeLLM:
    def invoke(self, prompt):
        class Response:
            content = '{"goals":[{"id":"g1","goal_type":"page_evidence","claim":"GPT-3 architecture evidence","query":"GPT-3 same model and architecture as GPT-2","source_hint":"gpt3.pdf","page_hint":7}]}'

        return Response()


class AgenticPlannerTest(unittest.TestCase):
    def test_planner_uses_source_hint_and_limits_goal_count(self):
        result = plan_evidence_goals(
            question="GPT-3 使用 Transformer 结构的证据在哪一页？",
            standalone_question="GPT-3 使用 Transformer 结构的证据在哪一页？",
            source_hints=["gpt3.pdf"],
            task_type="evidence",
            llm=FakeLLM(),
        )

        self.assertEqual(result[0]["goal_type"], "page_evidence")
        self.assertEqual(result[0]["source_hint"], "gpt3.pdf")
        self.assertEqual(result[0]["page_hint"], 7)

    def test_planner_falls_back_to_rule_goal_when_llm_fails(self):
        class BrokenLLM:
            def invoke(self, prompt):
                raise RuntimeError("offline")

        result = plan_evidence_goals(
            question="BERT 和 T5 在预训练目标上有什么不同？",
            standalone_question="BERT 和 T5 在预训练目标上有什么不同？",
            source_hints=["bert.pdf", "t5.pdf"],
            task_type="compare",
            llm=BrokenLLM(),
        )

        self.assertEqual(len(result), 2)
        self.assertTrue(all(goal["goal_type"] == "compare_dimension" for goal in result))
```

- [ ] **步骤 2：运行测试验证失败**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_planner
```

预期：FAIL，报错包含 `ModuleNotFoundError` 或 `ImportError`。

- [ ] **步骤 3：创建 Planner prompt**

创建 `prompts/agent_planner_prompt.txt`。

```text
你是论文 RAG 的证据规划器。请根据用户问题生成 1 到 4 个可执行、可验证的 evidence goals。

只输出 JSON 对象，不要 Markdown，不要解释。

字段要求：
- goals: 数组
- 每个 goal 包含 id, goal_type, claim, query, source_hint, page_hint
- goal_type 只能是 page_evidence, compare_dimension, method_overview, figure_evidence
- source_hint 必须来自可用来源；无法确定时为空字符串
- page_hint 不确定时为 null

用户原始问题：
{question}

独立问题：
{standalone_question}

任务类型：
{task_type}

可用来源：
{source_hints}
```

- [ ] **步骤 4：实现 Planner**

在 `paper_rag/agentic/planner.py` 中实现规则骨架、LLM 调用和 fallback。

```python
from __future__ import annotations

from typing import Any

from paper_rag.agentic.json_utils import parse_json_object
from paper_rag.agentic.schema import EvidenceGoal, normalize_goal
from utils.prompt_loader import load_prompt


def _rule_goals(standalone_question: str, source_hints: list[str], task_type: str) -> list[dict[str, Any]]:
    if task_type == "compare" and source_hints:
        return [
            {
                "id": f"g{index + 1}",
                "goal_type": "compare_dimension",
                "claim": f"{source} 与问题相关的对比证据",
                "query": f"{standalone_question} {source}",
                "source_hint": source,
                "page_hint": None,
            }
            for index, source in enumerate(source_hints[:4])
        ]
    goal_type = "figure_evidence" if task_type == "figure" else "page_evidence" if task_type == "evidence" else "method_overview"
    return [
        {
            "id": "g1",
            "goal_type": goal_type,
            "claim": standalone_question,
            "query": standalone_question,
            "source_hint": source_hints[0] if source_hints else "",
            "page_hint": None,
        }
    ]


def _response_text(response: Any) -> str:
    return str(response.content if hasattr(response, "content") else response)


def plan_evidence_goals(
    question: str,
    standalone_question: str,
    source_hints: list[str],
    task_type: str,
    llm: Any | None = None,
) -> list[EvidenceGoal]:
    allowed_sources = set(source_hints)
    raw_goals = _rule_goals(standalone_question, source_hints, task_type)

    if llm is not None:
        try:
            prompt = load_prompt("agent_planner_prompt").format(
                question=question,
                standalone_question=standalone_question,
                task_type=task_type,
                source_hints=", ".join(source_hints) or "无",
            )
            parsed = parse_json_object(_response_text(llm.invoke(prompt)))
            candidate_goals = parsed.get("goals")
            if isinstance(candidate_goals, list) and candidate_goals:
                raw_goals = candidate_goals
        except Exception:
            raw_goals = _rule_goals(standalone_question, source_hints, task_type)

    return [
        normalize_goal(goal, index=index, allowed_sources=allowed_sources)
        for index, goal in enumerate(raw_goals[:4])
    ]
```

- [ ] **步骤 5：运行 Planner 测试**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_planner
```

预期：PASS。

- [ ] **步骤 6：Commit**

```powershell
git add paper_rag/agentic/planner.py prompts/agent_planner_prompt.txt tests/test_agentic_planner.py
git commit -m "feat: add agentic evidence planner"
```

## 任务 4：实现 Verifier 的支撑判断与 fallback

**文件：**
- 创建：`paper_rag/agentic/verifier.py`
- 创建：`prompts/agent_verifier_prompt.txt`
- 创建：`tests/test_agentic_verifier.py`

- [ ] **步骤 1：编写失败的 Verifier 测试**

创建 `tests/test_agentic_verifier.py`。

```python
import unittest

from langchain_core.documents import Document

from paper_rag.agentic.verifier import verify_goal


class AgenticVerifierTest(unittest.TestCase):
    def test_verify_goal_uses_keyword_fallback_without_llm(self):
        goal = {
            "id": "g1",
            "claim": "Figure 14 multilingual safety performance",
            "query": "Figure 14 multilingual safety performance",
            "source_hint": "deepseekr1.pdf",
            "goal_type": "figure_evidence",
        }
        docs = [
            Document(
                page_content="Figure 14 | Multilingual safety performance for DeepSeek-R1",
                metadata={"source": "deepseekr1.pdf", "page": 51},
            )
        ]

        result = verify_goal(goal, docs, llm=None)

        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["supporting_sources"][0]["file"], "deepseekr1.pdf")

    def test_verify_goal_parses_llm_status(self):
        class FakeLLM:
            def invoke(self, prompt):
                class Response:
                    content = '{"status":"partial","reason":"only one part is supported","missing_terms":["NSP"]}'

                return Response()

        goal = {"id": "g1", "claim": "BERT uses MLM and NSP", "query": "BERT MLM NSP"}
        docs = [Document(page_content="BERT uses masked language modeling.", metadata={"source": "bert.pdf", "page": 1})]

        result = verify_goal(goal, docs, llm=FakeLLM())

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["missing_terms"], ["NSP"])
```

- [ ] **步骤 2：运行测试验证失败**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_verifier
```

预期：FAIL，报错包含 `ModuleNotFoundError` 或 `ImportError`。

- [ ] **步骤 3：创建 Verifier prompt**

创建 `prompts/agent_verifier_prompt.txt`。

```text
你是论文 RAG 的证据校验器。判断候选片段是否支持给定 claim。

只输出 JSON 对象，不要 Markdown，不要解释。

status 只能是 supported、partial、unsupported。

Claim:
{claim}

Query:
{query}

候选证据：
{evidence}
```

- [ ] **步骤 4：实现 Verifier**

创建 `paper_rag/agentic/verifier.py`。

```python
from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

from paper_rag.agentic.json_utils import parse_json_object
from paper_rag.agentic.schema import VALID_STATUSES, VerifiedEvidence, docs_to_agent_sources
from utils.prompt_loader import load_prompt


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]{2,}|[\u4e00-\u9fff]{2,}", text)
    }


def _keyword_status(goal: dict[str, Any], docs: list[Document]) -> str:
    terms = _terms(str(goal.get("claim") or goal.get("query") or ""))
    if not terms or not docs:
        return "unsupported"
    evidence_text = " ".join(doc.page_content for doc in docs).lower()
    hits = sum(1 for term in terms if term in evidence_text)
    if hits >= max(1, len(terms) // 2):
        return "supported"
    if hits > 0:
        return "partial"
    return "unsupported"


def _response_text(response: Any) -> str:
    return str(response.content if hasattr(response, "content") else response)


def verify_goal(goal: dict[str, Any], docs: list[Document], llm: Any | None = None) -> VerifiedEvidence:
    status = _keyword_status(goal, docs)
    missing_terms: list[str] = []

    if llm is not None and docs:
        try:
            evidence = "\n\n".join(
                f"[{index + 1}] {doc.metadata.get('source')} p{doc.metadata.get('page')}: {doc.page_content[:800]}"
                for index, doc in enumerate(docs[:4])
            )
            prompt = load_prompt("agent_verifier_prompt").format(
                claim=goal.get("claim") or "",
                query=goal.get("query") or "",
                evidence=evidence,
            )
            parsed = parse_json_object(_response_text(llm.invoke(prompt)))
            candidate_status = str(parsed.get("status") or "").strip()
            if candidate_status in VALID_STATUSES:
                status = candidate_status
            missing = parsed.get("missing_terms")
            if isinstance(missing, list):
                missing_terms = [str(term) for term in missing]
        except Exception:
            status = _keyword_status(goal, docs)

    return {
        "goal_id": str(goal.get("id") or ""),
        "claim": str(goal.get("claim") or ""),
        "status": status,  # type: ignore[typeddict-item]
        "supporting_sources": docs_to_agent_sources(docs[:4]) if status != "unsupported" else [],
        "missing_terms": missing_terms,
    }
```

- [ ] **步骤 5：运行 Verifier 测试**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_verifier
```

预期：PASS。

- [ ] **步骤 6：Commit**

```powershell
git add paper_rag/agentic/verifier.py prompts/agent_verifier_prompt.txt tests/test_agentic_verifier.py
git commit -m "feat: add agentic evidence verifier"
```

## 任务 5：实现 Collector 和 Context 组装

**文件：**
- 创建：`paper_rag/agentic/collector.py`
- 创建：`paper_rag/agentic/context.py`
- 创建：`prompts/agent_context_prompt.txt`
- 创建：`tests/test_agentic_collector.py`
- 创建：`tests/test_agentic_context.py`

- [ ] **步骤 1：编写失败的 Collector 测试**

创建 `tests/test_agentic_collector.py`。

```python
import unittest

from langchain_core.documents import Document

from paper_rag.agentic.collector import collect_for_goal


class FakeRouter:
    def __init__(self):
        self.calls = []

    def route(self, hybrid, question, llm_model="", temperature=0.0):
        self.calls.append(question)
        return [Document(page_content="mixed result", metadata={"source": "gpt3.pdf", "page": 7})], "mixed"


class AgenticCollectorTest(unittest.TestCase):
    def test_page_evidence_goal_uses_goal_query_without_hyde(self):
        router = FakeRouter()
        goal = {
            "id": "g1",
            "goal_type": "page_evidence",
            "query": "GPT-3 same model and architecture as GPT-2",
            "source_hint": "gpt3.pdf",
        }

        docs, route = collect_for_goal(goal, hybrid=object(), router=router)

        self.assertEqual(route, "agentic_page_evidence")
        self.assertEqual(router.calls[0], "GPT-3 same model and architecture as GPT-2")
        self.assertEqual(docs[0].metadata["source"], "gpt3.pdf")

    def test_figure_goal_prefers_vision_summary(self):
        class FigureHybrid:
            vector_store = object()

        def fake_load_vision_docs(hybrid, source_hint):
            return [
                Document(
                    page_content="Figure 14 | Multilingual safety performance",
                    metadata={"source": "deepseekr1.pdf", "page": 51, "paper_region": "vision"},
                )
            ]

        goal = {"id": "g1", "goal_type": "figure_evidence", "query": "Figure 14", "source_hint": "deepseekr1.pdf"}
        docs, route = collect_for_goal(goal, hybrid=FigureHybrid(), router=FakeRouter(), vision_loader=fake_load_vision_docs)

        self.assertEqual(route, "agentic_figure")
        self.assertEqual(docs[0].metadata["paper_region"], "vision")
```

- [ ] **步骤 2：编写失败的 Context 测试**

创建 `tests/test_agentic_context.py`。

```python
import unittest

from langchain_core.documents import Document

from paper_rag.agentic.context import assemble_agentic_context, build_verified_evidence_summary


class AgenticContextTest(unittest.TestCase):
    def test_build_verified_evidence_summary_marks_statuses(self):
        summary = build_verified_evidence_summary(
            [
                {"goal_id": "g1", "claim": "GPT-3 architecture", "status": "supported", "supporting_sources": [{"file": "gpt3.pdf", "page": 7}]},
                {"goal_id": "g2", "claim": "T5 objective", "status": "unsupported", "supporting_sources": []},
            ]
        )

        self.assertIn("Goal g1: supported", summary)
        self.assertIn("gpt3.pdf p7", summary)
        self.assertIn("unsupported", summary)

    def test_assemble_context_prioritizes_supported_docs(self):
        supported = Document(page_content="supported", metadata={"source": "gpt3.pdf", "page": 7})
        noisy = Document(page_content="noisy", metadata={"source": "other.pdf", "page": 1})

        result = assemble_agentic_context(
            docs=[noisy, supported],
            verified_evidence=[{"goal_id": "g1", "status": "supported", "supporting_sources": [{"file": "gpt3.pdf", "page": 7}]}],
            task_type="evidence",
        )

        self.assertEqual(result.final_docs[0].metadata["source"], "gpt3.pdf")
        self.assertIn("已校验证据", result.verified_summary)
```

- [ ] **步骤 3：运行测试验证失败**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_collector tests.test_agentic_context
```

预期：FAIL，报错包含缺少 `paper_rag.agentic.collector` 或 `paper_rag.agentic.context`。

- [ ] **步骤 4：实现 Collector**

创建 `paper_rag/agentic/collector.py`。

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document


VisionLoader = Callable[[Any, str], list[Document]]


def _default_vision_loader(hybrid: Any, source_hint: str) -> list[Document]:
    vector_store = getattr(hybrid, "vector_store", None)
    if vector_store is None:
        return []
    try:
        stored = vector_store.get(include=["documents", "metadatas"])
    except Exception:
        return []
    docs = []
    for content, metadata in zip(stored.get("documents", []), stored.get("metadatas", [])):
        source = str(metadata.get("source") or metadata.get("source_file") or "")
        if source_hint and source_hint not in source:
            continue
        if metadata.get("paper_region") == "vision" or metadata.get("chunk_strategy") == "vision_summary":
            docs.append(Document(page_content=str(content), metadata=dict(metadata)))
    return docs


def collect_for_goal(
    goal: dict[str, Any],
    hybrid: Any,
    router: Any,
    llm_model: str = "",
    temperature: float = 0.0,
    vision_loader: VisionLoader | None = None,
) -> tuple[list[Document], str]:
    goal_type = str(goal.get("goal_type") or "method_overview")
    query = str(goal.get("query") or goal.get("claim") or "")
    source_hint = str(goal.get("source_hint") or "")

    if goal_type == "figure_evidence":
        loader = vision_loader or _default_vision_loader
        vision_docs = loader(hybrid, source_hint)
        if vision_docs:
            return vision_docs, "agentic_figure"

    docs, _route = router.route(hybrid, query, llm_model=llm_model, temperature=temperature)
    if source_hint:
        filtered = [doc for doc in docs if source_hint in str(doc.metadata.get("source") or doc.metadata.get("source_file") or "")]
        docs = filtered or docs

    route_by_type = {
        "page_evidence": "agentic_page_evidence",
        "compare_dimension": "agentic_compare",
        "figure_evidence": "agentic_figure_text_fallback",
        "method_overview": "agentic_method",
    }
    return docs, route_by_type.get(goal_type, "agentic_mixed")
```

- [ ] **步骤 5：实现 Context**

创建 `paper_rag/agentic/context.py`。

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass(frozen=True)
class AgenticContextResult:
    final_docs: list[Document]
    verified_summary: str


def _source_key(source: object, page: object) -> tuple[str, int]:
    file = os.path.basename(str(source).replace("\\", "/"))
    try:
        page_int = int(page)
    except (TypeError, ValueError):
        page_int = -1
    return file, page_int


def build_verified_evidence_summary(verified_evidence: list[dict[str, Any]]) -> str:
    lines = ["【已校验证据】"]
    for row in verified_evidence:
        lines.append(f"- Goal {row.get('goal_id')}: {row.get('status')}")
        if row.get("claim"):
            lines.append(f"  Claim: {row['claim']}")
        sources = row.get("supporting_sources") or []
        if sources:
            rendered = ", ".join(f"{source.get('file')} p{source.get('page')}" for source in sources)
            lines.append(f"  Sources: {rendered}")
    lines.extend(
        [
            "",
            "【回答约束】",
            "- 优先使用 supported 证据。",
            "- partial 证据只能谨慎表述。",
            "- unsupported goal 必须说明未找到足够证据。",
            "- 不要把 A 论文的结论归因给 B 论文。",
        ]
    )
    return "\n".join(lines)


def assemble_agentic_context(
    docs: list[Document],
    verified_evidence: list[dict[str, Any]],
    task_type: str,
) -> AgenticContextResult:
    supported_keys = {
        _source_key(source.get("file"), source.get("page"))
        for row in verified_evidence
        if row.get("status") in {"supported", "partial"}
        for source in (row.get("supporting_sources") or [])
    }

    def score(doc: Document) -> tuple[int, int]:
        key = _source_key(doc.metadata.get("source") or doc.metadata.get("source_file"), doc.metadata.get("page"))
        region = str(doc.metadata.get("paper_region") or "")
        priority = 0
        if key in supported_keys:
            priority += 10
        if task_type == "figure" and region == "vision":
            priority += 5
        return (-priority, int(doc.metadata.get("page") or 0))

    ordered = sorted(docs, key=score)
    if task_type in {"evidence", "figure", "followup"} and supported_keys:
        kept = [
            doc
            for doc in ordered
            if _source_key(doc.metadata.get("source") or doc.metadata.get("source_file"), doc.metadata.get("page")) in supported_keys
            or doc.metadata.get("paper_region") == "vision"
        ]
        ordered = kept or ordered

    return AgenticContextResult(
        final_docs=ordered,
        verified_summary=build_verified_evidence_summary(verified_evidence),
    )
```

创建 `prompts/agent_context_prompt.txt`，内容与 `build_verified_evidence_summary` 的约束一致，用于后续如果需要从文件加载。

- [ ] **步骤 6：运行 Collector 和 Context 测试**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_collector tests.test_agentic_context
```

预期：PASS。

- [ ] **步骤 7：Commit**

```powershell
git add paper_rag/agentic/collector.py paper_rag/agentic/context.py prompts/agent_context_prompt.txt tests/test_agentic_collector.py tests/test_agentic_context.py
git commit -m "feat: add agentic evidence collection and context assembly"
```

## 任务 6：实现 LangGraph workflow

**文件：**
- 修改：`requirements.txt`
- 创建：`paper_rag/agentic/graph.py`
- 创建：`tests/test_agentic_graph.py`

- [ ] **步骤 1：编写失败的 Graph 测试**

创建 `tests/test_agentic_graph.py`。

```python
import unittest

from langchain_core.documents import Document

from paper_rag.agentic.graph import run_agentic_rag


class FakeRouter:
    def route(self, hybrid, question, llm_model="", temperature=0.0):
        if "missing" in question:
            return [], "mixed"
        return [Document(page_content="GPT-3 same model and architecture as GPT-2", metadata={"source": "gpt3.pdf", "page": 7})], "mixed"


class AgenticGraphTest(unittest.TestCase):
    def test_graph_runs_plan_collect_verify_assemble(self):
        result = run_agentic_rag(
            question="GPT-3 使用 Transformer 结构的证据在哪一页？",
            standalone_question="GPT-3 使用 Transformer 结构的证据在哪一页？",
            task_type="evidence",
            source_hints=["gpt3.pdf"],
            hybrid=object(),
            router=FakeRouter(),
            planner_llm=None,
            verifier_llm=None,
        )

        self.assertTrue(result["agent_trace"]["enabled"])
        self.assertEqual(result["route"], "agentic_page_evidence")
        self.assertGreaterEqual(len(result["final_docs"]), 1)

    def test_graph_limits_repair_to_one_round(self):
        result = run_agentic_rag(
            question="missing evidence",
            standalone_question="missing evidence",
            task_type="evidence",
            source_hints=["gpt3.pdf"],
            hybrid=object(),
            router=FakeRouter(),
            planner_llm=None,
            verifier_llm=None,
        )

        self.assertEqual(result["agent_trace"]["repair_rounds"], 1)
        self.assertEqual(result["verified_evidence"][0]["status"], "unsupported")
```

- [ ] **步骤 2：运行测试验证失败**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_graph
```

预期：FAIL，报错包含缺少 `paper_rag.agentic.graph` 或缺少 `langgraph`。

- [ ] **步骤 3：增加 LangGraph 依赖**

在 `requirements.txt` 末尾添加：

```text
langgraph>=0.2
```

如果本地环境还没有安装，运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

预期：安装成功；如果网络受限，按沙箱审批流程申请联网安装。

- [ ] **步骤 4：实现 graph workflow**

创建 `paper_rag/agentic/graph.py`。实现时使用 `StateGraph(AgenticRagState)`，节点内部调用已实现的 planner、collector、verifier、context。

```python
from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from paper_rag.agentic.collector import collect_for_goal
from paper_rag.agentic.context import assemble_agentic_context
from paper_rag.agentic.planner import plan_evidence_goals
from paper_rag.agentic.schema import AgenticRagState
from paper_rag.agentic.verifier import verify_goal


def _plan_node(state: AgenticRagState) -> AgenticRagState:
    goals = plan_evidence_goals(
        question=state["question"],
        standalone_question=state["standalone_question"],
        source_hints=state.get("source_hints", []),
        task_type=state.get("task_type", "method"),
        llm=state.get("_planner_llm"),
    )
    return {"goals": goals}


def _collect_node(state: AgenticRagState) -> AgenticRagState:
    docs = []
    routes = []
    for goal in state.get("goals", []):
        goal_docs, route = collect_for_goal(
            goal,
            hybrid=state["_hybrid"],
            router=state["_router"],
            llm_model=str(state.get("llm_model") or ""),
            temperature=float(state.get("temperature") or 0.0),
        )
        docs.extend(goal_docs)
        routes.append(route)
    return {"collected_docs": docs, "route": routes[0] if routes else "agentic_mixed"}


def _verify_node(state: AgenticRagState) -> AgenticRagState:
    docs = state.get("collected_docs", [])
    verified = [
        verify_goal(goal, docs, llm=state.get("_verifier_llm"))
        for goal in state.get("goals", [])
    ]
    missing = [row["goal_id"] for row in verified if row.get("status") == "unsupported"]
    return {"verified_evidence": verified, "missing_goal_ids": missing}


def _should_repair(state: AgenticRagState) -> str:
    if state.get("missing_goal_ids") and int(state.get("repair_rounds") or 0) < int(state.get("max_repair_rounds") or 1):
        return "repair"
    return "assemble"


def _repair_node(state: AgenticRagState) -> AgenticRagState:
    return {"repair_rounds": int(state.get("repair_rounds") or 0) + 1}


def _assemble_node(state: AgenticRagState) -> AgenticRagState:
    assembled = assemble_agentic_context(
        docs=state.get("collected_docs", []),
        verified_evidence=state.get("verified_evidence", []),
        task_type=state.get("task_type", "method"),
    )
    trace = {
        "enabled": True,
        "plan": state.get("goals", []),
        "verification": state.get("verified_evidence", []),
        "repair_rounds": int(state.get("repair_rounds") or 0),
        "repair_success": bool(state.get("missing_goal_ids")) is False,
        "agent_elapsed_sec": round(time.perf_counter() - float(state.get("_agent_start") or time.perf_counter()), 4),
    }
    return {
        "final_docs": assembled.final_docs,
        "verified_summary": assembled.verified_summary,
        "agent_trace": trace,
    }


def build_agentic_graph():
    graph = StateGraph(AgenticRagState)
    graph.add_node("plan", _plan_node)
    graph.add_node("collect", _collect_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("repair", _repair_node)
    graph.add_node("assemble", _assemble_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "collect")
    graph.add_edge("collect", "verify")
    graph.add_conditional_edges("verify", _should_repair, {"repair": "repair", "assemble": "assemble"})
    graph.add_edge("repair", "collect")
    graph.add_edge("assemble", END)
    return graph.compile()


def run_agentic_rag(
    question: str,
    standalone_question: str,
    task_type: str,
    source_hints: list[str],
    hybrid: Any,
    router: Any,
    planner_llm: Any | None = None,
    verifier_llm: Any | None = None,
    llm_model: str = "",
    temperature: float = 0.0,
    max_repair_rounds: int = 1,
) -> AgenticRagState:
    graph = build_agentic_graph()
    return graph.invoke(
        {
            "question": question,
            "standalone_question": standalone_question,
            "task_type": task_type,
            "source_hints": source_hints,
            "repair_rounds": 0,
            "max_repair_rounds": max_repair_rounds,
            "llm_model": llm_model,
            "temperature": temperature,
            "_hybrid": hybrid,
            "_router": router,
            "_planner_llm": planner_llm,
            "_verifier_llm": verifier_llm,
            "_agent_start": time.perf_counter(),
        }
    )
```

如果 LangGraph 不允许未在 `TypedDict` 中声明的内部 `_hybrid` 字段，补充 `AgenticRagState` 中的内部字段声明，字段名保持下划线前缀。

- [ ] **步骤 5：运行 Graph 测试**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_graph
```

预期：PASS。

- [ ] **步骤 6：Commit**

```powershell
git add requirements.txt paper_rag/agentic/graph.py tests/test_agentic_graph.py
git commit -m "feat: add langgraph agentic rag workflow"
```

## 任务 7：配置与生成 prompt 接入

**文件：**
- 修改：`config.yaml`
- 修改：`paper_rag/config/settings.py`
- 修改：`paper_rag/generation/service.py`
- 修改：`generation_service.py`
- 修改：`tests/test_config_settings.py`
- 创建：`tests/test_agentic_generation_prompt.py`

- [ ] **步骤 1：编写配置测试**

在 `tests/test_config_settings.py` 增加测试：

```python
def test_agentic_settings_have_stable_defaults(self):
    from paper_rag.config.settings import RagSettings

    settings = RagSettings.from_dict(
        {
            "persist_directory": "./db",
            "embedding_model": "BAAI/bge-m3",
        }
    )

    self.assertFalse(settings.enable_agentic_query)
    self.assertTrue(settings.agent_auto_for_complex)
    self.assertEqual(settings.agent_max_repair_rounds, 1)
```

- [ ] **步骤 2：编写生成 prompt 测试**

创建 `tests/test_agentic_generation_prompt.py`。

```python
import unittest

from langchain_core.documents import Document

from paper_rag.generation.service import build_rag_prompt


class AgenticGenerationPromptTest(unittest.TestCase):
    def test_build_rag_prompt_includes_agentic_verified_summary(self):
        docs = [Document(page_content="GPT-3 same model and architecture as GPT-2", metadata={"source": "gpt3.pdf", "page": 7})]

        prompt = build_rag_prompt(
            "GPT-3 使用 Transformer 结构的证据在哪一页？",
            docs,
            verified_evidence_summary="【已校验证据】\n- Goal g1: supported\n  Sources: gpt3.pdf p7",
        )

        self.assertIn("【已校验证据】", prompt)
        self.assertIn("gpt3.pdf p7", prompt)
```

- [ ] **步骤 3：运行测试验证失败**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config_settings tests.test_agentic_generation_prompt
```

预期：FAIL，配置字段或 `verified_evidence_summary` 参数不存在。

- [ ] **步骤 4：实现配置字段**

在 `paper_rag/config/settings.py` 的 `RagSettings` 添加字段：

```python
enable_agentic_query: bool = False
agent_auto_for_complex: bool = True
agent_max_repair_rounds: int = 1
agent_planner_model: str = ""
agent_verifier_model: str = ""
agent_verifier_temperature: float = 0.0
agent_debug_trace: bool = False
```

在 `from_dict` 中读取字段，空模型默认使用 `llm_model`：

```python
agent_planner_model=str(data.get("agent_planner_model") or data.get("llm_model") or ""),
agent_verifier_model=str(data.get("agent_verifier_model") or data.get("llm_model") or ""),
```

在 `as_dict` 中输出这些字段。

在 `config.yaml` 添加默认配置，保持 agent 关闭：

```yaml
# ============ Agentic RAG（LangGraph，默认关闭）============
enable_agentic_query: false
agent_auto_for_complex: true
agent_max_repair_rounds: 1
agent_planner_model: qwen2.5:3b
agent_verifier_model: qwen2.5:3b
agent_verifier_temperature: 0.0
agent_debug_trace: false
```

- [ ] **步骤 5：实现生成 prompt 前置 summary**

修改 `paper_rag/generation/service.py` 的 `build_rag_prompt` 签名，增加可选参数：

```python
def build_rag_prompt(question: str, docs: list[Document], verified_evidence_summary: str = "") -> str:
```

在 context 前拼接：

```python
agentic_prefix = f"{verified_evidence_summary.strip()}\n\n" if verified_evidence_summary.strip() else ""
```

最终 prompt 中把 `agentic_prefix` 放在普通上下文片段前。同步更新根目录薄壳 `generation_service.py` 的 re-export 不应需要改动；如果根目录文件包含真实实现，也同步签名。

- [ ] **步骤 6：运行测试验证通过**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config_settings tests.test_agentic_generation_prompt
```

预期：PASS。

- [ ] **步骤 7：Commit**

```powershell
git add config.yaml paper_rag/config/settings.py paper_rag/generation/service.py generation_service.py tests/test_config_settings.py tests/test_agentic_generation_prompt.py
git commit -m "feat: add agentic config and prompt context"
```

## 任务 8：接入 rag_pipeline 非流式路径与日志

**文件：**
- 修改：`rag_pipeline.py`
- 修改：`paper_rag/observability/query_logger.py`
- 修改：`paper_rag/observability/service.py`
- 修改：`query_logger.py`
- 创建：`tests/test_agentic_rag_pipeline.py`

- [ ] **步骤 1：编写失败的 pipeline 测试**

创建 `tests/test_agentic_rag_pipeline.py`。

```python
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

import rag_pipeline


class AgenticRagPipelineTest(unittest.TestCase):
    def test_ask_with_context_uses_agentic_docs_when_enabled(self):
        docs = [Document(page_content="verified", metadata={"source": "gpt3.pdf", "page": 7})]
        agent_state = {
            "final_docs": docs,
            "route": "agentic_page_evidence",
            "verified_summary": "【已校验证据】\n- Goal g1: supported",
            "agent_trace": {"enabled": True, "plan": [{"id": "g1"}], "verification": [], "repair_rounds": 0},
        }

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "query_runs.jsonl"
            with patch.object(rag_pipeline, "build_hybrid_retriever", return_value=object()), \
                patch.object(rag_pipeline, "run_agentic_rag", return_value=agent_state), \
                patch.object(rag_pipeline, "_generate_answer", return_value="answer"), \
                patch.dict(
                    rag_pipeline.config,
                    {
                        "enable_agentic_query": True,
                        "enable_query_logging": True,
                        "query_log_path": str(log_path),
                    },
                ):
                answer, sources = rag_pipeline.ask_with_context("GPT-3 使用 Transformer 结构的证据在哪一页？")

            self.assertEqual(answer, "answer")
            self.assertEqual(sources[0].metadata["page"], 7)
            self.assertIn("agent_trace", log_path.read_text(encoding="utf-8"))
```

- [ ] **步骤 2：运行测试验证失败**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_rag_pipeline
```

预期：FAIL，`rag_pipeline` 未导入或未调用 `run_agentic_rag`。

- [ ] **步骤 3：扩展 query log 字段**

在 `paper_rag/observability/query_logger.py` 的 `build_query_log_record` 增加可选参数：

```python
agent_trace: dict[str, Any] | None = None,
```

输出记录增加：

```python
"agent_trace": agent_trace or {},
```

在 feature flags 中加入：

```python
"agentic_query": bool(feature_flags.get("agentic_query", False)),
```

同步更新 `paper_rag/observability/service.py` 和根目录 `query_logger.py` 薄壳或调用点，使参数可传递。

- [ ] **步骤 4：接入非流式 agent 分支**

在 `rag_pipeline.py` 导入：

```python
from paper_rag.agentic.graph import run_agentic_rag
from paper_rag.retrieval.router import is_comparison_question, is_evidence_question
```

增加复杂题判断：

```python
def _should_use_agentic(question: str, settings: Any, force_agent: bool | None = None) -> bool:
    if force_agent is not None:
        return force_agent
    if not get_setting(settings, "enable_agentic_query", False):
        return False
    if not get_setting(settings, "agent_auto_for_complex", True):
        return True
    return is_comparison_question(question) or is_evidence_question(question) or bool(getattr(settings, "conversation_history", None))
```

在 `ask_with_context` 中，普通 `_route_retrieve` 前判断 agent：

```python
agent_trace = {}
verified_summary = ""
if _should_use_agentic(standalone_q, current_settings):
    router = RetrievalRouter(current_settings, ...)
    source_hints = mentioned_source_files(standalone_q, hybrid, current_settings)
    agent_state = run_agentic_rag(
        question=question,
        standalone_question=standalone_q,
        task_type=_classify_agentic_task(standalone_q),
        source_hints=source_hints,
        hybrid=hybrid,
        router=router,
        planner_llm=_create_llm(current_settings.agent_planner_model, temperature),
        verifier_llm=_create_llm(current_settings.agent_verifier_model, current_settings.agent_verifier_temperature),
        llm_model=llm_model,
        temperature=temperature,
        max_repair_rounds=current_settings.agent_max_repair_rounds,
    )
    docs = agent_state.get("final_docs", [])
    route = agent_state.get("route", "agentic_mixed")
    agent_trace = agent_state.get("agent_trace", {})
    verified_summary = agent_state.get("verified_summary", "")
else:
    docs, route = _route_retrieve(...)
```

调用 `_generate_answer` 时传递 `verified_evidence_summary=verified_summary`；如果 `_generate_answer` 当前不支持该参数，扩展它的签名并传给 generation service。

写日志时传 `agent_trace=agent_trace`，feature flag 包含 `agentic_query`。

- [ ] **步骤 5：运行 pipeline 测试**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_rag_pipeline tests.test_query_logger
```

预期：PASS。

- [ ] **步骤 6：Commit**

```powershell
git add rag_pipeline.py paper_rag/observability/query_logger.py paper_rag/observability/service.py query_logger.py tests/test_agentic_rag_pipeline.py tests/test_query_logger.py
git commit -m "feat: wire agentic rag into query pipeline"
```

## 任务 9：Benchmark、CLI、Streamlit 与流式状态接入

**文件：**
- 修改：`benchmarks/run_baseline.py`
- 修改：`main.py`
- 修改：`query.py`
- 修改：`app.py`
- 修改：`paper_rag/ui/services.py`
- 修改：`rag_pipeline.py`
- 创建：`tests/test_agentic_cli_benchmark.py`
- 创建：`tests/test_agentic_streaming.py`

- [ ] **步骤 1：编写 benchmark/CLI 测试**

创建 `tests/test_agentic_cli_benchmark.py`，用 fake argv 验证 `--agent` 传入 baseline 配置或调用参数。

```python
import unittest
from unittest.mock import patch

from benchmarks.run_baseline import parse_args


class AgenticCliBenchmarkTest(unittest.TestCase):
    def test_run_baseline_accepts_agent_flags(self):
        args = parse_args(["--agent", "--output", "benchmarks/agentic_results_qwen2.5_3b.jsonl"])

        self.assertTrue(args.agent)
        self.assertEqual(args.output, "benchmarks/agentic_results_qwen2.5_3b.jsonl")

    def test_run_baseline_accepts_no_agent_flag(self):
        args = parse_args(["--no-agent"])

        self.assertFalse(args.agent)
```

- [ ] **步骤 2：编写流式事件测试**

创建 `tests/test_agentic_streaming.py`。

```python
import unittest
from unittest.mock import patch

import rag_pipeline


class AgenticStreamingTest(unittest.TestCase):
    def test_ask_stream_emits_agent_status_before_tokens(self):
        with patch.dict(rag_pipeline.config, {"enable_agentic_query": True}), \
            patch.object(rag_pipeline, "_should_use_agentic", return_value=True), \
            patch.object(rag_pipeline, "_run_agentic_retrieval_for_stream", return_value=({"final_docs": [], "agent_trace": {"enabled": True}, "route": "agentic_mixed", "verified_summary": ""})), \
            patch.object(rag_pipeline, "stream_answer_tokens", return_value=iter(["hello"])):
            events = list(rag_pipeline.ask_stream("复杂问题"))

        self.assertEqual(events[0]["type"], "agent_status")
        self.assertEqual(events[-1]["type"], "token")
```

- [ ] **步骤 3：运行测试验证失败**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_cli_benchmark tests.test_agentic_streaming
```

预期：FAIL，缺少参数或流式 helper。

- [ ] **步骤 4：实现 benchmark agent flags**

在 `benchmarks/run_baseline.py` 的 `parse_args` 添加：

```python
parser.set_defaults(agent=None)
parser.add_argument("--agent", dest="agent", action="store_true")
parser.add_argument("--no-agent", dest="agent", action="store_false")
```

当 `args.agent is True` 且用户未显式传 `--output` 时，默认输出到：

```text
benchmarks/agentic_results_qwen2.5_3b.jsonl
```

调用 `rag_pipeline.ask_with_context` 时传递 `force_agent=args.agent`。如果当前函数签名不支持，则在任务 8 的基础上增加 `force_agent: bool | None = None`。

- [ ] **步骤 5：实现 CLI 和 UI 开关**

在 `main.py` / `query.py` 增加 `--agent` / `--no-agent` 参数，并传入问答调用。

在 `app.py` 侧边栏增加开关：

```python
agent_enabled = st.sidebar.toggle("Agentic 查询", value=False)
```

Streamlit 调用 `ask_stream` 时传 `force_agent=agent_enabled`。

- [ ] **步骤 6：实现流式 agent status**

在 `rag_pipeline.ask_stream` 中 agent 分支检索前后 yield：

```python
yield {"type": "agent_status", "data": "正在拆分证据目标..."}
agent_state = _run_agentic_retrieval_for_stream(...)
yield {"type": "agent_status", "data": "正在校验证据并组装上下文..."}
if current_settings.agent_debug_trace:
    yield {"type": "agent_trace", "data": agent_state.get("agent_trace", {})}
```

最终继续 yield 现有 `route`、`token`、`sources` 事件。

- [ ] **步骤 7：运行测试验证通过**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agentic_cli_benchmark tests.test_agentic_streaming tests.test_app_services
```

预期：PASS。

- [ ] **步骤 8：Commit**

```powershell
git add benchmarks/run_baseline.py main.py query.py app.py paper_rag/ui/services.py rag_pipeline.py tests/test_agentic_cli_benchmark.py tests/test_agentic_streaming.py
git commit -m "feat: expose agentic rag controls"
```

## 任务 10：全量回归与 agentic benchmark 验收

**文件：**
- 可能修改：`eval/run_eval.py`
- 可能修改：`benchmarks/README.md`
- 可能修改：`README.md`

- [ ] **步骤 1：运行单元测试全集**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

预期：PASS。若失败，先定位是 agentic 新测试还是旧功能回归；不要调整 benchmark 标注来掩盖实现问题。

- [ ] **步骤 2：运行 benchmark schema 检查**

```powershell
.\.venv\Scripts\python.exe benchmarks\run_baseline.py --no-generate
```

预期：PASS，新增 figure 样本格式有效。

- [ ] **步骤 3：运行 agentic benchmark**

```powershell
.\.venv\Scripts\python.exe benchmarks\run_baseline.py --agent --output benchmarks\agentic_results_qwen2.5_3b.jsonl
```

预期：生成 `benchmarks/agentic_results_qwen2.5_3b.jsonl`，每条 agent 复杂题包含 `agent_trace`。

- [ ] **步骤 4：生成 agentic eval 报告**

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py --input benchmarks\agentic_results_qwen2.5_3b.jsonl --label agentic_qwen2_5_3b
```

预期：生成 `eval/reports/report_agentic_qwen2_5_3b.json`，报告包含 `agent` 节。

- [ ] **步骤 5：检查关键验收样本**

用 PowerShell 过滤关键样本：

```powershell
Get-Content benchmarks\agentic_results_qwen2.5_3b.jsonl |
  Select-String -Pattern '"id": "q022"|"id":"q022"|"id": "q023"|"id":"q023"|"id": "q026"|"id":"q026"|"id": "q027"|"id":"q027"'
```

验收条件：

| 样本 | 预期 |
|---|---|
| `q022` | `retrieved_sources` 或 `agent_trace.verified_sources` 包含 `gpt3.pdf` page 7。 |
| `q023` | 答案覆盖 BERT 的 MLM + NSP，以及 T5 的 text-to-text/systematic comparison。 |
| `q026` | 能定位 `deepseekr1.pdf` page 51。 |
| `q027` | 能解释 Figure 14 的多语言安全表现比较，不编造精确数值。 |

- [ ] **步骤 6：更新文档**

在 `README.md` 或 `benchmarks/README.md` 添加 agentic 命令：

```powershell
.\.venv\Scripts\python.exe benchmarks\run_baseline.py --agent --output benchmarks\agentic_results_qwen2.5_3b.jsonl
.\.venv\Scripts\python.exe eval\run_eval.py --input benchmarks\agentic_results_qwen2.5_3b.jsonl --label agentic_qwen2_5_3b
```

说明 agent 默认关闭，可通过配置、CLI、UI、benchmark flag 启用。

- [ ] **步骤 7：最终 commit**

```powershell
git add README.md benchmarks/README.md eval/run_eval.py benchmarks/agentic_results_qwen2.5_3b.jsonl eval/reports/report_agentic_qwen2_5_3b.json
git commit -m "docs: document agentic rag benchmark workflow"
```

如果 benchmark 结果文件或 eval 报告按项目约定不应提交，则只提交 README 类文档和代码变更，并在最终交付说明中列出本地生成文件路径。

## 自检清单

| 规格要求 | 覆盖任务 |
|---|---|
| 复杂题 agentic workflow | 任务 6、8、9 |
| LangGraph | 任务 6 |
| TypedDict state | 任务 2、6 |
| Planner 规则骨架 + LLM 补全 | 任务 3 |
| Verifier 关键词预筛 + LLM 判断 | 任务 4 |
| Figure 分支 | 任务 1、5、6 |
| 最多 1 轮补查 | 任务 6 |
| 只读知识库、只写日志 | 任务 6、8 |
| `agent_trace` 日志 | 任务 1、8 |
| 独立 agentic benchmark 输出 | 任务 9、10 |
| 流式 agent 状态事件 | 任务 9 |
| CLI/UI 开关 | 任务 9 |
| 分层测试 | 任务 1-10 |

## 执行交接

计划已完成后，两种执行方式：

| 方式 | 说明 |
|---|---|
| 子代理驱动（推荐） | 每个任务调度一个新的子代理，任务间进行审查，快速迭代。 |
| 内联执行 | 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点。 |
