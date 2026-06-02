# Agentic RAG LangGraph 设计规格

## 背景

当前 Paper RAG 已具备混合检索、HyDE、Query Expansion、Rerank、Parent Retrieval、Context Compression、结构化分块、视觉摘要、查询日志和离线评估。已有评估显示，主 benchmark 与 holdout 的检索召回整体较高，主要失败集中在答案不完整、证据页定位不足、跨论文归因错误，以及图表证据使用不稳定。

因此第一版 agent 化目标不是让系统变成完全自主研究助手，而是把复杂论文问答改造成可控的证据驱动流程：先规划证据目标，再按目标检索和校验，缺证据时补查一次，最后把已校验证据注入生成上下文。

## 目标与非目标

| 类别 | 内容 |
|---|---|
| 目标 | 提升 `evidence`、`compare`、`followup`、`figure` 复杂题的答案完整度、证据可靠性和归因准确性。 |
| 目标 | 引入 LangGraph 表达 bounded workflow，并保留现有 RAG 路径用于 A/B 对照。 |
| 目标 | 新增 agent trace、agent 指标和独立 agentic benchmark 输出。 |
| 目标 | 新增适量 figure benchmark，覆盖图表定位、图表解释和图文联合证据。 |
| 非目标 | 不自动修改知识库、索引、配置或 prompt。 |
| 非目标 | 不在查询阶段动态调用视觉模型生成图表摘要。 |
| 非目标 | 不做无限自主循环，不做开放式多 agent 角色协作。 |
| 非目标 | 第一阶段不替换全部 `rag_pipeline.py` 主链路。 |

## 总体结论

| 决策点 | 规格 |
|---|---|
| Agent 类型 | 证据驱动的 Agentic RAG，而非通用自主 Agent。 |
| 框架 | 引入 LangGraph。 |
| 主入口 | 第一阶段保留 `rag_pipeline.py` 为兼容主入口，LangGraph 作为可选 agentic 分支。 |
| 启用范围 | 默认关闭；开启后复杂题自动启用，简单题仍走普通 RAG。 |
| 补查轮数 | 最多 1 轮。 |
| 权限边界 | 只读知识库，只写查询日志；失败样本回流必须人工触发。 |
| 最终生成 | Agent 可重排、过滤、标注上下文，但不直接绕开现有 generation service。 |

## 启用策略

| 入口 | 规格 |
|---|---|
| 配置 | 新增 `enable_agentic_query`，默认 `false`。 |
| 配置 | 新增 `agent_auto_for_complex`，开启 agent 后复杂题自动走 agent。 |
| CLI | 支持 `--agent` 和 `--no-agent` 强制当前查询启用或关闭。 |
| Streamlit | 侧边栏提供 Agentic 查询开关。 |
| Benchmark | `benchmarks/run_baseline.py` 支持 `--agent`，输出独立结果文件。 |
| 日志 | `feature_flags.agentic_query` 标明是否启用。 |

建议初始配置：

```yaml
enable_agentic_query: false
agent_auto_for_complex: true
agent_max_repair_rounds: 1
agent_planner_model: qwen2.5:3b
agent_verifier_model: qwen2.5:3b
agent_verifier_temperature: 0.0
agent_debug_trace: false
```

Planner 和 Verifier 使用独立配置项，但默认复用当前 `llm_model`，避免第一版增加额外模型依赖。

## LangGraph 工作流

```text
START
  -> route_question
  -> plan_goals
  -> collect_evidence
  -> verify_evidence
  -> should_repair?
      -> repair_queries
      -> collect_evidence
      -> verify_evidence
  -> assemble_context
  -> generate_answer
  -> END
```

| 节点 | 职责 |
|---|---|
| `route_question` | 判断是否需要 agent，识别 `evidence`、`compare`、`followup`、`figure` 等复杂题。 |
| `plan_goals` | 使用规则骨架 + LLM 补全，生成 1-4 个结构化 evidence goals。 |
| `collect_evidence` | 按 goal 类型调用现有 mixed、rerank、anchor、vision summary 或受控 HyDE。第一版串行检索。 |
| `verify_evidence` | 关键词预筛 + LLM 支撑判断，输出 `supported`、`partial`、`unsupported`。LLM 失败时规则 fallback。 |
| `repair_queries` | 只针对缺失或 unsupported goal 补查 1 轮。 |
| `assemble_context` | 重排、过滤、标注最终上下文，把 verified evidence 放在 prompt 前部。 |
| `generate_answer` | 复用现有 generation service，但 prompt 中加入 verified evidence summary 与回答约束。 |

## 状态 Schema

LangGraph state 使用 `TypedDict`，避免全程裸 dict。`EvidenceGoal` 与 `VerifiedEvidence` 可用 `TypedDict` 或 dataclass，但日志输出必须容易 JSON 序列化。

```python
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
```

## Planner 设计

| 决策 | 规格 |
|---|---|
| 规划方式 | 规则生成骨架 + LLM 补全 goals。 |
| 输出规模 | 每题 1-4 个 evidence goals，超过 4 个截断。 |
| 结构化解析 | JSON prompt + 容错提取 JSON 块 + schema 校验 + 规则 fallback。 |
| 来源修正 | `source_hint` 必须经 SourceResolver 校验；无效来源清空或修正。 |
| 多轮追问 | 第一版保留现有 standalone question 改写，Planner 使用改写后的问题，不接管 history 消解。 |

目标结构示例：

```json
{
  "id": "g2",
  "goal_type": "page_evidence",
  "claim": "GPT-3 使用与 GPT-2 相同的模型和架构，并在 Transformer 层中使用 attention patterns",
  "query": "GPT-3 same model and architecture as GPT-2 attention patterns transformer layers",
  "source_hint": "gpt3.pdf",
  "page_hint": 7
}
```

## 检索策略

| Goal 类型 | 默认策略 |
|---|---|
| `page_evidence` | mixed + source anchor，禁用 HyDE。 |
| `compare_dimension` | mixed + compare anchors，禁用 HyDE。 |
| `method_overview` | 可使用 HyDE。 |
| `figure_evidence` | vision summary + source anchor + 同页/相邻页文本，禁用 HyDE。 |
| repair fallback | 只允许 `method_overview` 使用 HyDE；证据页、图表、对比目标仍禁用。 |

第一版多个 goals 串行检索。后续若 trace 和评估稳定，可把 `collect_for_goal` 改造成并行 map 或线程池。

## Verifier 设计

| 决策 | 规格 |
|---|---|
| 校验方式 | 关键词/别名/页码/section metadata 预筛，然后 LLM 判断支撑关系。 |
| 状态集合 | 只允许 `supported`、`partial`、`unsupported`。 |
| Fallback | Verifier LLM 不可用或结构化输出失败时，退回规则校验。 |
| 输出用途 | 影响最终上下文排序、过滤、prompt 约束和 `agent_trace`。 |

输出结构示例：

```json
{
  "goal_id": "g2",
  "claim": "GPT-3 使用 Transformer 架构",
  "status": "supported",
  "supporting_sources": [
    {
      "file": "gpt3.pdf",
      "page": 7,
      "reason": "mentions same model and architecture as GPT-2 and attention patterns in transformer layers"
    }
  ],
  "missing_terms": []
}
```

## Figure 分支

新增 figure benchmark 后，Agentic RAG 必须支持图表定位、图表解释和图文联合证据。

| 决策 | 规格 |
|---|---|
| Vision 优先级 | 优先使用 `chunk_strategy=vision_summary` 或 `paper_region=vision`。 |
| 回退策略 | 没有 vision summary 时，只回退同页文本、图表标题和相邻正文页。 |
| 禁止行为 | 不在查询阶段调用视觉模型；不自动重建索引。 |
| Trace 字段 | 记录 `figure_evidence_mode=vision_summary|text_fallback|missing` 和 `vision_summary_used`。 |
| 第一版 benchmark | 新增 4-6 条，优先围绕 DeepSeek-R1 Figure 14。 |

第一版不测试精确数值读取，因为当前 vision summary 未必稳定支持图中细粒度数值。

## 上下文组装与生成约束

Agent 可以重排、过滤、标注最终上下文，但不直接绕开现有答案生成服务。

| 题型 | 上下文策略 |
|---|---|
| `evidence` | verified / partial 证据放前；明显无关来源丢弃。 |
| `figure` | vision summary + 同页文本优先；其他页降权或丢弃。 |
| `compare` | 每个被比较对象至少保留 1-2 条证据；未验证但同源首页/方法页可保留。 |
| `followup` | 保留历史消解后的目标来源；无关论文丢弃。 |
| 普通题 | 不走 agent，不改上下文。 |

最终 prompt 前置结构：

```text
【已校验证据】
- Goal g1: supported
  Claim: GPT-3 是自回归语言模型
  Sources: gpt3.pdf p0
- Goal g2: partial
  Claim: GPT-3 使用 Transformer 架构
  Sources: gpt3.pdf p7

【回答约束】
- 优先使用 supported 证据。
- partial 证据只能谨慎表述。
- unsupported goal 必须说明“未找到足够证据”。
- 不要把 A 论文的结论归因给 B 论文。

【原始上下文片段】
...
```

## 日志与可观测性

保持现有 `retrieved_sources` 兼容语义：表示最终送入生成上下文的来源。新增 `agent_trace` 记录 agent 过程，不直接改变旧字段含义。

| 字段 | 含义 |
|---|---|
| `feature_flags.agentic_query` | 是否启用 agent。 |
| `route` | 如 `agentic_mixed`、`agentic_compare`、`agentic_figure`。 |
| `agent_trace.plan` | evidence goals。 |
| `agent_trace.verification` | 每个 goal 的支撑状态。 |
| `agent_trace.repair_rounds` | 实际补查次数。 |
| `agent_trace.fallback_reason` | 回退原因。 |
| `agent_trace.verified_sources` | 被 verifier 支持的来源。 |
| `agent_trace.figure_evidence_mode` | figure 分支使用视觉、文本回退或缺失。 |

UI 默认只展示简短状态，不展示完整 trace。Debug 模式或日志中保留完整 trace。

流式路径新增事件：

```python
{"type": "agent_status", "data": "正在拆分证据目标..."}
{"type": "agent_status", "data": "正在校验 3 条证据..."}
{"type": "agent_trace", "data": {...}}
{"type": "route", "data": "agentic_mixed"}
{"type": "token", "data": "..."}
{"type": "sources", "data": [...]}
```

`agent_trace` 只在 debug 模式传给 UI。

## 失败与回退策略

| 失败点 | 处理方式 |
|---|---|
| Planner LLM 输出非法 JSON | 使用规则 planner 兜底。 |
| Planner 无法生成 goal | 回退普通 RAG，并记录 `agent_fallback_reason=planner_failed`。 |
| Collector 某个 goal 无结果 | 触发最多 1 轮补查。 |
| 补查后仍无结果 | 保留该 goal 为 `unsupported`，答案说明证据不足。 |
| Verifier LLM 失败 | 使用关键词/来源规则 fallback。 |
| 整个 workflow 异常 | 回退普通 RAG，并在日志记录 agent 错误。 |

用户不应因为 agent 模块失败而没有答案；日志必须能看出 agent 是否实际生效。

## Benchmark 与评估

Agentic benchmark 单独输出结果和报告，不覆盖现有 baseline、hybrid、holdout 文件。

| 文件 | 用途 |
|---|---|
| `benchmarks/agentic_results_qwen2.5_3b.jsonl` | 主 benchmark 的 agentic 结果。 |
| `benchmarks/agentic_holdout_results_qwen2.5_3b.jsonl` | holdout 的 agentic 结果。 |
| `eval/reports/report_agentic_qwen2_5_3b.json` | 主 benchmark agentic 报告。 |
| `eval/reports/report_agentic_holdout_qwen2_5_3b.json` | holdout agentic 报告。 |

新增 agent 指标：

| 指标 | 含义 |
|---|---|
| `agent_enabled_count` | 有多少样本实际走 agent。 |
| `avg_evidence_goal_count` | 平均拆了几个证据目标。 |
| `goal_support_rate` | evidence goal 被支持的比例。 |
| `repair_trigger_rate` | 触发补查比例。 |
| `repair_success_rate` | 补查后从 unsupported/partial 变 supported 的比例。 |
| `avg_agent_elapsed_sec` | agent 增加的耗时。 |
| `answer_completeness_delta` | 相比非 agent 的答案完整度变化。 |
| `retrieval_recall_delta` | 相比非 agent 的召回变化。 |

第一版验收样本：

| 样本 | 验收要求 |
|---|---|
| `q022` | 必须召回 `gpt3.pdf page 7`，并回答页码。 |
| `q023` | 必须覆盖 BERT 的 MLM + NSP，以及 T5 的 text-to-text/systematic comparison。 |
| `h014` | 不允许把 BERT 说成 text-to-text 框架。 |
| `h015` | 必须引用 GPT-3、T5、Attention 至少两个来源，最好三个来源。 |
| figure 新样本 | 覆盖 DeepSeek-R1 Figure 14 的图表定位、解释和图文联合证据。 |

上线门槛：

| 指标 | 门槛 |
|---|---|
| 复杂题答案完整度 | 至少提升 20%。 |
| evidence/compare/followup Recall@5 | 不下降。 |
| 平均耗时 | 不超过非 agent 的 2 倍。 |
| repair success rate | 大于 30%。 |

## 工程模块

新增 `paper_rag/agentic/` 包。

| 模块 | 职责 |
|---|---|
| `paper_rag/agentic/schema.py` | TypedDict、状态结构、日志序列化辅助。 |
| `paper_rag/agentic/planner.py` | 规则骨架、LLM 补全、JSON 容错解析。 |
| `paper_rag/agentic/collector.py` | 按 goal 类型调用现有检索能力。 |
| `paper_rag/agentic/verifier.py` | 证据支撑判断和 fallback。 |
| `paper_rag/agentic/context.py` | verified evidence summary、上下文重排和过滤。 |
| `paper_rag/agentic/graph.py` | LangGraph 节点和 workflow 编排。 |

Prompt 文件放在顶层 `prompts/`，复用现有 `utils.prompt_loader.load_prompt`。

| Prompt | 用途 |
|---|---|
| `prompts/agent_planner_prompt.txt` | Planner LLM 补全 evidence goals。 |
| `prompts/agent_verifier_prompt.txt` | Verifier 判断支撑关系。 |
| `prompts/agent_context_prompt.txt` | 可选，用于生成阶段的上下文约束说明。 |

## 依赖管理

第一版只在 `requirements.txt` 增加 LangGraph 依赖，不迁移依赖管理体系。

```text
langgraph>=0.2
```

如果实现时发现与当前 LangChain 版本存在兼容要求，再收紧版本范围。

## 测试策略

| 测试文件 | 覆盖范围 |
|---|---|
| `tests/test_agentic_planner.py` | JSON 解析、规则 fallback、goal 截断、source hint 修正。 |
| `tests/test_agentic_verifier.py` | `supported`、`partial`、`unsupported` 解析与关键词 fallback。 |
| `tests/test_agentic_collector.py` | goal 类型到检索策略的映射、HyDE 禁用规则、figure fallback。 |
| `tests/test_agentic_graph.py` | LangGraph 状态流转、repair 一轮限制、assemble context。 |
| `tests/test_agentic_rag_pipeline.py` | agent 开关、route、日志 `agent_trace`。 |
| `tests/test_agentic_eval_metrics.py` | agent 指标计算和报告输出。 |

原则：

| 原则 | 原因 |
|---|---|
| 单元测试使用 fake LLM response。 | 避免模型输出波动。 |
| Retriever 使用 fake hybrid/vector store。 | 避免依赖真实 Chroma。 |
| Graph 测状态，不测模型聪明程度。 | 流程必须可预测。 |
| 真实效果只通过 agentic benchmark 验证。 | 准确率提升不能靠单元测试证明。 |

## 分阶段实施

| 阶段 | 产物 |
|---|---|
| 1 | 新增 4-6 条 figure benchmark，并标记 q022/q023/h014/h015 为 agentic 验收样本。 |
| 2 | 扩展 eval，支持读取 `agent_trace` 并输出 agent 指标。 |
| 3 | 新增 agentic schema、planner、verifier、collector 单元测试和实现。 |
| 4 | 实现 LangGraph workflow，覆盖 plan -> collect -> verify -> repair -> assemble。 |
| 5 | 接入 `rag_pipeline.py` 的非流式路径和 benchmark `--agent`。 |
| 6 | 接入流式路径，输出 `agent_status` 事件。 |
| 7 | 接入 CLI 与 Streamlit 开关。 |
| 8 | 运行 agentic benchmark，与 baseline/hybrid/holdout 对照。 |

## 用户追问策略

Agent 只有来源或对象歧义时才追问。Benchmark 和 eval 模式永不追问。

| 场景 | 行为 |
|---|---|
| “这篇论文”但历史里没有可解析来源 | 可以追问。 |
| “两个模型比较一下”但未提模型名且历史无上下文 | 可以追问。 |
| “Figure 3 怎么样”但多个来源都有 Figure 3，且无 source hint | 可以追问。 |
| 只是证据不足 | 不追问，补查 1 轮后说明不足。 |
| benchmark / eval | 不追问，按自动流程输出。 |

## 已确认设计决策

| 编号 | 决策 |
|---|---|
| D01 | Agent 化第一目标是提升复杂论文问答准确率和证据可靠性。 |
| D02 | 第一版只让复杂题默认走 agent，并提供手动强制开关。 |
| D03 | 最多允许 1 轮补查，仍缺证据则明说。 |
| D04 | Verifier 使用关键词预筛 + LLM 支撑判断，并保留规则 fallback。 |
| D05 | Planner 输出 1-4 个结构化 evidence goals。 |
| D06 | 默认展示简短状态，完整 trace 只写入日志或 debug 模式。 |
| D07 | Agent 内按 goal 类型选择检索策略，证据页/图表/对比目标禁用 HyDE。 |
| D08 | 先离线评估 agent，再接入 UI/CLI。 |
| D09 | 第一版 agent 只读知识库，只写日志，不自动改索引或配置。 |
| D10 | Planner/Verifier 使用独立配置项，但默认复用当前 `llm_model`。 |
| D11 | 新建 `paper_rag/agentic/` 包承载 agent workflow。 |
| D12 | Agent 可重排、过滤、标注最终上下文，但不直接生成最终答案。 |
| D13 | Agent 可以丢弃明显无关片段，但不能只保留 verified evidence。 |
| D14 | Planner 采用规则骨架 + LLM 补全。 |
| D15 | 第一版验收聚焦 q022/q023/h014/h015 和 figure 样本。 |
| D16 | Figure benchmark 新增 4-6 条，优先围绕 DeepSeek-R1 Figure 14。 |
| D17 | Figure agent 优先 vision summary，没有时只回退文本/相邻页。 |
| D18 | Agent 只有来源/对象歧义时才追问；benchmark/eval 永不追问。 |
| D19 | Agent 采用分层回退，并写明 fallback reason。 |
| D20 | 配置默认关闭 agent；开启后复杂题自动走 agent。 |
| D21 | 流式路径输出简短 agent 状态事件，最终答案继续 token 流式。 |
| D22 | 保持 `retrieved_sources` 兼容语义，新增 `agent_trace`。 |
| D23 | Agentic benchmark 单独输出结果和报告。 |
| D24 | 引入 LangGraph。 |
| D25 | 第一阶段保留 `rag_pipeline.py` 作为兼容主入口。 |
| D26 | LangGraph state 使用 `TypedDict` 强类型 schema。 |
| D27 | 结构化输出使用 JSON prompt + 容错解析 + schema 校验 + fallback。 |
| D28 | 第一版多个 evidence goals 串行检索。 |
| D29 | 第一版保留现有多轮改写，agent 使用 standalone question。 |
| D30 | Agent planner/verifier prompt 独立放到顶层 `prompts/`。 |
| D31 | 最终 prompt 前置 verified evidence summary 和回答约束。 |
| D32 | UI 来源展示第一版不大改，goal 分组来源放 debug/log/eval。 |
| D33 | 第一版只在 `requirements.txt` 增加 LangGraph，不迁移依赖管理。 |
| D34 | 单元测试全部 fake LLM/retriever，真实效果通过 agentic benchmark 验证。 |
| D35 | 先补 benchmark 和评估指标，再实现 LangGraph 主流程。 |
