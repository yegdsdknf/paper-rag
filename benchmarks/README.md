# 人工标注基准集

本目录用于保存 RAG 系统的人工标注基准样本。基准集的目标是先定义「什么叫答得对」，再用离线评估脚本比较不同检索和生成策略。

## 当前文件

| 文件 | 说明 |
|---|---|
| `benchmark_v1.jsonl` | 任务文档约定的基准集入口文件，覆盖 25 条样本。 |
| `run_baseline.py` | 基线跑分脚本，读取基准集并输出当前 RAG 系统的回答结果。 |
| `baseline_results.jsonl` | 基线跑分输出文件，由 `run_baseline.py` 生成。 |
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
| `task_type` | 任务类型，包括 `definition`、`compare`、`method`、`experiment`、`detail`、`summary`、`followup`、`evidence`。 |
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

## 离线自动评估

任务 3 的评估脚本只读取已有 baseline 结果，不重新调用模型，避免把模型输出波动混入评估脚本验证。

```powershell
python eval/run_eval.py --input benchmarks/baseline_results_qwen2.5_3b.jsonl --label qwen2_5_3b
python eval/run_eval.py --input benchmarks/baseline_results_deepseek_r1_7b.jsonl --label deepseek_r1_7b
```

报告会写入 `eval/reports/`，包含样本数、平均 Recall@5、平均 MRR、来源命中统计、低召回样本清单，以及按题型和难度聚合的结果。
