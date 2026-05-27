# 人工标注基准集

本目录用于保存 RAG 系统的人工标注基准样本。基准集的目标是先定义「什么叫答得对」，再用离线评估脚本比较不同检索和生成策略。

## 当前文件

| 文件 | 说明 |
|---|---|
| `benchmark_v1.jsonl` | 任务文档约定的基准集入口文件，覆盖 25 条样本。 |
| `labels/manual_benchmark_v1.jsonl` | 第一版人工标注基准集源文件，覆盖 25 条样本。 |

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
