from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # 同时支持 `python -m eval.run_eval` 和直接运行 `python eval/run_eval.py`。
    sys.path.insert(0, str(ROOT))

from eval.metrics import mrr, recall_at_k, source_hit_status


DEFAULT_INPUT_PATH = Path("benchmarks") / "baseline_results_qwen2.5_3b.jsonl"
DEFAULT_OUTPUT_DIR = Path("eval") / "reports"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _group_summary(items: list[dict[str, Any]], k: int) -> dict[str, Any]:
    recall_key = f"recall_at_{k}"
    hit_counts = Counter(item["source_hit"] for item in items)
    return {
        "sample_count": len(items),
        f"avg_{recall_key}": _avg([item[recall_key] for item in items]),
        "avg_mrr": _avg([item["mrr"] for item in items]),
        "source_hit_counts": {
            "full": hit_counts.get("full", 0),
            "partial": hit_counts.get("partial", 0),
            "missing": hit_counts.get("missing", 0),
        },
    }


def evaluate_rows(rows: list[dict[str, Any]], label: str, k: int = 5) -> dict[str, Any]:
    recall_key = f"recall_at_{k}"
    evaluated: list[dict[str, Any]] = []

    for row in rows:
        # 当前只评估检索层；生成质量暂时保留在人工报告中判断。
        retrieved = row.get("retrieved_sources", []) or []
        gold = row.get("gold_sources", []) or []
        item = {
            "id": row.get("id"),
            "task_type": row.get("task_type", "unknown"),
            "difficulty": row.get("difficulty", "unknown"),
            recall_key: recall_at_k(retrieved, gold, k=k),
            "mrr": mrr(retrieved, gold),
            "source_hit": source_hit_status(retrieved, gold, k=k),
            "elapsed_sec": row.get("elapsed_sec", 0.0) or 0.0,
            "error": row.get("error"),
            "skipped": bool(row.get("skipped", False)),
        }
        evaluated.append(item)

    hit_counts = Counter(item["source_hit"] for item in evaluated)
    by_task_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in evaluated:
        by_task_type[item["task_type"]].append(item)
        by_difficulty[item["difficulty"]].append(item)

    low_recall_samples = [
        {
            "id": item["id"],
            "task_type": item["task_type"],
            "difficulty": item["difficulty"],
            recall_key: item[recall_key],
            "mrr": item["mrr"],
            "source_hit": item["source_hit"],
        }
        for item in evaluated
        if item[recall_key] < 1.0
    ]

    return {
        "label": label,
        "sample_count": len(evaluated),
        "k": k,
        "averages": {
            recall_key: _avg([item[recall_key] for item in evaluated]),
            "mrr": _avg([item["mrr"] for item in evaluated]),
            "elapsed_sec": _avg([float(item["elapsed_sec"]) for item in evaluated]),
        },
        "source_hit_counts": {
            "full": hit_counts.get("full", 0),
            "partial": hit_counts.get("partial", 0),
            "missing": hit_counts.get("missing", 0),
        },
        "error_count": sum(1 for item in evaluated if item["error"]),
        "skipped_count": sum(1 for item in evaluated if item["skipped"]),
        "low_recall_samples": low_recall_samples,
        "by_task_type": {
            task_type: _group_summary(items, k)
            for task_type, items in sorted(by_task_type.items())
        },
        "by_difficulty": {
            difficulty: _group_summary(items, k)
            for difficulty, items in sorted(by_difficulty.items())
        },
        "samples": evaluated,
    }


def evaluate_results_file(
    input_path: Path,
    label: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    k: int = 5,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)
    report = evaluate_rows(rows, label=label, k=k)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"report_{label}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    report["output_path"] = str(output_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Paper RAG baseline result JSONL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Baseline result JSONL.")
    parser.add_argument("--label", required=True, help="Report label, e.g. qwen2_5_3b.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Report output directory.")
    parser.add_argument("--k", type=int, default=5, help="Recall@k cutoff.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_results_file(
        input_path=args.input,
        label=args.label,
        output_dir=args.output_dir,
        k=args.k,
    )
    recall_key = f"recall_at_{args.k}"
    print(f"样本数: {report['sample_count']}")
    print(f"平均 Recall@{args.k}: {report['averages'][recall_key]}")
    print(f"平均 MRR: {report['averages']['mrr']}")
    print(f"来源命中: {report['source_hit_counts']}")
    print(f"低召回样本数: {len(report['low_recall_samples'])}")
    print(f"输出文件: {report['output_path']}")


if __name__ == "__main__":
    main()
