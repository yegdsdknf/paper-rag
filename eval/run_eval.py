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

from eval.metrics import answer_completeness, evidence_coverage, mrr, recall_at_k, source_hit_status


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


def _compression_ratio(context: dict[str, Any]) -> float:
    input_chars = float(context.get("input_chars") or 0.0)
    output_chars = float(context.get("output_chars") or 0.0)
    if input_chars <= 0:
        return 1.0
    return round(output_chars / input_chars, 4)


def _citation_accuracy(source_hit: str) -> float:
    return {"full": 1.0, "partial": 0.5, "missing": 0.0}.get(source_hit, 0.0)


def _error_bucket(error: str | None, skipped: bool, source_hit: str, completeness: float) -> str:
    if error:
        return "runtime_error"
    if skipped:
        return "skipped"
    if source_hit == "missing":
        return "retrieval_miss"
    if source_hit == "partial":
        return "partial_retrieval"
    if completeness < 1.0:
        return "answer_incomplete"
    return "ok"


def _layer_summary(items: list[dict[str, Any]], k: int) -> dict[str, Any]:
    recall_key = f"recall_at_{k}"
    hit_counts = Counter(item["retrieval"]["source_hit"] for item in items)
    bucket_counts = Counter(item["error_bucket"] for item in items)
    return {
        "retrieval": {
            f"avg_{recall_key}": _avg([item["retrieval"][recall_key] for item in items]),
            "avg_mrr": _avg([item["retrieval"]["mrr"] for item in items]),
            "source_hit_counts": {
                "full": hit_counts.get("full", 0),
                "partial": hit_counts.get("partial", 0),
                "missing": hit_counts.get("missing", 0),
            },
        },
        "context": {
            "avg_source_doc_count": _avg([item["context"]["source_doc_count"] for item in items]),
            "avg_context_doc_count": _avg([item["context"]["context_doc_count"] for item in items]),
            "avg_input_chars": _avg([item["context"]["input_chars"] for item in items]),
            "avg_output_chars": _avg([item["context"]["output_chars"] for item in items]),
            "avg_compression_ratio": _avg([item["context"]["compression_ratio"] for item in items]),
            "avg_parent_hit_count": _avg([item["context"]["parent_hit_count"] for item in items]),
        },
        "answer": {
            "avg_evidence_coverage": _avg([item["answer"]["evidence_coverage"] for item in items]),
            "avg_answer_completeness": _avg([item["answer"]["answer_completeness"] for item in items]),
            "avg_citation_accuracy": _avg([item["answer"]["citation_accuracy"] for item in items]),
            "avg_unsupported_claim_count": _avg([item["answer"]["unsupported_claim_count"] for item in items]),
        },
        "experience": {
            "avg_elapsed_sec": _avg([float(item["experience"]["elapsed_sec"]) for item in items]),
            "error_count": sum(1 for item in items if item["experience"]["error"]),
            "skipped_count": sum(1 for item in items if item["experience"]["skipped"]),
            "error_bucket_counts": dict(sorted(bucket_counts.items())),
        },
    }


def _agent_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    enabled_traces = [
        item["agent_trace"]
        for item in items
        if (item.get("agent_trace") or {}).get("enabled")
    ]
    verification_items = [
        verification
        for trace in enabled_traces
        for verification in (trace.get("verification") or [])
    ]
    repair_traces = [
        trace
        for trace in enabled_traces
        if int(trace.get("repair_rounds") or 0) > 0
    ]
    return {
        "enabled_count": len(enabled_traces),
        "avg_evidence_goal_count": _avg([
            float(len(trace.get("plan") or []))
            for trace in enabled_traces
        ]),
        "goal_support_rate": _avg([
            1.0 if verification.get("status") == "supported" else 0.0
            for verification in verification_items
        ]),
        "repair_trigger_rate": _avg([
            1.0 if int(trace.get("repair_rounds") or 0) > 0 else 0.0
            for trace in enabled_traces
        ]),
        "repair_success_rate": _avg([
            1.0 if trace.get("repair_success") else 0.0
            for trace in repair_traces
        ]),
        "avg_agent_elapsed_sec": _avg([
            float(trace.get("agent_elapsed_sec") or 0.0)
            for trace in enabled_traces
        ]),
    }


def evaluate_rows(rows: list[dict[str, Any]], label: str, k: int = 5) -> dict[str, Any]:
    recall_key = f"recall_at_{k}"
    evaluated: list[dict[str, Any]] = []

    for row in rows:
        # 当前只评估检索层；生成质量暂时保留在人工报告中判断。
        retrieved = row.get("retrieved_sources", []) or []
        gold = row.get("gold_sources", []) or []
        context = row.get("context", {}) or {}
        source_hit = source_hit_status(retrieved, gold, k=k)
        completeness = answer_completeness(row)
        coverage = evidence_coverage(
            str(row.get("predicted_answer", "")),
            [str(item) for item in (row.get("gold_evidence") or [])],
        )
        citation_accuracy = _citation_accuracy(source_hit)
        skipped = bool(row.get("skipped", False))
        error = row.get("error")
        agent_trace = row.get("agent_trace") or {}
        item = {
            "id": row.get("id"),
            "task_type": row.get("task_type", "unknown"),
            "difficulty": row.get("difficulty", "unknown"),
            recall_key: recall_at_k(retrieved, gold, k=k),
            "mrr": mrr(retrieved, gold),
            "source_hit": source_hit,
            "elapsed_sec": row.get("elapsed_sec", 0.0) or 0.0,
            "error": error,
            "skipped": skipped,
            "retrieval": {
                recall_key: recall_at_k(retrieved, gold, k=k),
                "mrr": mrr(retrieved, gold),
                "source_hit": source_hit,
                "retrieved_count": len(retrieved),
                "gold_source_count": len(gold),
            },
            "context": {
                "source_doc_count": int(context.get("source_doc_count") or len(retrieved)),
                "context_doc_count": int(context.get("context_doc_count") or len(retrieved)),
                "input_chars": int(context.get("input_chars") or 0),
                "output_chars": int(context.get("output_chars") or 0),
                "compression_ratio": _compression_ratio(context),
                "parent_hit_count": int(context.get("parent_hit_count") or 0),
            },
            "answer": {
                "evidence_coverage": coverage,
                "answer_completeness": completeness,
                "citation_accuracy": citation_accuracy,
                "unsupported_claim_count": int(row.get("unsupported_claim_count") or 0),
            },
            "experience": {
                "elapsed_sec": row.get("elapsed_sec", 0.0) or 0.0,
                "error": error,
                "skipped": skipped,
            },
            "agent_trace": agent_trace,
        }
        item["error_bucket"] = _error_bucket(error, skipped, source_hit, completeness)
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
            "error_bucket": item["error_bucket"],
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
        "layers": _layer_summary(evaluated, k),
        "agent": _agent_summary(evaluated),
        "error_bucket_counts": dict(sorted(Counter(item["error_bucket"] for item in evaluated).items())),
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
