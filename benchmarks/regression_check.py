from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_eval import evaluate_rows, load_jsonl


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    baseline: float
    current: float
    delta: float
    tolerance: float
    direction: str


@dataclass(frozen=True)
class RegressionReport:
    status: str
    exit_code: int
    message: str
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    hard_failures: list[MetricDelta] = field(default_factory=list)
    warnings: list[MetricDelta] = field(default_factory=list)


def _missing_rate(report: dict[str, Any]) -> float:
    sample_count = int(report.get("sample_count") or 0)
    if sample_count <= 0:
        return 0.0
    missing = int((report.get("source_hit_counts") or {}).get("missing", 0))
    return round(missing / sample_count, 4)


def _metric_summary(report: dict[str, Any]) -> dict[str, float]:
    averages = report.get("averages") or {}
    answer = ((report.get("layers") or {}).get("answer") or {})
    experience = ((report.get("layers") or {}).get("experience") or {})
    return {
        "recall@5": float(averages.get("recall_at_5", 0.0)),
        "mrr": float(averages.get("mrr", 0.0)),
        "missing_rate": _missing_rate(report),
        "answer_completeness": float(answer.get("avg_answer_completeness", 0.0)),
        "evidence_coverage": float(answer.get("avg_evidence_coverage", 0.0)),
        "avg_elapsed_sec": float(experience.get("avg_elapsed_sec", averages.get("elapsed_sec", 0.0))),
    }


def _ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("id")) for row in rows}


def _input_error(message: str) -> RegressionReport:
    return RegressionReport(status="ERROR", exit_code=2, message=message)


def compare_result_files(
    baseline_path: Path,
    current_path: Path,
    *,
    k: int = 5,
) -> RegressionReport:
    try:
        baseline_rows = load_jsonl(baseline_path)
        current_rows = load_jsonl(current_path)
    except Exception as exc:
        return _input_error(f"输入文件读取失败: {type(exc).__name__}: {exc}")

    baseline_ids = _ids(baseline_rows)
    current_ids = _ids(current_rows)
    if baseline_ids != current_ids:
        return _input_error(
            "样本 id 不一致: "
            f"baseline_only={sorted(baseline_ids - current_ids)}, "
            f"current_only={sorted(current_ids - baseline_ids)}"
        )

    baseline_report = evaluate_rows(baseline_rows, label="baseline", k=k)
    current_report = evaluate_rows(current_rows, label="current", k=k)
    baseline_metrics = _metric_summary(baseline_report)
    current_metrics = _metric_summary(current_report)

    hard_failures: list[MetricDelta] = []
    warnings: list[MetricDelta] = []
    _check_drop(hard_failures, "recall@5", baseline_metrics, current_metrics, tolerance=0.03)
    _check_drop(hard_failures, "mrr", baseline_metrics, current_metrics, tolerance=0.03)
    _check_rise(hard_failures, "missing_rate", baseline_metrics, current_metrics, tolerance=0.05)
    _check_drop(warnings, "answer_completeness", baseline_metrics, current_metrics, tolerance=0.05)
    _check_drop(warnings, "evidence_coverage", baseline_metrics, current_metrics, tolerance=0.05)
    _check_elapsed(warnings, baseline_metrics, current_metrics, tolerance_ratio=0.20)

    if hard_failures:
        status = "FAIL"
        exit_code = 1
        message = "检测到检索硬门禁回退"
    elif warnings:
        status = "WARN"
        exit_code = 0
        message = "未检测到硬门禁回退，但存在 WARN 指标变化"
    else:
        status = "OK"
        exit_code = 0
        message = "未检测到指标回退"

    return RegressionReport(
        status=status,
        exit_code=exit_code,
        message=message,
        metrics=current_metrics,
        baseline_metrics=baseline_metrics,
        hard_failures=hard_failures,
        warnings=warnings,
    )


def _check_drop(
    bucket: list[MetricDelta],
    metric: str,
    baseline: dict[str, float],
    current: dict[str, float],
    *,
    tolerance: float,
) -> None:
    delta = round(current[metric] - baseline[metric], 4)
    if delta < -tolerance:
        bucket.append(MetricDelta(metric, baseline[metric], current[metric], delta, tolerance, "drop"))


def _check_rise(
    bucket: list[MetricDelta],
    metric: str,
    baseline: dict[str, float],
    current: dict[str, float],
    *,
    tolerance: float,
) -> None:
    delta = round(current[metric] - baseline[metric], 4)
    if delta > tolerance:
        bucket.append(MetricDelta(metric, baseline[metric], current[metric], delta, tolerance, "rise"))


def _check_elapsed(
    bucket: list[MetricDelta],
    baseline: dict[str, float],
    current: dict[str, float],
    *,
    tolerance_ratio: float,
) -> None:
    metric = "avg_elapsed_sec"
    baseline_value = baseline[metric]
    current_value = current[metric]
    if baseline_value <= 0:
        return
    allowed = baseline_value * (1 + tolerance_ratio)
    if current_value > allowed:
        bucket.append(
            MetricDelta(
                metric,
                baseline_value,
                current_value,
                round(current_value - baseline_value, 4),
                tolerance_ratio,
                "rise",
            )
        )


def _delta_to_dict(delta: MetricDelta) -> dict[str, Any]:
    return {
        "metric": delta.metric,
        "baseline": delta.baseline,
        "current": delta.current,
        "delta": delta.delta,
        "tolerance": delta.tolerance,
        "direction": delta.direction,
    }


def report_to_dict(report: RegressionReport) -> dict[str, Any]:
    return {
        "status": report.status,
        "exit_code": report.exit_code,
        "message": report.message,
        "baseline_metrics": report.baseline_metrics,
        "metrics": report.metrics,
        "hard_failures": [_delta_to_dict(item) for item in report.hard_failures],
        "warnings": [_delta_to_dict(item) for item in report.warnings],
    }


def render_json(report: RegressionReport) -> str:
    return json.dumps(report_to_dict(report), ensure_ascii=False, indent=2)


def render_text(report: RegressionReport) -> str:
    lines = [
        f"Benchmark Regression: {report.status}",
        report.message,
    ]
    if report.metrics:
        lines.append("")
        lines.append("Metrics:")
        for metric, value in report.metrics.items():
            baseline = report.baseline_metrics.get(metric)
            lines.append(f"- {metric}: {value} (baseline={baseline})")
    if report.hard_failures:
        lines.append("")
        lines.append("Hard failures:")
        for item in report.hard_failures:
            lines.append(f"- {item.metric}: {item.current} vs {item.baseline} (delta={item.delta})")
    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        for item in report.warnings:
            lines.append(f"- {item.metric}: {item.current} vs {item.baseline} (delta={item.delta})")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two Paper-RAG baseline result JSONL files.")
    parser.add_argument("--baseline", required=True, type=Path, help="Baseline result JSONL path.")
    parser.add_argument("--current", required=True, type=Path, help="New/current result JSONL path.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    parser.add_argument("--k", type=int, default=5, help="Recall@k cutoff.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_result_files(args.baseline, args.current, k=args.k)
    print(render_json(report) if args.json else render_text(report))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
