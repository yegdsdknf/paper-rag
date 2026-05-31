"""
Run baseline answers for benchmarks/benchmark_v1.jsonl.

Examples:
  python benchmarks/run_baseline.py
  python benchmarks/run_baseline.py --limit 3
  python benchmarks/run_baseline.py --no-generate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks" / "benchmark_v1.jsonl"
DEFAULT_OUTPUT_PATH = ROOT / "benchmarks" / "baseline_results_qwen2.5_3b.jsonl"


def _ensure_project_importable() -> None:
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def load_samples(path: Path = BENCHMARK_PATH, limit: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample["_line_no"] = line_no
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    return samples


def _make_result(
    sample: dict[str, Any],
    predicted_answer: str,
    retrieved_sources: list[dict[str, Any]],
    elapsed_sec: float,
    error: str | None = None,
    skipped: bool = False,
) -> dict[str, Any]:
    return {
        "id": sample["id"],
        "question": sample["question"],
        "history": sample.get("history", []),
        "task_type": sample["task_type"],
        "difficulty": sample["difficulty"],
        "gold_answer": sample["gold_answer"],
        "gold_sources": sample["gold_sources"],
        "gold_evidence": sample.get("gold_evidence", []),
        "predicted_answer": predicted_answer,
        "retrieved_sources": retrieved_sources,
        "elapsed_sec": round(elapsed_sec, 2),
        "error": error,
        "skipped": skipped,
    }


def _run_sample(
    sample: dict[str, Any],
    hybrid: Any,
    conversation_cls: Any,
    rag_pipeline: Any,
    config: dict[str, Any],
    llm_model: str,
    source_serializer: Any,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        conversation = conversation_cls(
            llm_model=llm_model,
            temperature=config["temperature"],
            num_ctx=config.get("llm_num_ctx"),
            num_predict=config.get("llm_num_predict"),
        )
        conversation.history = list(sample.get("history", []))

        answer, docs = rag_pipeline.ask_with_context(
            hybrid,
            conversation,
            sample["question"],
            llm_model=llm_model,
            temperature=config["temperature"],
        )
        elapsed_sec = time.perf_counter() - start
        return _make_result(
            sample,
            predicted_answer=answer,
            retrieved_sources=source_serializer(docs, preview_chars=200),
            elapsed_sec=elapsed_sec,
        )
    except Exception as exc:
        elapsed_sec = time.perf_counter() - start
        return _make_result(
            sample,
            predicted_answer="",
            retrieved_sources=[],
            elapsed_sec=elapsed_sec,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_baseline(
    benchmark_path: Path = BENCHMARK_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    limit: int | None = None,
    no_generate: bool = False,
    llm_model: str | None = None,
) -> list[dict[str, Any]]:
    samples = load_samples(benchmark_path, limit=limit)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if no_generate:
        results = [
            _make_result(sample, "", [], 0.0, skipped=True)
            for sample in samples
        ]
    else:
        _ensure_project_importable()
        from conversation import ConversationManager
        import rag_pipeline
        from source_utils import sources_from_docs
        from utils.config_loader import load_config

        config = load_config()
        selected_model = llm_model or config["llm_model"]
        hybrid = rag_pipeline.build_hybrid_retriever()
        results = []

        for index, sample in enumerate(samples, 1):
            print(f"[{index}/{len(samples)}] {sample['id']} {sample['question']}")
            result = _run_sample(
                sample,
                hybrid=hybrid,
                conversation_cls=ConversationManager,
                rag_pipeline=rag_pipeline,
                config=config,
                llm_model=selected_model,
                source_serializer=sources_from_docs,
            )
            results.append(result)

    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    error_count = sum(1 for result in results if result.get("error"))
    print(f"样本数: {len(results)}")
    print(f"错误数: {error_count}")
    print(f"输出文件: {output_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Paper RAG baseline benchmark.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=BENCHMARK_PATH,
        help="Benchmark JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N samples.",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Only validate benchmark loading and output schema; skip RAG calls.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override the configured LLM model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_baseline(
        benchmark_path=args.benchmark,
        output_path=args.output,
        limit=args.limit,
        no_generate=args.no_generate,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
