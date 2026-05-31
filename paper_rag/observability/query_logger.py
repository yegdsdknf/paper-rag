from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_rag.observability.sources import sources_from_docs


DEFAULT_QUERY_LOG_PATH = Path("logs") / "query_runs.jsonl"


def build_query_log_record(
    question: str,
    standalone_question: str,
    route: str,
    llm_model: str,
    embedding_device: str,
    docs: list[Any],
    elapsed: dict[str, float],
    index_version: str | None = None,
    feature_flags: dict[str, bool] | None = None,
    query_variants: list[str] | None = None,
    context_stats: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "standalone_question": standalone_question,
        "route": route,
        "llm_model": llm_model,
        "embedding_device": embedding_device,
        "index_version": index_version or "unknown",
        "feature_flags": feature_flags or {},
        "query_variants": query_variants or [],
        "retrieved_sources": sources_from_docs(docs, preview_chars=200),
        "context": context_stats or {},
        "elapsed": {key: round(float(value), 4) for key, value in elapsed.items()},
        "error": error,
    }


def save_query_log_record(
    record: dict[str, Any],
    output_path: str | Path = DEFAULT_QUERY_LOG_PATH,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
