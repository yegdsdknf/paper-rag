from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_rag.observability.sources import sources_from_docs


DEFAULT_FEEDBACK_PATH = Path("benchmarks") / "feedback" / "feedback.jsonl"


def build_feedback_record(
    question: str,
    answer: str,
    sources: list[Any],
    note: str,
    route: str | None = None,
    llm_model: str | None = None,
) -> dict[str, Any]:
    """构造可人工转写为 benchmark 样本的真实使用反馈记录。"""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "sources": sources_from_docs(sources, preview_chars=300),
        "route": route or "",
        "llm_model": llm_model or "",
        "note": note.strip(),
        "status": "pending_label",
    }


def save_feedback_record(
    record: dict[str, Any],
    output_path: str | Path = DEFAULT_FEEDBACK_PATH,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
