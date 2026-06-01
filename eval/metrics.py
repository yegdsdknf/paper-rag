from __future__ import annotations

import os
import re
from typing import Any


def normalize_source(source: str) -> str:
    # baseline 结果可能保存绝对路径、Windows 路径或纯文件名，这里统一成文件名。
    return os.path.basename(str(source).replace("\\", "/")).strip().lower()


def normalize_page(page: Any) -> int | str:
    if page is None:
        return ""
    try:
        return int(page)
    except (TypeError, ValueError):
        return str(page).strip()


def source_matches(retrieved: dict[str, Any], gold: dict[str, Any]) -> bool:
    return (
        normalize_source(retrieved.get("file", "")) == normalize_source(gold.get("file", ""))
        and normalize_page(retrieved.get("page")) == normalize_page(gold.get("page"))
    )


def recall_at_k(
    retrieved_sources: list[dict[str, Any]],
    gold_sources: list[dict[str, Any]],
    k: int = 5,
) -> float:
    if not gold_sources:
        return 1.0

    top_k = retrieved_sources[:k]
    matched = 0
    for gold in gold_sources:
        if any(source_matches(retrieved, gold) for retrieved in top_k):
            matched += 1
    return matched / len(gold_sources)


def mrr(
    retrieved_sources: list[dict[str, Any]],
    gold_sources: list[dict[str, Any]],
) -> float:
    for index, retrieved in enumerate(retrieved_sources, 1):
        if any(source_matches(retrieved, gold) for gold in gold_sources):
            return 1.0 / index
    return 0.0


def source_hit_status(
    retrieved_sources: list[dict[str, Any]],
    gold_sources: list[dict[str, Any]],
    k: int | None = None,
) -> str:
    # 保留三档状态，方便报告区分完全缺失和部分证据命中。
    if not gold_sources:
        return "full"

    candidates = retrieved_sources if k is None else retrieved_sources[:k]
    hits = sum(
        1
        for gold in gold_sources
        if any(source_matches(retrieved, gold) for retrieved in candidates)
    )
    if hits == len(gold_sources):
        return "full"
    if hits > 0:
        return "partial"
    return "missing"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def evidence_coverage(answer: str, evidence_items: list[str]) -> float:
    """用轻量字符串覆盖率衡量答案是否包含人工标注的关键证据。"""
    if not evidence_items:
        return 1.0
    normalized_answer = _normalize_text(answer)
    hits = sum(1 for item in evidence_items if _normalize_text(item) in normalized_answer)
    return round(hits / len(evidence_items), 4)


def answer_completeness(row: dict[str, Any]) -> float:
    targets = row.get("key_points") or row.get("gold_evidence") or []
    if isinstance(targets, str):
        targets = [targets]
    return evidence_coverage(str(row.get("predicted_answer", "")), [str(item) for item in targets])
