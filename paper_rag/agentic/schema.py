from __future__ import annotations

import os
from typing import Any, Literal, TypedDict

from langchain_core.documents import Document


GoalType = Literal["page_evidence", "compare_dimension", "method_overview", "figure_evidence"]
VerificationStatus = Literal["supported", "partial", "unsupported"]

VALID_GOAL_TYPES: set[str] = {"page_evidence", "compare_dimension", "method_overview", "figure_evidence"}
VALID_STATUSES: set[str] = {"supported", "partial", "unsupported"}


class EvidenceGoal(TypedDict, total=False):
    id: str
    goal_type: GoalType
    claim: str
    query: str
    source_hint: str
    page_hint: int | None


class VerifiedEvidence(TypedDict, total=False):
    goal_id: str
    claim: str
    status: VerificationStatus
    supporting_sources: list[dict[str, Any]]
    missing_terms: list[str]


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


def normalize_goal(raw: dict[str, Any], index: int, allowed_sources: set[str] | None = None) -> EvidenceGoal:
    goal_type = str(raw.get("goal_type") or "").strip()
    if goal_type not in VALID_GOAL_TYPES:
        goal_type = "method_overview"

    goal_id = str(raw.get("id") or "").strip() or f"g{index + 1}"
    claim = str(raw.get("claim") or "").strip()
    query = str(raw.get("query") or claim).strip()
    source_hint = _normalize_source_hint(raw.get("source_hint"), allowed_sources)

    return {
        "id": goal_id,
        "goal_type": goal_type,
        "claim": claim,
        "query": query,
        "source_hint": source_hint,
        "page_hint": _to_int_or_none(raw.get("page_hint")),
    }


def docs_to_agent_sources(docs: list[Document]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for doc in docs:
        metadata = doc.metadata or {}
        source_value = metadata.get("source") or metadata.get("source_file")
        sources.append(
            {
                "file": _basename(source_value) or "unknown",
                "page": _to_int_or_default(metadata.get("page"), -1),
                "content_preview": _preview(doc.page_content),
            }
        )
    return sources


def normalize_verified_evidence(raw: dict[str, Any], goal_id: str | None = None) -> VerifiedEvidence:
    normalized_goal_id = str(goal_id if goal_id is not None else raw.get("goal_id") or "").strip()
    status = str(raw.get("status") or "").strip()
    if status not in VALID_STATUSES:
        status = "unsupported"

    supporting_sources = raw.get("supporting_sources")
    if not isinstance(supporting_sources, list):
        supporting_sources = []
    supporting_sources = [dict(item) for item in supporting_sources if isinstance(item, dict)]

    missing_terms = raw.get("missing_terms")
    if not isinstance(missing_terms, list):
        missing_terms = []
    cleaned_missing_terms = []
    for item in missing_terms:
        if item is None:
            continue
        term = str(item).strip()
        if term:
            cleaned_missing_terms.append(term)

    return {
        "goal_id": normalized_goal_id,
        "claim": str(raw.get("claim") or "").strip(),
        "status": status,
        "supporting_sources": supporting_sources,
        "missing_terms": cleaned_missing_terms,
    }


def _normalize_source_hint(value: Any, allowed_sources: set[str] | None) -> str:
    source_hint = _basename(value)
    if allowed_sources is not None and source_hint not in allowed_sources:
        return ""
    return source_hint


def _basename(value: Any) -> str:
    return os.path.basename(str(value or "").strip().replace("\\", "/"))


def _to_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_int_or_default(value: Any, default: int) -> int:
    parsed = _to_int_or_none(value)
    if parsed is None:
        return default
    return parsed


def _preview(text: str) -> str:
    return " ".join(str(text).split())[:180]
