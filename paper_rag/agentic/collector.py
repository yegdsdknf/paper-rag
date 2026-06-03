from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document

from paper_rag.agentic.schema import EvidenceGoal


VisionLoader = Callable[[Any, str], list[Document]]


_ROUTES_BY_GOAL_TYPE = {
    "page_evidence": "agentic_page_evidence",
    "compare_dimension": "agentic_compare",
    "method_overview": "agentic_method",
}


def collect_for_goal(
    goal: EvidenceGoal,
    hybrid: Any,
    router: Any,
    llm_model: str = "",
    temperature: float = 0.0,
    vision_loader: VisionLoader | None = None,
    repair_round: int = 0,
) -> tuple[list[Document], str]:
    goal_type = str(goal.get("goal_type") or "method_overview")
    query = str(goal.get("query") or goal.get("claim") or "").strip()
    source_hint = str(goal.get("source_hint") or "").strip()
    log_context = _goal_log_context(goal, repair_round)

    if goal_type == "figure_evidence":
        loader = vision_loader or _load_vision_docs_from_hybrid
        vision_docs = _filter_by_source_hint(loader(hybrid, source_hint), source_hint, fallback_to_all=False)
        if vision_docs:
            if _is_page_locator_query(query or str(goal.get("claim") or "")):
                return _deduplicate_docs(vision_docs), "agentic_figure"
            neighbor_docs = _load_neighbor_text_docs_from_hybrid(hybrid, vision_docs, source_hint)
            return _deduplicate_docs(vision_docs + neighbor_docs), "agentic_figure"

        docs = _route_docs(router, hybrid, query, llm_model, temperature, log_context)
        return _filter_by_source_hint(docs, source_hint), "agentic_figure_text_fallback"

    route_name = _ROUTES_BY_GOAL_TYPE.get(goal_type, "agentic_method")
    docs = _route_docs(router, hybrid, query, llm_model, temperature, log_context)
    return _filter_by_source_hint(docs, source_hint), route_name


def _deduplicate_docs(docs: list[Document]) -> list[Document]:
    seen: set[tuple[str, int]] = set()
    unique: list[Document] = []
    for doc in docs:
        metadata = doc.metadata or {}
        key = (_doc_source_basename(doc), _to_int(metadata.get("page"), -1))
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _route_docs(router: Any, hybrid: Any, query: str, llm_model: str, temperature: float, log_context: str) -> list[Document]:
    try:
        docs, _strategy = router.route(hybrid, query, llm_model=llm_model, temperature=temperature, log_context=log_context)
    except TypeError:
        docs, _strategy = router.route(hybrid, query, llm_model=llm_model, temperature=temperature)
    return list(docs)


def _goal_log_context(goal: EvidenceGoal, repair_round: int) -> str:
    goal_id = str(goal.get("id") or "unknown").strip() or "unknown"
    goal_type = str(goal.get("goal_type") or "method_overview").strip() or "method_overview"
    if repair_round > 0:
        return f"[agent_repair round={repair_round} goal={goal_id} type={goal_type}]"
    return f"[agent_collect goal={goal_id} type={goal_type}]"


def _load_vision_docs_from_hybrid(hybrid: Any, source_hint: str) -> list[Document]:
    vector_store = getattr(hybrid, "vector_store", None)
    if vector_store is None:
        return []

    try:
        raw = vector_store.get(include=["documents", "metadatas"])
    except Exception:
        return []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []

    vision_docs: list[Document] = []
    for content, metadata in zip(documents, metadatas):
        if not isinstance(metadata, dict):
            continue
        if isinstance(content, Document):
            merged_metadata = {**dict(content.metadata or {}), **dict(metadata)}
            page_content = content.page_content
        else:
            merged_metadata = dict(metadata)
            page_content = str(content)

        if (
            merged_metadata.get("paper_region") != "vision"
            and merged_metadata.get("chunk_strategy") != "vision_summary"
        ):
            continue
        vision_docs.append(Document(page_content=page_content, metadata=merged_metadata))

    return _filter_by_source_hint(vision_docs, source_hint, fallback_to_all=False)


def _load_neighbor_text_docs_from_hybrid(
    hybrid: Any,
    vision_docs: list[Document],
    source_hint: str,
    max_pages_ahead: int = 2,
) -> list[Document]:
    vector_store = getattr(hybrid, "vector_store", None)
    if vector_store is None or not vision_docs:
        return []

    wanted: dict[str, set[int]] = {}
    for doc in vision_docs:
        source = _doc_source_basename(doc) or _basename(source_hint)
        page = _to_int((doc.metadata or {}).get("page"), -1)
        if not source or page < 0:
            continue
        wanted.setdefault(source.lower(), set()).update(range(page + 1, page + max_pages_ahead + 1))
    if not wanted:
        return []

    try:
        raw = vector_store.get(include=["documents", "metadatas"])
    except Exception:
        return []

    selected: list[Document] = []
    seen: set[tuple[str, int]] = set()
    for content, metadata in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        if not isinstance(metadata, dict):
            continue
        if isinstance(content, Document):
            candidate = Document(page_content=content.page_content, metadata={**content.metadata, **metadata})
        else:
            candidate = Document(page_content=str(content), metadata=dict(metadata))
        if _is_vision_doc(candidate):
            continue
        source = _doc_source_basename(candidate).lower()
        page = _to_int((candidate.metadata or {}).get("page"), -1)
        if page not in wanted.get(source, set()):
            continue
        if source_hint and not _matches_source_hint(candidate, source_hint):
            continue
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
    return selected


def _filter_by_source_hint(docs: list[Document], source_hint: str, fallback_to_all: bool = True) -> list[Document]:
    if not source_hint:
        return list(docs)

    filtered = [doc for doc in docs if _matches_source_hint(doc, source_hint)]
    if filtered or not fallback_to_all:
        return filtered
    return list(docs)


def _matches_source_hint(doc: Document, source_hint: str) -> bool:
    hint = _basename(source_hint).lower()
    if not hint:
        return True

    metadata = doc.metadata or {}
    candidates = [
        metadata.get("source"),
        metadata.get("source_file"),
        _basename(metadata.get("source")),
        _basename(metadata.get("source_file")),
    ]
    return any(hint in str(candidate or "").replace("\\", "/").lower() for candidate in candidates)


def _doc_source_basename(doc: Document) -> str:
    metadata = doc.metadata or {}
    return _basename(metadata.get("source") or metadata.get("source_file"))


def _is_vision_doc(doc: Document) -> bool:
    metadata = doc.metadata or {}
    return metadata.get("paper_region") == "vision" or metadata.get("chunk_strategy") == "vision_summary"


def _is_page_locator_query(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(signal in lowered for signal in ["在哪一页", "哪一页", "页码", "which page", "page number"])


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _basename(value: Any) -> str:
    return os.path.basename(str(value or "").strip().replace("\\", "/"))
