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
) -> tuple[list[Document], str]:
    goal_type = str(goal.get("goal_type") or "method_overview")
    query = str(goal.get("query") or goal.get("claim") or "").strip()
    source_hint = str(goal.get("source_hint") or "").strip()

    if goal_type == "figure_evidence":
        loader = vision_loader or _load_vision_docs_from_hybrid
        vision_docs = _filter_by_source_hint(loader(hybrid, source_hint), source_hint)
        if vision_docs:
            return vision_docs, "agentic_figure"

        docs = _route_docs(router, hybrid, query, llm_model, temperature)
        return _filter_by_source_hint(docs, source_hint), "agentic_figure_text_fallback"

    route_name = _ROUTES_BY_GOAL_TYPE.get(goal_type, "agentic_method")
    docs = _route_docs(router, hybrid, query, llm_model, temperature)
    return _filter_by_source_hint(docs, source_hint), route_name


def _route_docs(router: Any, hybrid: Any, query: str, llm_model: str, temperature: float) -> list[Document]:
    docs, _strategy = router.route(hybrid, query, llm_model=llm_model, temperature=temperature)
    return list(docs)


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

    return _filter_by_source_hint(vision_docs, source_hint)


def _filter_by_source_hint(docs: list[Document], source_hint: str) -> list[Document]:
    if not source_hint:
        return list(docs)

    filtered = [doc for doc in docs if _matches_source_hint(doc, source_hint)]
    return filtered or list(docs)


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


def _basename(value: Any) -> str:
    return os.path.basename(str(value or "").strip().replace("\\", "/"))
