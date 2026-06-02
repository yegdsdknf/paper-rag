from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass(frozen=True)
class AgenticContextResult:
    final_docs: list[Document]
    verified_summary: str


def build_verified_evidence_summary(verified_evidence: list[dict[str, Any]]) -> str:
    lines = ["【已校验证据】"]
    for item in verified_evidence:
        goal_id = str(item.get("goal_id") or "unknown")
        status = str(item.get("status") or "unsupported")
        lines.append(f"Goal {goal_id}: {status}")

        claim = str(item.get("claim") or "").strip()
        if claim:
            lines.append(f"Claim: {claim}")

        sources = _format_supporting_sources(item.get("supporting_sources"))
        if sources:
            lines.append(f"Sources: {sources}")

    lines.extend(
        [
            "回答约束：",
            "- 优先使用 supported 证据回答。",
            "- partial 证据需谨慎表述，明确不确定或缺失部分。",
            "- unsupported 需说明未找到足够证据，不要编造。",
            "- 不要跨论文错归因；引用来源必须与证据文件和页码一致。",
        ]
    )
    return "\n".join(lines)


def assemble_agentic_context(
    docs: list[Document],
    verified_evidence: list[dict[str, Any]],
    task_type: str,
) -> AgenticContextResult:
    supported_keys = _supported_source_keys(verified_evidence)
    sorted_docs = sorted(
        list(docs),
        key=lambda doc: (
            0 if task_type == "figure" and _is_vision_doc(doc) else 1,
            0 if _doc_key(doc) in supported_keys else 1,
        ),
    )

    final_docs = sorted_docs
    if task_type in {"evidence", "figure", "followup"} and supported_keys:
        filtered = [
            doc
            for doc in sorted_docs
            if _doc_key(doc) in supported_keys or (task_type == "figure" and _is_vision_doc(doc))
        ]
        if filtered:
            final_docs = filtered

    return AgenticContextResult(
        final_docs=final_docs,
        verified_summary=build_verified_evidence_summary(verified_evidence),
    )


def _supported_source_keys(verified_evidence: list[dict[str, Any]]) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for item in verified_evidence:
        if item.get("status") not in {"supported", "partial"}:
            continue
        sources = item.get("supporting_sources")
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, dict):
                continue
            key = _source_key(source.get("file"), source.get("page"))
            if key is not None:
                keys.add(key)
    return keys


def _doc_key(doc: Document) -> tuple[str, int] | None:
    metadata = doc.metadata or {}
    return _source_key(metadata.get("source") or metadata.get("source_file"), metadata.get("page"))


def _source_key(source: Any, page: Any) -> tuple[str, int] | None:
    basename = _basename(source)
    parsed_page = _to_int(page)
    if not basename or parsed_page is None:
        return None
    return basename, parsed_page


def _format_supporting_sources(raw_sources: Any) -> str:
    if not isinstance(raw_sources, list):
        return ""

    rendered: list[str] = []
    for source in raw_sources:
        if not isinstance(source, dict):
            continue
        file_name = _basename(source.get("file")) or "unknown"
        page = _to_int(source.get("page"))
        if page is None:
            rendered.append(file_name)
        else:
            rendered.append(f"{file_name} p{page}")
    return ", ".join(rendered)


def _is_vision_doc(doc: Document) -> bool:
    metadata = doc.metadata or {}
    return metadata.get("paper_region") == "vision" or metadata.get("chunk_strategy") == "vision_summary"


def _basename(value: Any) -> str:
    return os.path.basename(str(value or "").strip().replace("\\", "/"))


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
