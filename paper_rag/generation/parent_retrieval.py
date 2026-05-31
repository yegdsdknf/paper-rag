from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

from langchain_core.documents import Document


def _source_key(source: Any) -> str:
    return os.path.basename(str(source).replace("\\", "/")).lower()


def _page_key(page: Any) -> int | str:
    try:
        return int(page)
    except (TypeError, ValueError):
        return str(page)


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def expand_parent_pages(
    hybrid: Any,
    child_docs: list[Document],
    max_chars_per_parent: int = 2500,
) -> list[Document]:
    """按 child chunk 的来源页回溯同页 chunks，合并为生成阶段使用的 parent page。"""
    vector_store = getattr(hybrid, "vector_store", None)
    if vector_store is None or not child_docs:
        return child_docs

    wanted: OrderedDict[tuple[str, int | str], dict[str, Any]] = OrderedDict()
    for doc in child_docs:
        key = (_source_key(doc.metadata.get("source", "")), _page_key(doc.metadata.get("page")))
        if key not in wanted:
            wanted[key] = {
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page"),
                "child_count": 0,
            }
        wanted[key]["child_count"] += 1

    try:
        stored = vector_store.get(include=["documents", "metadatas"])
    except Exception as exc:
        print(f"WARNING: Parent retrieval load failed: {type(exc).__name__}: {exc}")
        return child_docs

    grouped: dict[tuple[str, int | str], list[str]] = {key: [] for key in wanted}
    for content, metadata in zip(stored.get("documents", []), stored.get("metadatas", [])):
        key = (_source_key(metadata.get("source", "")), _page_key(metadata.get("page")))
        if key in grouped and content:
            grouped[key].append(str(content).strip())

    parent_docs: list[Document] = []
    for key, info in wanted.items():
        parts = [part for part in grouped.get(key, []) if part]
        if not parts:
            continue

        page_text = "\n\n".join(parts)
        metadata = {
            "source": info["source"],
            "page": info["page"],
            "parent_context": True,
            "parent_child_count": info["child_count"],
            "parent_original_chars": len(page_text),
        }
        clipped = _clip_text(page_text, max_chars_per_parent)
        metadata["parent_context_chars"] = len(clipped)
        parent_docs.append(Document(page_content=clipped, metadata=metadata))

    return parent_docs or child_docs
