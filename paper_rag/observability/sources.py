from __future__ import annotations

import os
from typing import Any


def source_from_doc(doc: Any, preview_chars: int = 200) -> dict[str, Any]:
    """统一将 LangChain Document 转为日志、反馈和评估可复用的来源结构。"""
    metadata = getattr(doc, "metadata", {}) or {}
    source = metadata.get("source", "unknown")
    item = {
        "file": os.path.basename(str(source).replace("\\", "/")),
        "page": metadata.get("page", -1),
        "content_preview": getattr(doc, "page_content", "")[:preview_chars],
    }
    if "rerank_score" in metadata:
        item["rerank_score"] = metadata["rerank_score"]
    return item


def sources_from_docs(docs: list[Any], preview_chars: int = 200) -> list[dict[str, Any]]:
    return [source_from_doc(doc, preview_chars=preview_chars) for doc in docs]

