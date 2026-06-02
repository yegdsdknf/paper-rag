from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from typing import Mapping

from langchain_core.documents import Document


def _source_file(source: object) -> str:
    return os.path.basename(str(source).replace("\\", "/"))


def _short_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _doc_id(source_file: str, source_file_hashes: Mapping[str, str] | None) -> str:
    hashes = source_file_hashes or {}
    file_hash = hashes.get(source_file) or hashes.get(source_file.lower())
    if file_hash:
        return f"doc_{file_hash}"
    return f"doc_{_short_hash(source_file.lower())}"


def infer_paper_region(section_title: object, existing: object = None) -> str:
    if existing:
        return str(existing)

    title = str(section_title or "").strip().lower()
    if not title:
        return "unknown"
    if "abstract" in title or title == "摘要":
        return "abstract"
    if "reference" in title or "bibliography" in title or "参考文献" in title:
        return "references"
    if "appendix" in title or "supplement" in title or "附录" in title:
        return "appendix"
    if "title" in title or "front" in title:
        return "front_matter"
    return "body"


def attach_chunk_metadata(
    docs: list[Document],
    strategy: str,
    schema_version: str,
    source_file_hashes: Mapping[str, str] | None = None,
) -> list[Document]:
    """为 chunk 注入论文 RAG 可追踪元数据，不修改输入 Document。"""
    page_counts: defaultdict[tuple[str, object], int] = defaultdict(int)
    enriched: list[Document] = []

    for global_index, doc in enumerate(docs):
        metadata = dict(doc.metadata)
        source = metadata.get("source", "")
        source_file = _source_file(source)
        page = metadata.get("page", -1)
        page_key = (source_file.lower(), page)
        page_chunk_index = page_counts[page_key]
        page_counts[page_key] += 1

        doc_id = str(metadata.get("doc_id") or _doc_id(source_file, source_file_hashes))
        content_hash = _short_hash(doc.page_content)
        metadata.update(
            {
                "doc_id": doc_id,
                "source_file": source_file,
                "global_chunk_index": global_index,
                "page_chunk_index": page_chunk_index,
                "chunk_strategy": strategy,
                "chunk_schema_version": schema_version,
                "content_hash": content_hash,
                "chunk_id": f"{doc_id}:{schema_version}:{global_index}:{content_hash}",
                "paper_region": infer_paper_region(
                    metadata.get("section_title"),
                    metadata.get("paper_region"),
                ),
            }
        )
        enriched.append(Document(page_content=doc.page_content, metadata=metadata))

    return enriched
