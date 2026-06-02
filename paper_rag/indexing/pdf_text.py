from __future__ import annotations

import os
import re

from langchain_core.documents import Document


_UNICODE_ESCAPE_RE = re.compile(r"/uni[0-9a-fA-F]{8}")


def is_noisy_pdf_text(
    text: str,
    min_unicode_escape_count: int = 50,
    min_unicode_escape_ratio: float = 0.08,
) -> bool:
    """识别 PDF 字体编码泄漏造成的垃圾页，例如大量 /uni00000019 token。"""
    if not text:
        return False

    matches = _UNICODE_ESCAPE_RE.findall(text)
    if len(matches) < min_unicode_escape_count:
        return False

    escaped_chars = sum(len(match) for match in matches)
    return escaped_chars / max(len(text), 1) >= min_unicode_escape_ratio


def filter_noisy_pdf_pages(docs: list[Document]) -> tuple[list[Document], list[dict[str, object]]]:
    kept, skipped = analyze_pdf_text_quality(docs)
    legacy_skipped = [
        {"source": item["source"], "page": item["page"], "reason": item["reason"]}
        for item in skipped
    ]
    return kept, legacy_skipped


def analyze_pdf_text_quality(docs: list[Document]) -> tuple[list[Document], list[dict[str, object]]]:
    kept: list[Document] = []
    skipped: list[dict[str, object]] = []

    for doc in docs:
        if is_noisy_pdf_text(doc.page_content):
            skipped.append(
                {
                    "source": os.path.basename(str(doc.metadata.get("source", ""))),
                    "page": doc.metadata.get("page", -1),
                    "reason": "unicode_escape_noise",
                    "quality_flags": ["unicode_escape_noise"],
                }
            )
            continue
        metadata = dict(doc.metadata)
        flags = list(metadata.get("quality_flags", []) or [])
        if "text_quality_checked" not in flags:
            flags.append("text_quality_checked")
        metadata["quality_flags"] = flags
        kept.append(Document(page_content=doc.page_content, metadata=metadata))

    return kept, skipped
