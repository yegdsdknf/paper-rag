from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import html
import os
import re
from typing import Any, Iterable

from langchain_core.documents import Document


_EN_STOPWORDS = {
    "about",
    "after",
    "does",
    "from",
    "have",
    "into",
    "that",
    "their",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "uses",
}
_ZH_STOP_FRAGMENTS = ("什么", "是什么", "为什么", "如何", "怎么", "请", "解释", "说明", "主要", "区别", "以及")


@dataclass(frozen=True)
class SourceViewModel:
    title: str
    source: str
    page: int | str
    score_label: str
    highlight_html: str
    raw_preview: str
    metadata: dict[str, Any]


def build_source_view_models(
    docs: Iterable[Document],
    question: str,
    standalone_question: str | None = None,
    *,
    max_sentences: int = 3,
    raw_preview_chars: int = 1200,
) -> list[SourceViewModel]:
    keywords = extract_highlight_terms(" ".join(part for part in [question, standalone_question or ""] if part))
    return [
        source_view_from_doc(
            doc,
            keywords,
            max_sentences=max_sentences,
            raw_preview_chars=raw_preview_chars,
        )
        for doc in docs
    ]


def source_view_from_doc(
    doc: Document,
    keywords: list[str],
    *,
    max_sentences: int = 3,
    raw_preview_chars: int = 1200,
) -> SourceViewModel:
    metadata = dict(doc.metadata or {})
    source = _source_filename(metadata)
    page = metadata.get("page", "?")
    title = f"{source} · p{page}"
    raw_preview = str(doc.page_content or "")[:raw_preview_chars]
    selected = select_relevant_excerpt(raw_preview, keywords, max_sentences=max_sentences)
    return SourceViewModel(
        title=title,
        source=source,
        page=page,
        score_label=_score_label(metadata.get("rerank_score")),
        highlight_html=highlight_excerpt(selected, keywords),
        raw_preview=raw_preview,
        metadata=metadata,
    )


def extract_highlight_terms(text: str) -> list[str]:
    terms: list[str] = []
    for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", text):
        lowered = word.lower()
        if lowered not in _EN_STOPWORDS:
            terms.append(word)
    for segment in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        cleaned = segment
        for fragment in _ZH_STOP_FRAGMENTS:
            cleaned = cleaned.replace(fragment, "")
        cleaned = cleaned.strip("的了呢吗么是和与或在中")
        if len(cleaned) >= 2:
            terms.append(cleaned)
    return _dedupe_longest_first(terms)


def select_relevant_excerpt(text: str, keywords: list[str], *, max_sentences: int = 3) -> str:
    sentences = _split_sentences(text)
    if not sentences or not keywords:
        return text
    scored = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        score = sum(1 for term in keywords if term.lower() in lowered)
        if score:
            scored.append((score, -index, index))
    if not scored:
        return text
    best_index = sorted(scored, reverse=True)[0][2]
    selected = sentences[best_index : best_index + max_sentences]
    return " ".join(selected)


def highlight_excerpt(text: str, keywords: list[str]) -> str:
    escaped = html.escape(text)
    for term in keywords:
        if not term:
            continue
        escaped_term = html.escape(term)
        pattern = re.compile(re.escape(escaped_term), re.IGNORECASE)
        escaped = pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", escaped)
    return escaped


def _source_filename(metadata: dict[str, Any]) -> str:
    source = str(metadata.get("source") or "").replace("\\", "/")
    filename = os.path.basename(source)
    return filename or "unknown"


def _score_label(score: Any) -> str:
    if score is None:
        return ""
    try:
        rounded = Decimal(str(score)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        return f"rerank {rounded}"
    except (InvalidOperation, TypeError, ValueError):
        return f"rerank {score}"


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def _dedupe_longest_first(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for term in sorted(terms, key=len, reverse=True):
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(term)
    return unique
