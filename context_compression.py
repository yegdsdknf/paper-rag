from __future__ import annotations

import re
from collections import Counter
from typing import Any

from langchain_core.documents import Document


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "based",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "with",
    "什么",
    "如何",
    "是否",
    "这个",
    "论文",
    "模型",
}


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    parts = re.split(r"(?<=[。！？!?\.])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _query_terms(query: str) -> Counter[str]:
    tokens = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9\-]{1,}|[\u4e00-\u9fff]{2,}", query.lower())
    return Counter(token for token in tokens if token not in _STOPWORDS)


def _sentence_score(sentence: str, terms: Counter[str], index: int = 0) -> float:
    if not terms:
        return 0.0

    lowered = sentence.lower()
    term_score = 0
    distinct_hits = 0
    for term, weight in terms.items():
        if term in lowered:
            term_score += weight
            distinct_hits += 1
    if term_score <= 0:
        return 0.0

    coverage_bonus = 1.0 + (distinct_hits / max(len(terms), 1))
    length_factor = _length_factor(len(sentence))
    position_bonus = 0.12 / (index + 1)
    return float(term_score * coverage_bonus * length_factor + position_bonus)


def _length_factor(length: int) -> float:
    if length < 16:
        return 0.75
    if length <= 160:
        return 1.15
    if length <= 260:
        return 0.85
    return 0.55


def compress_chunk(query: str, chunk_text: str, max_sentences: int = 3) -> str:
    """抽取与问题最相关的证据句；无可靠命中时保留原 chunk。"""
    sentences = _split_sentences(chunk_text)
    if not sentences or max_sentences <= 0:
        return chunk_text

    terms = _query_terms(query)
    scored = [
        (score, index, sentence)
        for index, sentence in enumerate(sentences)
        if (score := _sentence_score(sentence, terms, index)) > 0
    ]
    if not scored:
        return chunk_text

    selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
    selected_in_order = sorted(selected, key=lambda item: item[1])
    return " ".join(sentence for _, _, sentence in selected_in_order)


def compress_documents(
    query: str,
    docs: list[Document],
    max_sentences: int = 3,
) -> list[Document]:
    compressed_docs: list[Document] = []
    for doc in docs:
        if _should_preserve_doc(doc):
            compressed_text = doc.page_content
        else:
            compressed_text = compress_chunk(query, doc.page_content, max_sentences=max_sentences)
        metadata = dict(doc.metadata)
        metadata["context_original_chars"] = len(doc.page_content)
        metadata["context_compressed_chars"] = len(compressed_text)
        metadata["context_compressed"] = compressed_text != doc.page_content
        compressed_docs.append(Document(page_content=compressed_text, metadata=metadata))
    return compressed_docs


def _is_vision_summary(doc: Document) -> bool:
    return (
        doc.metadata.get("block_type") == "vision_summary"
        or doc.metadata.get("chunk_strategy") == "vision_summary"
    )


def _should_preserve_doc(doc: Document) -> bool:
    if _is_vision_summary(doc):
        return True
    metadata = doc.metadata or {}
    if _is_abstract(metadata):
        return True
    if _is_structural_evidence(metadata, doc.page_content):
        return True
    return False


def _is_abstract(metadata: dict[str, Any]) -> bool:
    paper_region = str(metadata.get("paper_region", "")).lower()
    section_title = str(metadata.get("section_title", "")).lower()
    return paper_region == "abstract" or section_title in {"abstract", "摘要"}


def _is_structural_evidence(metadata: dict[str, Any], text: str) -> bool:
    block_type = str(metadata.get("block_type", "")).lower()
    chunk_strategy = str(metadata.get("chunk_strategy", "")).lower()
    if block_type in {"table", "equation", "formula"} or chunk_strategy in {"table", "equation", "formula"}:
        return True

    head = text.strip()[:160].lower()
    if re.match(r"^(table|figure|fig\.|equation)\s*[\divx]+", head):
        return True
    formula_chars = "√²³ᵀ⁻⁺→∑∏∫∂±="
    return sum(1 for char in text if char in formula_chars) >= 2
