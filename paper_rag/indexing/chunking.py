from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paper_rag.config import get_setting
from paper_rag.indexing.metadata import attach_chunk_metadata, infer_paper_region


_KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experiment",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "references",
    "bibliography",
    "appendix",
    "摘要",
    "引言",
    "绪论",
    "相关工作",
    "方法",
    "实验",
    "结果",
    "讨论",
    "结论",
    "参考文献",
    "附录",
}


@dataclass(frozen=True)
class _Section:
    title: str | None
    level: int
    index: int
    text: str


def _recursive_split(documents: list[Document], settings: Any) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(get_setting(settings, "chunk_size", 500)),
        chunk_overlap=int(get_setting(settings, "chunk_overlap", 100)),
        separators=list(get_setting(settings, "separators", ["\n\n", "\n", " ", ""])),
        length_function=len,
    )
    return splitter.split_documents(documents)


def _heading_level(line: str) -> int:
    stripped = line.strip()
    if re.match(r"^\d+(?:\.\d+)*\.?\s+\S+", stripped):
        return stripped.split()[0].count(".") + 1
    if re.match(r"^[IVXLC]+\.?\s+\S+", stripped, flags=re.IGNORECASE):
        return 1
    return 1


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    if stripped.endswith((".", "。", ",", "，", ";", "；")):
        return False

    lowered = stripped.lower().strip(":")
    if lowered in _KNOWN_HEADINGS:
        return True
    if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z\u4e00-\u9fff][^\n]{1,100}$", stripped):
        return True
    if re.match(r"^[IVXLC]+\.?\s+[A-Z][^\n]{1,100}$", stripped, flags=re.IGNORECASE):
        return True
    if re.match(r"^[\u4e00-\u9fff]{2,10}$", stripped) and stripped in _KNOWN_HEADINGS:
        return True
    return False


def _sections_from_document(doc: Document, detect_headings: bool) -> list[_Section]:
    lines = doc.page_content.splitlines()
    sections: list[_Section] = []
    current_title: str | None = None
    current_level = 0
    current_lines: list[str] = []
    section_index = 0

    def flush() -> None:
        nonlocal section_index, current_lines
        text = "\n".join(line for line in current_lines).strip()
        if not text:
            return
        sections.append(_Section(current_title, current_level, section_index, text))
        section_index += 1
        current_lines = []

    for line in lines:
        if detect_headings and _is_heading(line):
            flush()
            current_title = line.strip()
            current_level = _heading_level(current_title)
            current_lines = [current_title]
            continue
        current_lines.append(line)

    flush()
    if not sections and doc.page_content.strip():
        return [_Section(None, 0, 0, doc.page_content.strip())]
    return sections


def _section_metadata(base: dict[str, Any], section: _Section) -> dict[str, Any]:
    metadata = dict(base)
    metadata["section_title"] = section.title
    metadata["section_level"] = section.level
    metadata["section_index"] = section.index
    metadata["paper_region"] = infer_paper_region(section.title, metadata.get("paper_region"))
    return metadata


def _section_aware_split(documents: list[Document], settings: Any) -> list[Document]:
    max_chars = int(get_setting(settings, "section_max_chars", 900))
    detect_headings = bool(get_setting(settings, "section_heading_detection", True))
    chunks: list[Document] = []

    for doc in documents:
        for section in _sections_from_document(doc, detect_headings):
            section_doc = Document(
                page_content=section.text,
                metadata=_section_metadata(doc.metadata, section),
            )
            if len(section.text) <= max_chars:
                chunks.append(section_doc)
            else:
                chunks.extend(_recursive_split([section_doc], settings))

    return chunks


def _split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[。！？!?\.])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b + 1e-8)


def _semantic_split_document(doc: Document, settings: Any, embeddings: Any) -> list[Document]:
    sentences = _split_sentences(doc.page_content)
    max_chars = int(get_setting(settings, "semantic_max_chars", 900))
    if len(sentences) <= 1:
        if len(doc.page_content) <= max_chars:
            return [doc]
        return _recursive_split([doc], settings)

    vectors = embeddings.embed_documents(sentences)
    threshold = float(get_setting(settings, "semantic_similarity_threshold", 0.7))
    min_chars = int(get_setting(settings, "semantic_min_chars", 180))
    chunks: list[str] = []
    current: list[str] = [sentences[0]]

    for index in range(1, len(sentences)):
        candidate = sentences[index]
        similarity = _cosine(list(vectors[index - 1]), list(vectors[index]))
        current_text = " ".join(current)
        should_break = similarity < threshold and len(current_text) >= min_chars
        would_exceed = len(current_text) + 1 + len(candidate) > max_chars
        if should_break or would_exceed:
            chunks.append(current_text)
            current = [candidate]
        else:
            current.append(candidate)

    if current:
        chunks.append(" ".join(current))

    result = []
    for text in chunks:
        metadata = dict(doc.metadata)
        metadata["semantic_sentence_count"] = len(_split_sentences(text))
        chunk_doc = Document(page_content=text, metadata=metadata)
        if len(text) > max_chars:
            result.extend(_recursive_split([chunk_doc], settings))
        else:
            result.append(chunk_doc)
    return result


def _semantic_split(documents: list[Document], settings: Any, embeddings: Any | None) -> list[Document]:
    if embeddings is None:
        raise ValueError("semantic chunking requires embeddings")
    chunks: list[Document] = []
    for doc in documents:
        chunks.extend(_semantic_split_document(doc, settings, embeddings))
    return chunks


def split_documents(
    documents: list[Document],
    settings: Any,
    embeddings: Any | None = None,
    source_file_hashes: dict[str, str] | None = None,
) -> list[Document]:
    strategy = str(get_setting(settings, "chunk_strategy", "recursive_character"))
    schema_version = str(get_setting(settings, "chunk_schema_version", "v1"))

    if strategy == "recursive_character":
        chunks = _recursive_split(documents, settings)
    elif strategy == "section_aware":
        chunks = _section_aware_split(documents, settings)
    elif strategy == "semantic":
        chunks = _semantic_split(documents, settings, embeddings)
    elif strategy == "hybrid_section_semantic":
        section_chunks = _section_aware_split(documents, settings)
        chunks = _semantic_split(section_chunks, settings, embeddings)
    else:
        raise ValueError(f"Unsupported chunk_strategy: {strategy}")

    return attach_chunk_metadata(
        chunks,
        strategy=strategy,
        schema_version=schema_version,
        source_file_hashes=source_file_hashes,
    )
