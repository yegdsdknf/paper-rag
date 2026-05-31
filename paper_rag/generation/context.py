from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from context_compression import compress_documents
from paper_rag.config import get_setting
from parent_retrieval import expand_parent_pages


def _compress_docs_for_context(question: str, docs: list[Document], settings: Any) -> list[Document]:
    """只压缩送入 prompt 的上下文，保留原始 docs 用于来源展示。"""
    if not get_setting(settings, "enable_context_compression", False):
        return docs

    max_sentences = int(get_setting(settings, "context_compression_max_sentences", 3))
    compressed_docs = compress_documents(question, docs, max_sentences=max_sentences)
    before = sum(len(doc.page_content) for doc in docs)
    after = sum(len(doc.page_content) for doc in compressed_docs)
    print(f"Context compression: {before} -> {after} chars")
    return compressed_docs


def prepare_docs_for_context(
    question: str,
    docs: list[Document],
    hybrid: Any = None,
    settings: Any = None,
) -> list[Document]:
    """生成阶段上下文增强：先补同页 parent，再做压缩；不影响返回给 UI 的 docs。"""
    settings = settings or {}
    context_docs = docs
    if hybrid is not None and get_setting(settings, "enable_parent_retrieval", False):
        max_chars = int(get_setting(settings, "parent_max_chars_per_page", 2500))
        parent_docs = expand_parent_pages(hybrid, docs, max_chars_per_parent=max_chars)
        before = sum(len(doc.page_content) for doc in docs)
        after = sum(len(doc.page_content) for doc in parent_docs)
        print(f"Parent retrieval: {len(docs)} chunks -> {len(parent_docs)} parent pages ({before} -> {after} chars)")
        context_docs = parent_docs

    return _compress_docs_for_context(question, context_docs, settings)


def build_context_stats(original_docs: list[Document], context_docs: list[Document]) -> dict[str, int]:
    return {
        "source_doc_count": len(original_docs),
        "context_doc_count": len(context_docs),
        "input_chars": sum(len(doc.page_content) for doc in original_docs),
        "output_chars": sum(len(doc.page_content) for doc in context_docs),
    }
