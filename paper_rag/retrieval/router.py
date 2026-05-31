from __future__ import annotations

import os
import re
from typing import Any, Callable

from langchain_core.documents import Document

from paper_rag.config import get_setting
from query_expansion import expand_query


def deduplicate_docs(docs: list[Document]) -> list[Document]:
    """同一来源同一页只保留一个 chunk，减少重复 token 消耗。"""
    seen = set()
    unique = []
    for doc in docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", -1))
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def is_comparison_question(question: str) -> bool:
    comparison_signals = [
        "vs", "versus", "difference", "differences",
        "compare", "comparison", "between", "similarity", "similarities", "common",
        "不同", "比较", "相比", "之间", "对比", "区别", "差别", "差异", "共同点", "相同", "共性",
    ]
    q_lower = question.lower()
    return any(sig in q_lower for sig in comparison_signals)


def is_overview_question(question: str) -> bool:
    overview_signals = [
        "是什么", "什么是", "介绍", "简介", "定义",
        "what is", "definition", "overview", "introduction",
    ]
    q_lower = question.lower()
    return any(sig in q_lower for sig in overview_signals)


def is_evidence_question(question: str) -> bool:
    evidence_signals = [
        "证据", "依据", "在哪一页", "哪一页", "页码", "原文",
        "evidence", "which page", "page number", "where does", "quote",
    ]
    q_lower = question.lower()
    return any(sig in q_lower for sig in evidence_signals)


def _normalize_source_signal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _source_stem_signal(doc: Document) -> str:
    source = os.path.basename(str(doc.metadata.get("source", "")))
    return _normalize_source_signal(os.path.splitext(source)[0])


def _filter_docs_to_mentioned_sources(question: str, docs: list[Document]) -> list[Document]:
    """证据定位题优先保留题目中点名的论文，减少通用 Transformer 页干扰。"""
    question_signal = _normalize_source_signal(question)
    matched_sources = {
        signal
        for doc in docs
        if (signal := _source_stem_signal(doc)) and signal in question_signal
    }

    if not matched_sources:
        return docs

    return [doc for doc in docs if _source_stem_signal(doc) in matched_sources]


SOURCE_ALIASES = [
    ("bert.pdf", ["bert"]),
    ("gpt3.pdf", ["gpt-3", "gpt3"]),
    ("t5.pdf", ["t5"]),
    ("vit.pdf", ["vit", "vision transformer"]),
    ("deepseekr1.pdf", ["deepseek-r1", "deepseekr1", "deepseek"]),
    ("attention is all you need.pdf", ["attention is all you need"]),
]


def mentioned_source_files(question: str) -> list[str]:
    q_lower = question.lower()
    matches: list[tuple[int, str]] = []
    for source, aliases in SOURCE_ALIASES:
        positions = [q_lower.find(alias) for alias in aliases if q_lower.find(alias) >= 0]
        if positions:
            matches.append((min(positions), source))

    ordered = [source for _, source in sorted(matches, key=lambda item: item[0])]
    commonality_signals = ["共同点", "相同", "共性", "similarity", "similarities", "common"]
    if (
        len(ordered) >= 2
        and "attention is all you need.pdf" not in ordered
        and any(signal in q_lower for signal in commonality_signals)
    ):
        ordered.append("attention is all you need.pdf")
    return ordered


def load_anchor_docs_by_page(hybrid: Any, source_files: list[str], pages: list[int]) -> list[Document]:
    vector_store = getattr(hybrid, "vector_store", None)
    if vector_store is None or not source_files:
        return []

    try:
        stored = vector_store.get(include=["documents", "metadatas"])
    except Exception as exc:
        print(f"Compare anchor load failed: {type(exc).__name__}: {exc}")
        return []

    wanted_sources = {_normalize_source_signal(source): source for source in source_files}
    wanted_pages = set(pages)
    selected: dict[tuple[str, int], Document] = {}
    for content, metadata in zip(stored.get("documents", []), stored.get("metadatas", [])):
        source = os.path.basename(str(metadata.get("source", "")))
        source_key = _normalize_source_signal(source)
        page = metadata.get("page")
        if source_key not in wanted_sources or page not in wanted_pages:
            continue
        key = (wanted_sources[source_key], int(page))
        if key not in selected:
            selected[key] = Document(page_content=content, metadata=dict(metadata))

    return [selected[key] for key in [(source, page) for source in source_files for page in pages] if key in selected]


def get_compare_anchor_docs(hybrid: Any, question: str) -> list[Document]:
    source_files = mentioned_source_files(question)
    if not source_files:
        return []

    front_pages = load_anchor_docs_by_page(hybrid, source_files, [0])
    explicit_sources = [source for source in source_files if source != "attention is all you need.pdf"]
    early_pages = load_anchor_docs_by_page(hybrid, explicit_sources, [1, 2, 3])
    return deduplicate_docs(front_pages + early_pages)


class RetrievalRouter:
    def __init__(
        self,
        settings: Any,
        llm_factory: Callable[[str, float], Any] | None = None,
        hyde_retrieve_fn: Callable[[Any, str, str, float], list[Document]] | None = None,
        multi_query_retrieve_fn: Callable[[Any, str, str, float], tuple[list[Document], list[str]]] | None = None,
        apply_rerank_fn: Callable[..., list[Document]] | None = None,
        embedding_device_fn: Callable[[], str] | None = None,
    ):
        self.settings = settings
        self.llm_factory = llm_factory
        self.hyde_retrieve_fn = hyde_retrieve_fn
        self.multi_query_retrieve_fn = multi_query_retrieve_fn
        self.apply_rerank_fn = apply_rerank_fn
        self.embedding_device_fn = embedding_device_fn or (lambda: "cpu")

    def retrieve(self, hybrid: Any, query: str) -> list[Document]:
        retriever = hybrid.get_retriever(query)
        docs = retriever.invoke(query)
        return deduplicate_docs(docs)

    def retrieve_multi_query(
        self,
        hybrid: Any,
        question: str,
        llm_model: str,
        temperature: float,
    ) -> tuple[list[Document], list[str]]:
        if self.multi_query_retrieve_fn is not None:
            return self.multi_query_retrieve_fn(hybrid, question, llm_model, temperature)

        original_docs = self.retrieve(hybrid, question)
        if not get_setting(self.settings, "enable_query_expansion", False):
            return original_docs, []

        if self.llm_factory is None:
            return original_docs, []

        n_variants = int(get_setting(self.settings, "query_expansion_variants", 2))
        expansion_model = get_setting(self.settings, "query_expansion_model", llm_model)
        llm = self.llm_factory(expansion_model, temperature)
        try:
            variants = expand_query(question, llm, n_variants=n_variants)
        except Exception as exc:
            print(f"Query expansion failed: {type(exc).__name__}: {exc}; using original query only")
            return original_docs, []

        if not variants:
            return original_docs, []

        print("Query variants:")
        for index, variant in enumerate(variants, 1):
            print(f"  {index}. {variant}")

        merged_docs = list(original_docs)
        for variant in variants:
            merged_docs.extend(self.retrieve(hybrid, variant))

        merged_docs = deduplicate_docs(merged_docs)
        max_multiplier = int(get_setting(self.settings, "query_expansion_max_multiplier", n_variants + 1))
        max_docs = max(len(original_docs) * max_multiplier, len(original_docs))
        return merged_docs[:max_docs], variants

    def route(
        self,
        hybrid: Any,
        question: str,
        llm_model: str = "",
        temperature: float = 0.0,
    ) -> tuple[list[Document], str]:
        is_comparison = is_comparison_question(question)
        if is_comparison or is_overview_question(question) or is_evidence_question(question):
            print("Using standard mixed retrieval")
            docs, variants = self.retrieve_multi_query(hybrid, question, llm_model, temperature)
            if is_evidence_question(question):
                docs = _filter_docs_to_mentioned_sources(question, docs)
                source_files = mentioned_source_files(question)
                anchors = load_anchor_docs_by_page(hybrid, source_files, [0])
                docs = deduplicate_docs(anchors + docs)[:get_setting(self.settings, "rerank_top_k", len(docs))]
                strategy = "mixed_multi_query" if variants else "mixed"
                return docs, strategy

            rerank_top_k = get_setting(self.settings, "rerank_top_k", get_setting(self.settings, "k", len(docs)))
            if self.apply_rerank_fn is not None:
                docs = self.apply_rerank_fn(
                    question,
                    docs,
                    enabled=get_setting(self.settings, "enable_rerank", False),
                    model_name=get_setting(self.settings, "reranker_model", "BAAI/bge-reranker-v2-m3"),
                    top_k=rerank_top_k,
                    device=self.embedding_device_fn(),
                )
            if is_comparison:
                docs = deduplicate_docs(get_compare_anchor_docs(hybrid, question) + docs)[:rerank_top_k]
            strategy = "mixed_multi_query" if variants else "mixed"
            return docs, strategy

        print("Using HyDE retrieval")
        if self.hyde_retrieve_fn is None:
            return [], "hyde"
        docs = self.hyde_retrieve_fn(hybrid, question, llm_model, temperature)
        return docs, "hyde"
