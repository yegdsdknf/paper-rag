from __future__ import annotations

import os
import re
from typing import Any, Callable

from langchain_core.documents import Document

from paper_rag.config import get_setting
from paper_rag.retrieval.source_resolver import (
    SourceResolver,
    normalize_source_signal,
    source_stem_signal,
)
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


def _source_stem_signal(doc: Document) -> str:
    source = os.path.basename(str(doc.metadata.get("source", "")))
    return source_stem_signal(source)


def _filter_docs_to_mentioned_sources(question: str, docs: list[Document]) -> list[Document]:
    """证据定位题优先保留题目中点名的论文，减少通用 Transformer 页干扰。"""
    question_signal = normalize_source_signal(question)
    matched_sources = {
        signal
        for doc in docs
        if (signal := _source_stem_signal(doc)) and signal in question_signal
    }

    if not matched_sources:
        return docs

    return [doc for doc in docs if _source_stem_signal(doc) in matched_sources]


def _filter_docs_to_source_files(docs: list[Document], source_files: list[str]) -> list[Document]:
    if not source_files:
        return docs
    wanted = {normalize_source_signal(source) for source in source_files}
    filtered = [doc for doc in docs if normalize_source_signal(os.path.basename(str(doc.metadata.get("source", "")))) in wanted]
    return filtered or docs


def _source_page_docs(hybrid: Any, source_files: list[str]) -> list[Document]:
    vector_store = getattr(hybrid, "vector_store", None)
    if vector_store is None or not source_files:
        return []

    try:
        stored = vector_store.get(include=["documents", "metadatas"])
    except Exception as exc:
        print(f"Source evidence load failed: {type(exc).__name__}: {exc}")
        return []

    wanted = {normalize_source_signal(source) for source in source_files}
    docs = []
    for content, metadata in zip(stored.get("documents", []), stored.get("metadatas", [])):
        source = os.path.basename(str(metadata.get("source", "")))
        if normalize_source_signal(source) not in wanted:
            continue
        docs.append(Document(page_content=str(content), metadata=dict(metadata)))
    return docs


def _expanded_term_set(seed_terms: set[str]) -> set[str]:
    expansions = {
        "few-shot": {"few-shot", "few shot", "fewshot", "demonstration", "demonstrations", "text interaction", "in-context"},
        "zero-shot": {"zero-shot", "zero shot", "zeroshot", "without examples"},
        "one-shot": {"one-shot", "one shot", "oneshot"},
        "prompt": {"prompt", "prompting", "text interaction"},
        "parameter": {"parameter", "parameters", "billion parameters"},
        "corpus": {"corpus", "dataset", "data set"},
        "pre-training": {"pre-training", "pretraining", "pretrain", "objective", "objectives"},
        "text-to-text": {"text-to-text", "text to text"},
        "architecture": {"architecture", "architectures", "architectural", "transformer", "layer", "layers", "model and architecture", "attention patterns"},
    }
    terms = set(seed_terms)
    for term in list(seed_terms):
        terms.update(expansions.get(term, set()))
        if "-" in term:
            terms.add(term.replace("-", " "))
    return {term.lower() for term in terms if len(term.strip()) >= 3}


def _evidence_terms(question: str, front_page_text: str) -> set[str]:
    q_lower = question.lower()
    seed_terms: set[str] = set()
    signal_terms = [
        ("few-shot", ["few-shot", "few shot", "少样本"]),
        ("zero-shot", ["zero-shot", "zero shot", "零样本"]),
        ("one-shot", ["one-shot", "one shot"]),
        ("prompt", ["prompt", "提示"]),
        ("parameter", ["参数", "parameter"]),
        ("corpus", ["语料", "corpus", "dataset", "数据集"]),
        ("pre-training", ["预训练", "pretrain", "pre-training", "objective"]),
        ("text-to-text", ["text-to-text", "统一框架"]),
        ("architecture", ["架构", "结构", "architecture", "architectures", "architectural", "transformer"]),
    ]
    for term, signals in signal_terms:
        if any(signal in q_lower for signal in signals):
            seed_terms.add(term)

    summary_signals = ["核心特点", "总结", "summary", "overview", "key feature", "key features"]
    if any(signal in q_lower for signal in summary_signals):
        front_lower = front_page_text.lower()
        seed_terms.update(re.findall(r"\b[a-z]+(?:-[a-z]+)+\b", front_lower))
        for term, _signals in signal_terms:
            if term in front_lower:
                seed_terms.add(term)

    return _expanded_term_set(seed_terms)


def get_source_evidence_docs(
    hybrid: Any,
    question: str,
    source_files: list[str],
    max_docs: int = 4,
) -> list[Document]:
    docs = _source_page_docs(hybrid, source_files)
    if not docs:
        return []

    front_page_text = " ".join(doc.page_content for doc in docs if int(doc.metadata.get("page") or 0) == 0)[:4000]
    terms = _evidence_terms(question, front_page_text)
    if not terms:
        return []

    scored: list[tuple[int, int, Document]] = []
    for doc in docs:
        text = doc.page_content.lower()
        score = sum(1 for term in terms if term in text)
        if score <= 0:
            continue
        page = int(doc.metadata.get("page") or 0)
        scored.append((score, page, doc))

    selected = [doc for _score, _page, doc in sorted(scored, key=lambda item: (-item[0], item[1]))]
    return deduplicate_docs(selected)[:max_docs]


def _source_resolver(hybrid: Any, settings: Any | None = None) -> SourceResolver:
    return SourceResolver.from_hybrid(hybrid, settings=settings)


def mentioned_source_files(question: str, hybrid: Any | None = None, settings: Any | None = None) -> list[str]:
    return _source_resolver(hybrid, settings).resolve_source_files(question)


def _source_files_for_question(question: str, hybrid: Any, settings: Any | None = None) -> list[str]:
    return mentioned_source_files(question, hybrid, settings)


def _is_generic_transformer_origin_question(question: str, hybrid: Any, settings: Any | None = None) -> bool:
    resolver = _source_resolver(hybrid, settings)
    return not resolver.explicit_source_matches(question) and bool(resolver.resolve_origin_source(question))


def _source_anchor_pages(question: str, hybrid: Any, settings: Any | None = None) -> list[int]:
    if _is_generic_transformer_origin_question(question, hybrid, settings):
        return [0]
    q_lower = question.lower()
    multi_evidence_signals = [
        "两个", "哪些", "为什么", "训练", "预训练", "任务", "思路", "方法",
        "why", "how", "training", "pretrain", "method", "objective",
    ]
    if any(signal in q_lower for signal in multi_evidence_signals):
        return [0, 1, 2, 3]
    return [0]


def get_source_anchor_docs(hybrid: Any, question: str, source_files: list[str] | None = None, settings: Any | None = None) -> list[Document]:
    sources = source_files or _source_files_for_question(question, hybrid, settings)
    if not sources:
        return []
    return load_anchor_docs_by_page(hybrid, sources, _source_anchor_pages(question, hybrid, settings))


def load_anchor_docs_by_page(hybrid: Any, source_files: list[str], pages: list[int]) -> list[Document]:
    vector_store = getattr(hybrid, "vector_store", None)
    if vector_store is None or not source_files:
        return []

    try:
        stored = vector_store.get(include=["documents", "metadatas"])
    except Exception as exc:
        print(f"Compare anchor load failed: {type(exc).__name__}: {exc}")
        return []

    wanted_sources = {normalize_source_signal(source): source for source in source_files}
    wanted_pages = set(pages)
    selected: dict[tuple[str, int], Document] = {}
    for content, metadata in zip(stored.get("documents", []), stored.get("metadatas", [])):
        source = os.path.basename(str(metadata.get("source", "")))
        source_key = normalize_source_signal(source)
        page = metadata.get("page")
        if source_key not in wanted_sources or page not in wanted_pages:
            continue
        key = (wanted_sources[source_key], int(page))
        if key not in selected:
            selected[key] = Document(page_content=content, metadata=dict(metadata))

    return [selected[key] for key in [(source, page) for source in source_files for page in pages] if key in selected]


def get_compare_anchor_docs(hybrid: Any, question: str, settings: Any | None = None) -> list[Document]:
    source_files = mentioned_source_files(question, hybrid, settings)
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
        source_files = _source_files_for_question(question, hybrid, self.settings)
        if is_comparison or is_overview_question(question) or is_evidence_question(question) or source_files:
            print("Using standard mixed retrieval")
            docs, variants = self.retrieve_multi_query(hybrid, question, llm_model, temperature)
            if is_evidence_question(question):
                docs = _filter_docs_to_mentioned_sources(question, docs)
                source_files = mentioned_source_files(question, hybrid, self.settings)
                anchors = load_anchor_docs_by_page(hybrid, source_files, [0])
                evidence_docs = get_source_evidence_docs(hybrid, question, source_files)
                docs = deduplicate_docs(anchors + evidence_docs + docs)[:get_setting(self.settings, "rerank_top_k", len(docs))]
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
                docs = deduplicate_docs(get_compare_anchor_docs(hybrid, question, self.settings) + docs)[:rerank_top_k]
            elif source_files:
                docs = _filter_docs_to_source_files(docs, source_files)
                docs = deduplicate_docs(
                    get_source_anchor_docs(hybrid, question, source_files, self.settings)
                    + get_source_evidence_docs(hybrid, question, source_files)
                    + docs
                )[:rerank_top_k]
            strategy = "mixed_multi_query" if variants else "mixed"
            return docs, strategy

        print("Using HyDE retrieval")
        if self.hyde_retrieve_fn is None:
            return [], "hyde"
        docs = self.hyde_retrieve_fn(hybrid, question, llm_model, temperature)
        return docs, "hyde"
