from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document

from paper_rag.retrieval.query_expansion import (
    expand_query,
    filter_query_variants,
    query_variant_embed_fn_from_hybrid,
)
from paper_rag.retrieval.router import RetrievalRouter, deduplicate_docs


@dataclass(frozen=True)
class MultiQueryRetrievalResult:
    docs: list[Document]
    variants: list[str]
    rejections: list[dict[str, Any]]


def retrieve_documents(hybrid: Any, query: str) -> list[Document]:
    retriever = hybrid.get_retriever(query)
    docs = retriever.invoke(query)
    return deduplicate_docs(docs)


def retrieve_with_hyde(
    hybrid: Any,
    question: str,
    llm_model: str,
    temperature: float,
    *,
    load_prompt_fn: Callable[[str], str] | None = None,
    get_llm_fn: Callable[[str, float], Any | None] | None = None,
    retrieve_fn: Callable[[Any, str], list[Document]] = retrieve_documents,
) -> list[Document]:
    if load_prompt_fn is None:
        from utils.prompt_loader import load_prompt

        load_prompt_fn = load_prompt
    if get_llm_fn is None:
        raise ValueError("get_llm_fn is required")

    hyde_prompt = load_prompt_fn("hyde_prompt").format(query=question)
    llm = get_llm_fn(llm_model, temperature)
    if llm is None:
        print("⚠️  LLM 不可用，降级为混合检索")
        return retrieve_fn(hybrid, question)

    try:
        hyde_response = llm.invoke(hyde_prompt)
        hyde_doc = hyde_response.content if hasattr(hyde_response, "content") else str(hyde_response)
        print(f"🧠 HyDE 生成假设性文档（{len(hyde_doc)} 字符）")
    except Exception as exc:
        print(f"⚠️  HyDE 生成失败：{exc}，降级为混合检索")
        return retrieve_fn(hybrid, question)

    docs = retrieve_fn(hybrid, hyde_doc)
    print(f"📄 HyDE 检索到 {len(docs)} 个相关文档（去重后）")
    return docs


def retrieve_multi_query(
    hybrid: Any,
    question: str,
    settings: Any,
    llm_model: str,
    temperature: float,
    *,
    llm_factory: Callable[[str, float], Any | None] | None = None,
    retrieve_fn: Callable[[Any, str], list[Document]] = retrieve_documents,
    expand_query_fn: Callable[[str, Any, int], list[str]] = expand_query,
    filter_query_variants_fn: Callable[..., Any] = filter_query_variants,
    query_variant_embed_fn: Callable[[Any], Any] = query_variant_embed_fn_from_hybrid,
) -> MultiQueryRetrievalResult:
    original_docs = retrieve_fn(hybrid, question)
    if not settings.enable_query_expansion:
        return MultiQueryRetrievalResult(docs=original_docs, variants=[], rejections=[])

    if llm_factory is None:
        raise ValueError("llm_factory is required when query expansion is enabled")

    n_variants = settings.query_expansion_variants
    expansion_model = settings.query_expansion_model or llm_model
    llm = llm_factory(expansion_model, temperature)
    try:
        variants = expand_query_fn(question, llm, n_variants=n_variants)
    except Exception as exc:
        print(f"⚠️  Query expansion 失败：{type(exc).__name__}: {exc}，仅使用原始 query")
        return MultiQueryRetrievalResult(docs=original_docs, variants=[], rejections=[])

    if not variants:
        return MultiQueryRetrievalResult(docs=original_docs, variants=[], rejections=[])

    filter_result = filter_query_variants_fn(
        question,
        variants,
        embed_fn=query_variant_embed_fn(hybrid),
        enabled=settings.enable_query_expansion_similarity_filter,
        min_similarity=settings.query_expansion_min_similarity,
        max_similarity=settings.query_expansion_max_similarity,
    )
    variants = list(filter_result.variants)
    rejections = list(filter_result.rejections)
    for item in rejections:
        variant = item.get("variant") or "<all>"
        print(f"⚠️  Query variant filtered: {item.get('reason')} · {variant}")

    if not variants:
        return MultiQueryRetrievalResult(docs=original_docs, variants=[], rejections=rejections)

    print("🔎 Query variants:")
    for index, variant in enumerate(variants, 1):
        print(f"  {index}. {variant}")

    merged_docs = list(original_docs)
    for variant in variants:
        merged_docs.extend(retrieve_fn(hybrid, variant))

    merged_docs = deduplicate_docs(merged_docs)
    max_docs = max(len(original_docs) * settings.query_expansion_max_multiplier, len(original_docs))
    return MultiQueryRetrievalResult(docs=merged_docs[:max_docs], variants=variants, rejections=rejections)


def route_retrieve(
    hybrid: Any,
    question: str,
    settings: Any,
    llm_model: str,
    temperature: float,
    *,
    llm_factory: Callable[[str, float], Any] | None = None,
    hyde_retrieve_fn: Callable[..., list[Document]] | None = None,
    multi_query_retrieve_fn: Callable[..., tuple[list[Document], list[str]]] | None = None,
    apply_rerank_fn: Callable[..., list[Document]] | None = None,
    embedding_device_fn: Callable[[], str] | None = None,
    router_cls: type = RetrievalRouter,
) -> tuple[list[Document], str]:
    router = router_cls(
        settings=settings,
        llm_factory=llm_factory,
        hyde_retrieve_fn=hyde_retrieve_fn,
        multi_query_retrieve_fn=multi_query_retrieve_fn,
        apply_rerank_fn=apply_rerank_fn,
        embedding_device_fn=embedding_device_fn,
    )
    return router.route(hybrid, question, llm_model, temperature)
