from __future__ import annotations

from typing import Any, Callable, MutableMapping


LLMCacheKey = tuple[str, float, int, int]


def select_embedding_device(torch_module: Any | None) -> str:
    if torch_module is not None and torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def build_hybrid_retriever(
    settings: Any,
    *,
    torch_module: Any | None = None,
    device: str | None = None,
    embeddings_cls: Callable[..., Any] | None = None,
    vector_store_cls: Callable[..., Any] | None = None,
    hybrid_retriever_cls: Callable[..., Any] | None = None,
) -> Any:
    if embeddings_cls is None:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        embeddings_cls = HuggingFaceBgeEmbeddings
    if vector_store_cls is None:
        from langchain_chroma import Chroma

        vector_store_cls = Chroma
    if hybrid_retriever_cls is None:
        from paper_rag.retrieval.hybrid import HybridRetriever

        hybrid_retriever_cls = HybridRetriever

    selected_device = device or select_embedding_device(torch_module)
    embeddings = embeddings_cls(
        model_name=settings.embedding_model,
        model_kwargs={"device": selected_device, "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = vector_store_cls(
        persist_directory=settings.persist_directory,
        embedding_function=embeddings,
        collection_name=settings.collection_name,
    )
    return hybrid_retriever_cls(
        vector_store=vector_store,
        top_k=settings.k,
        default_vector_weight=settings.default_vector_weight,
        default_bm25_weight=settings.default_bm25_weight,
        embedding_model=embeddings,
        persist_directory=settings.persist_directory,
        collection_name=settings.collection_name,
        chunk_schema_version=settings.chunk_schema_version,
        index_manifest_filename=settings.index_manifest_filename,
    )


def get_cached_llm(
    cache: MutableMapping[LLMCacheKey, Any | None],
    create_llm_fn: Callable[..., Any],
    *,
    model: str,
    temperature: float,
    num_ctx: int,
    num_predict: int,
    ping_prompt: str = "ping",
    on_success: Callable[[str], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> Any | None:
    key = (model, temperature, num_ctx, num_predict)
    if key in cache:
        return cache[key]

    llm = create_llm_fn(
        model=model,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )
    try:
        llm.invoke(ping_prompt)
        if on_success is not None:
            on_success(model)
        cache[key] = llm
    except Exception as exc:
        if on_error is not None:
            on_error(exc)
        cache[key] = None
    return cache[key]
