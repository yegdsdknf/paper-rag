"""兼容薄壳：新代码请从 paper_rag.retrieval.reranker 导入。"""

from langchain_core.documents import Document

from paper_rag.retrieval.reranker import Reranker, get_reranker as _get_reranker


get_reranker = _get_reranker


def apply_rerank(
    query: str,
    docs: list[Document],
    enabled: bool,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_k: int | None = None,
    device: str | None = None,
) -> list[Document]:
    if not enabled:
        return docs

    try:
        reranker = get_reranker(model_name=model_name, device=device)
        return reranker.rerank(query, docs, top_k=top_k)
    except Exception as exc:
        print(f"[WARN] Rerank unavailable; keeping original order: {type(exc).__name__}: {exc}")
        return docs


__all__ = ["Reranker", "apply_rerank", "get_reranker"]
