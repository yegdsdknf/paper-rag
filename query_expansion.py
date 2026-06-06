"""兼容薄壳：新代码请从 paper_rag.retrieval.query_expansion 导入。"""

__compat_replacement__ = "paper_rag.retrieval.query_expansion"

from paper_rag.retrieval.query_expansion import (
    QueryVariantFilterResult,
    expand_query,
    filter_query_variants,
    query_variant_embed_fn_from_hybrid,
)

__all__ = [
    "QueryVariantFilterResult",
    "expand_query",
    "filter_query_variants",
    "query_variant_embed_fn_from_hybrid",
]
