"""Compatibility metadata for root-level wrapper modules."""

from __future__ import annotations

from types import MappingProxyType


COMPAT_WRAPPER_REPLACEMENTS = MappingProxyType(
    {
        "app_services": "paper_rag.ui.services",
        "app_state": "paper_rag.ui.state",
        "context_builder": "paper_rag.generation.context",
        "context_compression": "paper_rag.generation.context_compression",
        "feedback": "paper_rag.observability.feedback",
        "generation_service": "paper_rag.generation.service",
        "hybrid_retriever": "paper_rag.retrieval.hybrid",
        "parent_retrieval": "paper_rag.generation.parent_retrieval",
        "query_expansion": "paper_rag.retrieval.query_expansion",
        "query_logger": "paper_rag.observability.query_logger",
        "reranker": "paper_rag.retrieval.reranker",
        "retrieval_router": "paper_rag.retrieval.router",
        "source_utils": "paper_rag.observability.sources",
    }
)


def _build_retirement_policy() -> MappingProxyType:
    return MappingProxyType(
        {
            module_name: MappingProxyType(
                {
                    "replacement": replacement,
                    "stage": "keep_compat_wrapper",
                    "allowed_internal_imports": ("tests",),
                    "next_action": "audit external imports before warning or removal",
                }
            )
            for module_name, replacement in COMPAT_WRAPPER_REPLACEMENTS.items()
        }
    )


COMPAT_WRAPPER_RETIREMENT_POLICY = _build_retirement_policy()

__all__ = ["COMPAT_WRAPPER_REPLACEMENTS", "COMPAT_WRAPPER_RETIREMENT_POLICY"]
