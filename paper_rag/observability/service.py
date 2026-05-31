from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from paper_rag.config import get_setting
from paper_rag.indexing import resolve_index_version
from paper_rag.observability.query_logger import build_query_log_record, save_query_log_record


def _feature_flags(settings: Any) -> dict[str, bool]:
    return {
        "rerank": bool(get_setting(settings, "enable_rerank", False)),
        "query_expansion": bool(get_setting(settings, "enable_query_expansion", False)),
        "context_compression": bool(get_setting(settings, "enable_context_compression", False)),
        "parent_retrieval": bool(get_setting(settings, "enable_parent_retrieval", False)),
    }


def write_query_log(
    settings: Any,
    question: str,
    standalone_question: str,
    route: str,
    llm_model: str,
    docs: list[Any],
    elapsed: dict[str, float],
    embedding_device_fn: Callable[[], str],
    context_stats: dict[str, Any] | None = None,
    error: str | None = None,
) -> Path | None:
    """按配置写入查询日志；关闭日志时不产生副作用。"""
    if not get_setting(settings, "enable_query_logging", False):
        return None

    record = build_query_log_record(
        question=question,
        standalone_question=standalone_question,
        route=route,
        llm_model=llm_model,
        embedding_device=embedding_device_fn(),
        index_version=resolve_index_version(settings),
        feature_flags=_feature_flags(settings),
        docs=docs,
        elapsed=elapsed,
        context_stats=context_stats,
        error=error,
    )
    return save_query_log_record(record, get_setting(settings, "query_log_path", "logs/query_runs.jsonl"))
