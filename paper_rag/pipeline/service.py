from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ReformulationResult:
    standalone_question: str
    rewritten: bool


def reformulate_question(
    conversation: Any,
    question: str,
    *,
    require_history: bool,
) -> ReformulationResult:
    """统一多轮改写决策，保留非流式和流式入口的历史判断差异。"""
    if require_history and not getattr(conversation, "history", []):
        return ReformulationResult(standalone_question=question, rewritten=False)

    standalone_question = conversation.reformulate(question)
    return ReformulationResult(
        standalone_question=standalone_question,
        rewritten=standalone_question != question,
    )


def write_pipeline_query_log(
    *,
    settings: Any,
    question: str,
    standalone_question: str,
    route: str,
    llm_model: str,
    docs: list[Any],
    elapsed: dict[str, float],
    embedding_device_fn: Callable[[], str] | Any,
    query_trace: dict[str, list[Any]] | None = None,
    context_stats: dict[str, Any] | None = None,
    error: str | None = None,
    write_query_log_fn: Callable[..., None] | None = None,
) -> None:
    """统一 query log trace 选择，避免入口层重复判断 route。"""
    if write_query_log_fn is None:
        from paper_rag.observability.service import write_query_log

        write_query_log_fn = write_query_log

    trace = query_trace if route.startswith("mixed") and query_trace else {"variants": [], "rejections": []}
    write_query_log_fn(
        settings=settings,
        question=question,
        standalone_question=standalone_question,
        route=route,
        llm_model=llm_model,
        docs=docs,
        elapsed=elapsed,
        embedding_device_fn=embedding_device_fn,
        query_variants=trace["variants"],
        query_variant_rejections=trace["rejections"],
        context_stats=context_stats,
        error=error,
    )
