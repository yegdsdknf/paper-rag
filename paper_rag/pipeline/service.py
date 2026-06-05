from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


NO_DOCS_MESSAGE = "❌ 未找到相关内容"
LLM_STREAM_DISCONNECTED_MESSAGE = "❌ LLM 模型未连接"
LLM_DISCONNECTED_ERROR = "LLM 模型未连接"


@dataclass(frozen=True)
class ReformulationResult:
    standalone_question: str
    rewritten: bool


@dataclass(frozen=True)
class PipelineContext:
    context_docs: list[Any]
    context_stats: dict[str, Any]


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


def prepare_pipeline_context(
    *,
    question: str,
    docs: list[Any],
    hybrid: Any,
    settings: Any,
    prepare_docs_fn: Callable[..., list[Any]] | None = None,
    build_stats_fn: Callable[[list[Any], list[Any]], dict[str, Any]] | None = None,
) -> PipelineContext:
    """准备生成上下文及其统计信息，供非流式和流式入口复用。"""
    if prepare_docs_fn is None:
        from paper_rag.generation.context import prepare_docs_for_context

        prepare_docs_fn = prepare_docs_for_context
    if build_stats_fn is None:
        from paper_rag.generation.context import build_context_stats

        build_stats_fn = build_context_stats

    context_docs = prepare_docs_fn(question, docs, hybrid=hybrid, settings=settings)
    return PipelineContext(
        context_docs=context_docs,
        context_stats=build_stats_fn(docs, context_docs),
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


def handle_no_docs_response(
    *,
    question: str,
    standalone_question: str,
    route: str,
    llm_model: str,
    elapsed: dict[str, float],
    stream: bool = False,
    write_query_log_fn: Callable[..., None],
) -> tuple[str, list[Any]] | list[dict[str, str]]:
    """统一无检索结果时的日志和返回值，保留入口层事件顺序。"""
    write_query_log_fn(
        question=question,
        standalone_question=standalone_question,
        route=route,
        llm_model=llm_model,
        docs=[],
        elapsed=elapsed,
    )
    if stream:
        return [{"type": "token", "data": NO_DOCS_MESSAGE}]
    return NO_DOCS_MESSAGE, []


def handle_llm_unavailable_response(
    *,
    question: str,
    standalone_question: str,
    route: str,
    llm_model: str,
    docs: list[Any],
    elapsed: dict[str, float],
    context_stats: dict[str, Any],
    write_query_log_fn: Callable[..., None],
) -> list[dict[str, str]]:
    """统一流式生成中 LLM 不可用时的错误日志和 token 事件。"""
    write_query_log_fn(
        question=question,
        standalone_question=standalone_question,
        route=route,
        llm_model=llm_model,
        docs=docs,
        elapsed=elapsed,
        context_stats=context_stats,
        error=LLM_DISCONNECTED_ERROR,
    )
    return [{"type": "token", "data": LLM_STREAM_DISCONNECTED_MESSAGE}]


def write_successful_response_log(
    *,
    question: str,
    standalone_question: str,
    route: str,
    llm_model: str,
    docs: list[Any],
    elapsed: dict[str, float],
    context_stats: dict[str, Any],
    write_query_log_fn: Callable[..., None],
) -> None:
    """统一成功生成后的日志写入，避免入口层重复拼装字段。"""
    write_query_log_fn(
        question=question,
        standalone_question=standalone_question,
        route=route,
        llm_model=llm_model,
        docs=docs,
        elapsed=elapsed,
        context_stats=context_stats,
    )


def stream_token_events(tokens: Any) -> Any:
    """将生成层文本 token 包装为 ask_stream 的事件格式。"""
    for token in tokens:
        yield {"type": "token", "data": token}


def stream_retrieval_events(route: str, docs: list[Any]) -> Any:
    """将检索策略和检索结果包装为 ask_stream 的前置事件。"""
    yield {"type": "route", "data": route}
    yield {"type": "docs", "data": docs}


def stream_rewrite_events(result: ReformulationResult) -> Any:
    """仅在多轮改写发生时产出 rewrite 事件。"""
    if result.rewritten:
        yield {"type": "rewrite", "data": result.standalone_question}
