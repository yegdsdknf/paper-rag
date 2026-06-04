from __future__ import annotations

from typing import Any, Callable, Iterable


ANSWER_ORDER_INSTRUCTION = "请严格按“结论 -> 证据 -> 限制”的顺序回答。"
LLM_DISCONNECTED_MESSAGE = "❌ LLM 模型未连接，请检查 Ollama 服务"
LLM_STREAM_DISCONNECTED_MESSAGE = "❌ LLM 模型未连接"


def build_rag_prompt(
    prompt_template: str,
    context: str,
    question: str,
    history_text: str = "",
) -> str:
    """统一构造 RAG prompt，避免流式和非流式生成各拼一套。"""
    return history_text + ANSWER_ORDER_INSTRUCTION + "\n" + prompt_template.format(context=context, question=question)


def format_docs(docs: Iterable[Any]) -> str:
    """统一格式化检索片段，供非流式和流式生成共享。"""
    blocks = []
    for index, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "?")
        header = f"[片段{index} | 来源={source} | 页码={page}]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _content_from_response(response: Any) -> str:
    return response.content if hasattr(response, "content") else str(response)


def generate_answer(
    llm: Any,
    prompt_template: str,
    context: str,
    question: str,
    history_text: str = "",
) -> str:
    if llm is None:
        return LLM_DISCONNECTED_MESSAGE

    full_prompt = build_rag_prompt(
        prompt_template=prompt_template,
        context=context,
        question=question,
        history_text=history_text,
    )
    return _content_from_response(llm.invoke(full_prompt)).strip()


def generate_answer_from_docs(
    question: str,
    docs: list[Any],
    *,
    history_text: str = "",
    llm_model: str,
    temperature: float,
    hybrid: Any | None = None,
    settings: Any | None = None,
    prepared_context_docs: list[Any] | None = None,
    prepare_docs_fn: Callable[..., list[Any]] | None = None,
    format_docs_fn: Callable[[Iterable[Any]], str] = format_docs,
    load_prompt_fn: Callable[[str], str] | None = None,
    get_llm_fn: Callable[[str, float], Any | None] | None = None,
    generate_answer_fn: Callable[..., str] = generate_answer,
) -> str:
    """把文档上下文准备、prompt 加载和 LLM 调用收敛到生成服务层。"""
    if prepare_docs_fn is None:
        from paper_rag.generation.context import prepare_docs_for_context

        prepare_docs_fn = prepare_docs_for_context
    if load_prompt_fn is None:
        from utils.prompt_loader import load_prompt

        load_prompt_fn = load_prompt
    if get_llm_fn is None:
        raise ValueError("get_llm_fn is required")

    context_docs = prepared_context_docs or prepare_docs_fn(
        question,
        docs,
        hybrid=hybrid,
        settings=settings,
    )
    context = format_docs_fn(context_docs)
    prompt_txt = load_prompt_fn("rag_summary_prompt")
    llm = get_llm_fn(llm_model, temperature)
    return generate_answer_fn(
        llm,
        prompt_template=prompt_txt,
        context=context,
        question=question,
        history_text=history_text,
    )


def stream_answer_tokens(
    llm: Any,
    prompt_template: str,
    context: str,
    question: str,
    history_text: str = "",
) -> Iterable[str]:
    if llm is None:
        yield LLM_STREAM_DISCONNECTED_MESSAGE
        return

    full_prompt = build_rag_prompt(
        prompt_template=prompt_template,
        context=context,
        question=question,
        history_text=history_text,
    )
    for chunk in llm.stream(full_prompt):
        text = _content_from_response(chunk)
        if text:
            yield text

