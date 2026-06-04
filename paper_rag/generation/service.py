from __future__ import annotations

from typing import Any, Iterable


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

