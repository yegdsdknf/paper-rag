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
    verified_evidence_summary: str = "",
) -> str:
    """统一构造 RAG prompt，避免流式和非流式生成各拼一套。"""
    verified_summary = verified_evidence_summary.strip()
    agentic_prefix = f"{verified_summary}\n\n" if verified_summary else ""
    context_with_agentic = agentic_prefix + context
    return history_text + ANSWER_ORDER_INSTRUCTION + "\n" + prompt_template.format(
        context=context_with_agentic,
        question=question,
    )


def _content_from_response(response: Any) -> str:
    return response.content if hasattr(response, "content") else str(response)


def generate_answer(
    llm: Any,
    prompt_template: str,
    context: str,
    question: str,
    history_text: str = "",
    verified_evidence_summary: str = "",
) -> str:
    if llm is None:
        return LLM_DISCONNECTED_MESSAGE

    full_prompt = build_rag_prompt(
        prompt_template=prompt_template,
        context=context,
        question=question,
        history_text=history_text,
        verified_evidence_summary=verified_evidence_summary,
    )
    return _content_from_response(llm.invoke(full_prompt)).strip()


def stream_answer_tokens(
    llm: Any,
    prompt_template: str,
    context: str,
    question: str,
    history_text: str = "",
    verified_evidence_summary: str = "",
) -> Iterable[str]:
    if llm is None:
        yield LLM_STREAM_DISCONNECTED_MESSAGE
        return

    full_prompt = build_rag_prompt(
        prompt_template=prompt_template,
        context=context,
        question=question,
        history_text=history_text,
        verified_evidence_summary=verified_evidence_summary,
    )
    for chunk in llm.stream(full_prompt):
        text = _content_from_response(chunk)
        if text:
            yield text

