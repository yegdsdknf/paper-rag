"""兼容薄壳：新代码请从 paper_rag.generation.service 导入。"""

from paper_rag.generation.service import (
    ANSWER_ORDER_INSTRUCTION,
    LLM_DISCONNECTED_MESSAGE,
    LLM_STREAM_DISCONNECTED_MESSAGE,
    build_rag_prompt,
    format_docs,
    generate_answer,
    generate_answer_from_docs,
    stream_answer_from_docs,
    stream_answer_tokens,
)

__all__ = [
    "ANSWER_ORDER_INSTRUCTION",
    "LLM_DISCONNECTED_MESSAGE",
    "LLM_STREAM_DISCONNECTED_MESSAGE",
    "build_rag_prompt",
    "format_docs",
    "generate_answer",
    "generate_answer_from_docs",
    "stream_answer_from_docs",
    "stream_answer_tokens",
]
