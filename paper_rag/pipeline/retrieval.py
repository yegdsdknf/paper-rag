from __future__ import annotations

from typing import Any, Callable

from langchain_core.documents import Document

from paper_rag.retrieval.router import deduplicate_docs


def retrieve_documents(hybrid: Any, query: str) -> list[Document]:
    retriever = hybrid.get_retriever(query)
    docs = retriever.invoke(query)
    return deduplicate_docs(docs)


def retrieve_with_hyde(
    hybrid: Any,
    question: str,
    llm_model: str,
    temperature: float,
    *,
    load_prompt_fn: Callable[[str], str] | None = None,
    get_llm_fn: Callable[[str, float], Any | None] | None = None,
    retrieve_fn: Callable[[Any, str], list[Document]] = retrieve_documents,
) -> list[Document]:
    if load_prompt_fn is None:
        from utils.prompt_loader import load_prompt

        load_prompt_fn = load_prompt
    if get_llm_fn is None:
        raise ValueError("get_llm_fn is required")

    hyde_prompt = load_prompt_fn("hyde_prompt").format(query=question)
    llm = get_llm_fn(llm_model, temperature)
    if llm is None:
        print("⚠️  LLM 不可用，降级为混合检索")
        return retrieve_fn(hybrid, question)

    try:
        hyde_response = llm.invoke(hyde_prompt)
        hyde_doc = hyde_response.content if hasattr(hyde_response, "content") else str(hyde_response)
        print(f"🧠 HyDE 生成假设性文档（{len(hyde_doc)} 字符）")
    except Exception as exc:
        print(f"⚠️  HyDE 生成失败：{exc}，降级为混合检索")
        return retrieve_fn(hybrid, question)

    docs = retrieve_fn(hybrid, hyde_doc)
    print(f"📄 HyDE 检索到 {len(docs)} 个相关文档（去重后）")
    return docs
