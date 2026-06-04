"""
RAG 问答链模块
功能：连接已构建的向量数据库，创建检索问答链
支持多轮对话上下文 + 流式输出
"""
try:
    import torch
except ImportError:
    torch = None

import os

# 基准评估默认只使用本地模型文件，避免 transformers 导入后启动联网元数据查询。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from langchain_core.documents import Document
from utils.config_loader import load_config
from utils.ollama_client import create_chat_ollama
from utils.prompt_loader import load_prompt
from paper_rag.config import RagSettings
from paper_rag.generation.context import build_context_stats, prepare_docs_for_context
from paper_rag.generation.service import (
    LLM_STREAM_DISCONNECTED_MESSAGE,
    format_docs as format_generation_docs,
    generate_answer,
    generate_answer_from_docs,
    stream_answer_from_docs,
    stream_answer_tokens,
)
from paper_rag.observability.service import write_query_log
from paper_rag.observability.trace import TraceTimer
from paper_rag.pipeline.retrieval import (
    retrieve_documents,
    retrieve_multi_query as retrieve_multi_query_pipeline,
    retrieve_with_hyde as retrieve_with_hyde_pipeline,
    route_retrieve as route_retrieve_pipeline,
)
from paper_rag.retrieval.hybrid import HybridRetriever
from paper_rag.retrieval.query_expansion import (
    expand_query,
    filter_query_variants,
    query_variant_embed_fn_from_hybrid,
)
from paper_rag.retrieval.reranker import apply_rerank
from paper_rag.retrieval.router import (
    deduplicate_docs,
    get_compare_anchor_docs,
    is_comparison_question,
    is_evidence_question,
    is_overview_question,
    load_anchor_docs_by_page,
    mentioned_source_files,
)
from paper_rag.runtime.models import (
    build_hybrid_retriever as build_runtime_hybrid_retriever,
    get_cached_llm,
    select_embedding_device,
)

config = load_config()
settings = RagSettings.from_mapping(config)
CHROMA_DB_DIR = settings.persist_directory
EMBEDDING_MODEL = settings.embedding_model
LLM_MODEL = settings.llm_model
TEMPERATURE = settings.temperature
LLM_NUM_CTX = settings.llm_num_ctx
LLM_NUM_PREDICT = settings.llm_num_predict
TOP_K_RESULTS = settings.k

_query_expansion_trace: dict[str, list] = {"variants": [], "rejections": []}


def _get_settings() -> RagSettings:
    """从当前 config 构建 settings，保留测试和运行期 patch config 的兼容语义。"""
    return RagSettings.from_mapping(config)


# ── LLM 单例缓存 ──────────────────────────────────────
# 按 (model, temperature) 缓存多个 ChatOllama 实例，方便在默认/对照模型之间切换。
_llm_cache = {}


def _get_embedding_device() -> str:
    """优先使用 GPU；如果不可用则回退到 CPU。"""
    return select_embedding_device(torch)

def _get_llm(
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    num_ctx: int = LLM_NUM_CTX,
    num_predict: int = LLM_NUM_PREDICT,
):
    """获取 LLM 实例（按模型名缓存，避免重复创建连接）"""
    return get_cached_llm(
        _llm_cache,
        create_chat_ollama,
        model=llm_model,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        on_success=lambda model: print(f"✅ LLM 模型 {model} 连接成功"),
        on_error=lambda exc: print(f"❌ LLM 连接失败：{exc}"),
    )


def build_hybrid_retriever():
    """构建检索器：从 Chroma 加载向量库 → 包装为 retriever"""
    current_settings = _get_settings()
    device = _get_embedding_device()
    hybrid = build_runtime_hybrid_retriever(
        current_settings,
        device=device,
    )
    print(f"🔀 混合检索器已就绪（向量 + BM25, device={device}）")
    return hybrid


# ── 基础检索与生成（多轮对话复用）──────────────────────

def _format_docs(docs):
    """统一格式化检索到的文档片段"""
    return format_generation_docs(docs)


def _write_query_log(
    question: str,
    standalone_question: str,
    route: str,
    llm_model: str,
    docs: list,
    elapsed: dict[str, float],
    context_stats: dict | None = None,
    error: str | None = None,
) -> None:
    trace = _query_expansion_trace if route.startswith("mixed") else {"variants": [], "rejections": []}
    write_query_log(
        settings=_get_settings(),
        question=question,
        standalone_question=standalone_question,
        route=route,
        llm_model=llm_model,
        docs=docs,
        elapsed=elapsed,
        embedding_device_fn=_get_embedding_device,
        query_variants=trace["variants"],
        query_variant_rejections=trace["rejections"],
        context_stats=context_stats,
        error=error,
    )


def _deduplicate_docs(docs):
    return deduplicate_docs(docs)


def _mentioned_source_files(question: str) -> list[str]:
    return mentioned_source_files(question)


def _load_anchor_docs_by_page(hybrid: HybridRetriever, source_files: list[str], pages: list[int]) -> list[Document]:
    return load_anchor_docs_by_page(hybrid, source_files, pages)


def _get_compare_anchor_docs(hybrid: HybridRetriever, question: str) -> list[Document]:
    return get_compare_anchor_docs(hybrid, question)


def _retrieve(hybrid: HybridRetriever, query: str) -> list:
    """纯检索，不生成答案（已去重）"""
    return retrieve_documents(hybrid, query)


def _retrieve_multi_query(
    hybrid: HybridRetriever,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
) -> tuple[list, list[str]]:
    """对原始 query 和改写 query 分别召回，合并去重后返回。"""
    _query_expansion_trace["variants"] = []
    _query_expansion_trace["rejections"] = []
    result = retrieve_multi_query_pipeline(
        hybrid=hybrid,
        question=question,
        settings=_get_settings(),
        llm_model=llm_model,
        temperature=temperature,
        llm_factory=_get_llm,
        retrieve_fn=_retrieve,
    )
    _query_expansion_trace["variants"] = list(result.variants)
    _query_expansion_trace["rejections"] = list(result.rejections)
    return result.docs, result.variants


def _retrieve_with_hyde(
    hybrid: HybridRetriever,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
) -> list:
    """HyDE 纯检索，返回去重后的文档列表"""
    return retrieve_with_hyde_pipeline(
        hybrid,
        question,
        llm_model,
        temperature,
        load_prompt_fn=load_prompt,
        get_llm_fn=_get_llm,
        retrieve_fn=_retrieve,
    )


def _generate_answer(
    question: str,
    docs: list,
    history_text: str = "",
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    hybrid: HybridRetriever | None = None,
    prepared_context_docs: list | None = None,
) -> str:
    """用检索到的文档 + 可选多轮历史生成最终答案（非流式）"""
    return generate_answer_from_docs(
        question=question,
        docs=docs,
        history_text=history_text,
        llm_model=llm_model,
        temperature=temperature,
        hybrid=hybrid,
        settings=_get_settings(),
        prepared_context_docs=prepared_context_docs,
        prepare_docs_fn=prepare_docs_for_context,
        format_docs_fn=_format_docs,
        load_prompt_fn=load_prompt,
        get_llm_fn=_get_llm,
        generate_answer_fn=generate_answer,
    )


# ── 路由 ──────────────────────────────────────────────

def _route_retrieve(
    hybrid: HybridRetriever,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
) -> tuple[list, str]:
    """统一路由检索入口（兼容旧调用方，内部委托 retrieval_router）。"""
    return route_retrieve_pipeline(
        hybrid=hybrid,
        question=question,
        settings=_get_settings(),
        llm_model=llm_model,
        temperature=temperature,
        llm_factory=_get_llm,
        hyde_retrieve_fn=_retrieve_with_hyde,
        multi_query_retrieve_fn=_retrieve_multi_query,
        apply_rerank_fn=apply_rerank,
        embedding_device_fn=_get_embedding_device,
    )


def route_question(
    hybrid: HybridRetriever,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
):
    """
    智能路由（单轮）：
    - 概述型 / 对比型 → 混合检索
    - 其他 → HyDE 增强检索
    """
    docs, _ = _route_retrieve(hybrid, question, llm_model, temperature)
    if not docs:
        return "❌ 未找到相关内容", []
    answer = _generate_answer(question, docs, llm_model=llm_model, temperature=temperature, hybrid=hybrid)
    return answer, docs


def ask_with_hyde(
    hybrid: HybridRetriever,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
):
    """带 HyDE 增强的问答，返回答案和检索到的源文档"""
    docs = _retrieve_with_hyde(hybrid, question, llm_model, temperature)
    if not docs:
        return "❌ HyDE 检索未找到相关内容", []
    answer = _generate_answer(question, docs, llm_model=llm_model, temperature=temperature, hybrid=hybrid)
    return answer, docs


# ── 多轮对话 ──────────────────────────────────────────

def ask_with_context(
    hybrid: HybridRetriever,
    conversation,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
):
    """
    带多轮对话上下文的问答：
    1. 用历史改写追问 → 独立可检索的问题
    2. 检索（对比/概述 → 混合 / 其他 → HyDE）
    3. 用原始问题 + 检索 + 历史生成连贯回答
    """
    timer = TraceTimer()
    rewrite_start = timer.start_stage()
    standalone_q = conversation.reformulate(question)
    rewrite_elapsed = timer.elapsed_since(rewrite_start)
    if standalone_q != question:
        print(f'🔄 改写追问: "{standalone_q}"')

    retrieve_start = timer.start_stage()
    docs, route = _route_retrieve(hybrid, standalone_q, llm_model, temperature)
    retrieve_elapsed = timer.elapsed_since(retrieve_start)

    if not docs:
        _write_query_log(
            question=question,
            standalone_question=standalone_q,
            route=route,
            llm_model=llm_model,
            docs=[],
            elapsed=timer.elapsed_map(rewrite_elapsed, retrieve_elapsed, 0.0),
        )
        return "❌ 未找到相关内容", []

    history_text = conversation.format_history()
    generate_start = timer.start_stage()
    context_docs = prepare_docs_for_context(question, docs, hybrid=hybrid, settings=_get_settings())
    answer = _generate_answer(
        question,
        docs,
        history_text,
        llm_model=llm_model,
        temperature=temperature,
        hybrid=hybrid,
        prepared_context_docs=context_docs,
    )
    generate_elapsed = timer.elapsed_since(generate_start)
    _write_query_log(
        question=question,
        standalone_question=standalone_q,
        route=route,
        llm_model=llm_model,
        docs=docs,
        elapsed=timer.elapsed_map(rewrite_elapsed, retrieve_elapsed, generate_elapsed),
        context_stats=build_context_stats(docs, context_docs),
    )
    return answer, docs


# ── 流式生成（Streamlit Web 调用入口）─────────────────

def ask_stream(
    hybrid: HybridRetriever,
    conversation,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
):
    """
    一次调用完成全链路：改写追问 → 路由检索 → 流式生成 → 返回来源

    Yields:
        {"type": "docs", "data": [Document, ...]}   — 检索到的文档
        {"type": "token", "data": "字"}             — 逐字输出
        {"type": "rewrite", "data": "改写后问题"}    — （可选）多轮改写结果
        {"type": "route", "data": "mixed"/"hyde"}   — （可选）使用的检索策略

    用法：
        for event in ask_stream(hybrid, conv, question):
            if event["type"] == "docs":
                docs = event["data"]
            elif event["type"] == "token":
                yield event["data"]  # 逐字输出到 Web
    """
    timer = TraceTimer()
    # Step 1: 改写追问（多轮时）
    rewrite_start = timer.start_stage()
    if conversation.history:
        standalone_q = conversation.reformulate(question)
        if standalone_q != question:
            yield {"type": "rewrite", "data": standalone_q}
    else:
        standalone_q = question
    rewrite_elapsed = timer.elapsed_since(rewrite_start)

    # Step 2: 路由检索（统一入口）
    retrieve_start = timer.start_stage()
    docs, strategy = _route_retrieve(hybrid, standalone_q, llm_model, temperature)
    retrieve_elapsed = timer.elapsed_since(retrieve_start)
    yield {"type": "route", "data": strategy}
    yield {"type": "docs", "data": docs}

    if not docs:
        _write_query_log(
            question=question,
            standalone_question=standalone_q,
            route=strategy,
            llm_model=llm_model,
            docs=[],
            elapsed=timer.elapsed_map(rewrite_elapsed, retrieve_elapsed, 0.0),
        )
        yield {"type": "token", "data": "❌ 未找到相关内容"}
        return

    # Step 3: 流式生成
    generate_start = timer.start_stage()
    history_text = conversation.format_history()
    context_docs = prepare_docs_for_context(question, docs, hybrid=hybrid, settings=_get_settings())
    llm = _get_llm(llm_model, temperature)
    if llm is None:
        _write_query_log(
            question=question,
            standalone_question=standalone_q,
            route=strategy,
            llm_model=llm_model,
            docs=docs,
            elapsed=timer.elapsed_map(
                rewrite_elapsed,
                retrieve_elapsed,
                timer.elapsed_since(generate_start),
            ),
            context_stats=build_context_stats(docs, context_docs),
            error="LLM 模型未连接",
        )
        yield {"type": "token", "data": LLM_STREAM_DISCONNECTED_MESSAGE}
        return

    for text in stream_answer_from_docs(
        question=question,
        docs=docs,
        history_text=history_text,
        llm_model=llm_model,
        temperature=temperature,
        hybrid=hybrid,
        settings=_get_settings(),
        prepared_context_docs=context_docs,
        prepare_docs_fn=prepare_docs_for_context,
        format_docs_fn=_format_docs,
        load_prompt_fn=load_prompt,
        get_llm_fn=lambda _model, _temperature: llm,
        stream_answer_tokens_fn=stream_answer_tokens,
    ):
        yield {"type": "token", "data": text}

    _write_query_log(
        question=question,
        standalone_question=standalone_q,
        route=strategy,
        llm_model=llm_model,
        docs=docs,
        elapsed=timer.elapsed_map(
            rewrite_elapsed,
            retrieve_elapsed,
            timer.elapsed_since(generate_start),
        ),
        context_stats=build_context_stats(docs, context_docs),
    )


# ── 测试 ──────────────────────────────────────────────

def main():
    """测试问答链是否正常工作"""
    print("=" * 50)
    print("🔧 RAG 问答链测试")
    print("=" * 50)
    print("\n🔗 连接向量数据库...")
    hybrid = build_hybrid_retriever()

    print("\n🛠️  构建问答链...")
    test_question = "What are the differences between how BERT and ViT use the Transformer?"
    print(f"\n📝 测试问题：{test_question}")
    answer, sources = route_question(hybrid, test_question)
    print(f"\n💬 回答：{answer}")

    print(f"\n📎 参考来源（{len(sources)} 个）：")
    for i, doc in enumerate(sources):
        source_file = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "?")
        excerpt = doc.page_content[:100].replace("\n", " ")
        print(f"  [{i + 1}] {source_file} (第{page}页): {excerpt}...")


if __name__ == '__main__':
    main()
