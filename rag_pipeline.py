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
import re
from typing import Any

# 基准评估默认只使用本地模型文件，避免 transformers 导入后启动联网元数据查询。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document
from utils.config_loader import load_config
from utils.ollama_client import create_chat_ollama
from utils.prompt_loader import load_prompt
from hybrid_retriever import HybridRetriever
from langchain_chroma import Chroma
from paper_rag.config import RagSettings
from paper_rag.agentic.graph import run_agentic_rag
from query_expansion import expand_query
from reranker import apply_rerank
from context_builder import build_context_stats, prepare_docs_for_context
from generation_service import LLM_STREAM_DISCONNECTED_MESSAGE, generate_answer, stream_answer_tokens
from paper_rag.observability.service import write_query_log
from paper_rag.observability.trace import TraceTimer
from retrieval_router import (
    RetrievalRouter,
    deduplicate_docs,
    get_compare_anchor_docs,
    is_comparison_question,
    is_evidence_question,
    is_overview_question,
    load_anchor_docs_by_page,
    mentioned_source_files,
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


def _get_settings() -> RagSettings:
    """从当前 config 构建 settings，保留测试和运行期 patch config 的兼容语义。"""
    return RagSettings.from_mapping(config)


# ── LLM 单例缓存 ──────────────────────────────────────
# 按 (model, temperature) 缓存多个 ChatOllama 实例，方便在默认/对照模型之间切换。
_llm_cache = {}


def _get_embedding_device() -> str:
    """优先使用 GPU；如果不可用则回退到 CPU。"""
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _get_llm(
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    num_ctx: int = LLM_NUM_CTX,
    num_predict: int = LLM_NUM_PREDICT,
):
    """获取 LLM 实例（按模型名缓存，避免重复创建连接）"""
    global _llm_cache
    cache_key = (llm_model, temperature, num_ctx, num_predict)
    if cache_key not in _llm_cache:
        llm = create_chat_ollama(
            model=llm_model,
            temperature=temperature,
            num_ctx=num_ctx,
            num_predict=num_predict,
        )
        try:
            llm.invoke("ping")
            print(f"✅ LLM 模型 {llm_model} 连接成功")
            _llm_cache[cache_key] = llm
        except Exception as e:
            print(f"❌ LLM 连接失败：{e}")
            _llm_cache[cache_key] = None
    return _llm_cache[cache_key]


def build_hybrid_retriever():
    """构建检索器：从 Chroma 加载向量库 → 包装为 retriever"""
    current_settings = _get_settings()
    device = _get_embedding_device()
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=current_settings.embedding_model,
        model_kwargs={"device": device, "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        persist_directory=current_settings.persist_directory,
        embedding_function=embeddings,
        collection_name=current_settings.collection_name,
    )
    hybrid = HybridRetriever(
        vector_store=vector_store,
        top_k=current_settings.k,
        default_vector_weight=current_settings.default_vector_weight,
        default_bm25_weight=current_settings.default_bm25_weight,
        embedding_model=embeddings,
    )
    print(f"🔀 混合检索器已就绪（向量 + BM25, device={device}）")
    return hybrid


# ── 基础检索与生成（多轮对话复用）──────────────────────

def _format_docs(docs):
    """统一格式化检索到的文档片段"""
    blocks = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "?")
        header = f"[片段{i} | 来源={source} | 页码={page}]"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _write_query_log(
    question: str,
    standalone_question: str,
    route: str,
    llm_model: str,
    docs: list,
    elapsed: dict[str, float],
    context_stats: dict | None = None,
    agent_trace: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    write_query_log(
        settings=_get_settings(),
        question=question,
        standalone_question=standalone_question,
        route=route,
        llm_model=llm_model,
        docs=docs,
        elapsed=elapsed,
        embedding_device_fn=_get_embedding_device,
        context_stats=context_stats,
        agent_trace=agent_trace,
        error=error,
    )


def _deduplicate_docs(docs):
    return deduplicate_docs(docs)


def _join_log_context(*parts: str) -> str:
    return " ".join(part for part in parts if part)


def _mentioned_source_files(question: str) -> list[str]:
    return mentioned_source_files(question)


def _load_anchor_docs_by_page(hybrid: HybridRetriever, source_files: list[str], pages: list[int]) -> list[Document]:
    return load_anchor_docs_by_page(hybrid, source_files, pages)


def _get_compare_anchor_docs(hybrid: HybridRetriever, question: str) -> list[Document]:
    return get_compare_anchor_docs(hybrid, question)


def _create_llm(llm_model: str = LLM_MODEL, temperature: float = TEMPERATURE):
    """为 agentic planner/verifier 创建 LLM，沿用现有缓存与连接检查。"""
    return _get_llm(llm_model or LLM_MODEL, temperature)


def _classify_agentic_task(question: str) -> str:
    if is_comparison_question(question):
        return "compare"
    q_lower = question.lower()
    figure_signals = ["figure", "fig.", "图表", "图片", "示意图", "图中", "图里"]
    has_numbered_figure = re.search(r"(?:图|figure)\s*\d+", q_lower) is not None
    has_figure_signal = has_numbered_figure or any(signal in q_lower for signal in figure_signals)
    is_evidence = is_evidence_question(question)
    has_standalone_figure_evidence = is_evidence and "图" in q_lower and "图像" not in q_lower and "图注意力" not in q_lower
    if has_figure_signal or has_standalone_figure_evidence:
        return "figure"
    if is_evidence:
        return "evidence"
    return "method"


def _should_use_agentic(
    question: str,
    settings: RagSettings,
    force_agent: bool | None = None,
) -> bool:
    if force_agent is not None:
        return force_agent
    if not settings.enable_agentic_query:
        return False
    if not settings.agent_auto_for_complex:
        return True
    if is_comparison_question(question) or is_evidence_question(question):
        return True

    q_lower = question.lower()
    complex_signals = [
        "分别", "同时", "多个", "哪些", "为什么", "如何证明", "是否支持",
        "总结并", "先", "再", "compare and", "summarize and", "why", "how does",
        "multiple", "several", "evidence for",
    ]
    return len(question) >= 80 and any(signal in q_lower for signal in complex_signals)


def _retrieve(hybrid: HybridRetriever, query: str, log_context: str = "") -> list:
    """纯检索，不生成答案（已去重）"""
    try:
        retriever = hybrid.get_retriever(query, log_context=log_context)
    except TypeError:
        retriever = hybrid.get_retriever(query)
    docs = retriever.invoke(query)
    return _deduplicate_docs(docs)


def _retrieve_multi_query(
    hybrid: HybridRetriever,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    log_context: str = "",
) -> tuple[list, list[str]]:
    """对原始 query 和改写 query 分别召回，合并去重后返回。"""
    current_settings = _get_settings()
    original_context = _join_log_context(log_context, "[query original]")
    original_docs = _retrieve(hybrid, question, log_context=original_context)
    if not current_settings.enable_query_expansion:
        return original_docs, []

    n_variants = current_settings.query_expansion_variants
    expansion_model = current_settings.query_expansion_model or llm_model
    llm = _get_llm(expansion_model, temperature)
    try:
        variants = expand_query(question, llm, n_variants=n_variants)
    except Exception as exc:
        print(f"⚠️  Query expansion 失败：{type(exc).__name__}: {exc}，仅使用原始 query")
        return original_docs, []

    if not variants:
        return original_docs, []

    print(f"🔎 {_join_log_context(log_context, '[query expansion]')} Query variants:")
    for index, variant in enumerate(variants, 1):
        print(f"  [query variant {index}/{len(variants)}] {variant}")

    merged_docs = list(original_docs)
    for index, variant in enumerate(variants, 1):
        variant_context = _join_log_context(log_context, f"[query variant {index}/{len(variants)}]")
        merged_docs.extend(_retrieve(hybrid, variant, log_context=variant_context))

    merged_docs = _deduplicate_docs(merged_docs)
    max_multiplier = current_settings.query_expansion_max_multiplier
    max_docs = max(len(original_docs) * max_multiplier, len(original_docs))
    return merged_docs[:max_docs], variants


def _retrieve_with_hyde(
    hybrid: HybridRetriever,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
) -> list:
    """HyDE 纯检索，返回去重后的文档列表"""
    hyde_prompt_txt = load_prompt("hyde_prompt")
    hyde_prompt = hyde_prompt_txt.format(query=question)

    llm = _get_llm(llm_model, temperature)
    if llm is None:
        print("⚠️  LLM 不可用，降级为混合检索")
        return _retrieve(hybrid, question)

    try:
        hyde_response = llm.invoke(hyde_prompt)
        hyde_doc = hyde_response.content if hasattr(hyde_response, "content") else str(hyde_response)
        print(f"🧠 HyDE 生成假设性文档（{len(hyde_doc)} 字符）")
    except Exception as e:
        print(f"⚠️  HyDE 生成失败：{e}，降级为混合检索")
        return _retrieve(hybrid, question)

    retriever = hybrid.get_retriever(hyde_doc)
    docs = retriever.invoke(hyde_doc)
    docs = _deduplicate_docs(docs)
    print(f"📄 HyDE 检索到 {len(docs)} 个相关文档（去重后）")
    return docs


def _generate_answer(
    question: str,
    docs: list,
    history_text: str = "",
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    hybrid: HybridRetriever | None = None,
    prepared_context_docs: list | None = None,
    verified_evidence_summary: str = "",
) -> str:
    """用检索到的文档 + 可选多轮历史生成最终答案（非流式）"""
    context_docs = prepared_context_docs or prepare_docs_for_context(question, docs, hybrid=hybrid, settings=_get_settings())
    context = _format_docs(context_docs)
    prompt_txt = load_prompt("rag_summary_prompt")
    llm = _get_llm(llm_model, temperature)
    return generate_answer(
        llm,
        prompt_template=prompt_txt,
        context=context,
        question=question,
        history_text=history_text,
        verified_evidence_summary=verified_evidence_summary,
    )


# ── 路由 ──────────────────────────────────────────────

def _route_retrieve(
    hybrid: HybridRetriever,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
) -> tuple[list, str]:
    """统一路由检索入口（兼容旧调用方，内部委托 retrieval_router）。"""
    router = RetrievalRouter(
        settings=_get_settings(),
        llm_factory=_get_llm,
        hyde_retrieve_fn=_retrieve_with_hyde,
        multi_query_retrieve_fn=_retrieve_multi_query,
        apply_rerank_fn=apply_rerank,
        embedding_device_fn=_get_embedding_device,
    )
    return router.route(hybrid, question, llm_model, temperature)


def _run_agentic_retrieval(
    hybrid: HybridRetriever,
    question: str,
    standalone_question: str,
    current_settings: RagSettings,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
) -> tuple[list, str, dict[str, Any], str]:
    router = RetrievalRouter(
        settings=current_settings,
        llm_factory=_get_llm,
        hyde_retrieve_fn=_retrieve_with_hyde,
        multi_query_retrieve_fn=_retrieve_multi_query,
        apply_rerank_fn=apply_rerank,
        embedding_device_fn=_get_embedding_device,
    )
    source_hints = mentioned_source_files(standalone_question, hybrid, current_settings)
    agent_state = run_agentic_rag(
        question=question,
        standalone_question=standalone_question,
        task_type=_classify_agentic_task(standalone_question),
        source_hints=source_hints,
        hybrid=hybrid,
        router=router,
        planner_llm=_create_llm(current_settings.agent_planner_model, temperature),
        verifier_llm=_create_llm(
            current_settings.agent_verifier_model,
            current_settings.agent_verifier_temperature,
        ),
        llm_model=llm_model,
        temperature=temperature,
        max_repair_rounds=current_settings.agent_max_repair_rounds,
    )
    docs = list(agent_state.get("final_docs") or [])
    route = str(agent_state.get("route") or "agentic")
    agent_trace = agent_state.get("agent_trace") or {}
    verified_evidence_summary = str(agent_state.get("verified_summary") or "")
    return docs, route, agent_trace, verified_evidence_summary


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

def _ask_with_context_impl(
    hybrid: HybridRetriever,
    conversation,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    force_agent: bool | None = None,
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

    current_settings = _get_settings()
    use_agentic = _should_use_agentic(standalone_q, current_settings, force_agent=force_agent)

    retrieve_start = timer.start_stage()
    agent_trace: dict[str, Any] | None = None
    verified_evidence_summary = ""
    if use_agentic:
        docs, route, agent_trace, verified_evidence_summary = _run_agentic_retrieval(
            hybrid=hybrid,
            question=question,
            standalone_question=standalone_q,
            current_settings=current_settings,
            llm_model=llm_model,
            temperature=temperature,
        )
    else:
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
            agent_trace=agent_trace,
        )
        return "❌ 未找到相关内容", [], agent_trace or {}

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
        verified_evidence_summary=verified_evidence_summary,
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
        agent_trace=agent_trace,
    )
    return answer, docs, agent_trace or {}


def ask_with_context(
    hybrid: HybridRetriever,
    conversation,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    force_agent: bool | None = None,
):
    """
    带多轮对话上下文的问答：
    1. 用历史改写追问 → 独立可检索的问题
    2. 检索（对比/概述 → 混合 / 其他 → HyDE）
    3. 用原始问题 + 检索 + 历史生成连贯回答
    """
    answer, docs, _ = _ask_with_context_impl(
        hybrid=hybrid,
        conversation=conversation,
        question=question,
        llm_model=llm_model,
        temperature=temperature,
        force_agent=force_agent,
    )
    return answer, docs


def ask_with_context_trace(
    hybrid: HybridRetriever,
    conversation,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    force_agent: bool | None = None,
) -> tuple[str, list, dict[str, Any]]:
    """诊断/benchmark 入口：保留原问答行为，并额外返回 agent trace。"""
    return _ask_with_context_impl(
        hybrid=hybrid,
        conversation=conversation,
        question=question,
        llm_model=llm_model,
        temperature=temperature,
        force_agent=force_agent,
    )


# ── 流式生成（Streamlit Web 调用入口）─────────────────

def ask_stream(
    hybrid: HybridRetriever,
    conversation,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
    force_agent: bool | None = None,
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
    current_settings = _get_settings()
    use_agentic = _should_use_agentic(standalone_q, current_settings, force_agent=force_agent)
    retrieve_start = timer.start_stage()
    agent_trace: dict[str, Any] | None = None
    verified_evidence_summary = ""
    if use_agentic:
        yield {"type": "agent_status", "data": "正在拆分证据目标..."}
        docs, strategy, agent_trace, verified_evidence_summary = _run_agentic_retrieval(
            hybrid=hybrid,
            question=question,
            standalone_question=standalone_q,
            current_settings=current_settings,
            llm_model=llm_model,
            temperature=temperature,
        )
        yield {"type": "agent_status", "data": "正在校验证据并组装上下文..."}
        if current_settings.agent_debug_trace:
            yield {"type": "agent_trace", "data": agent_trace}
    else:
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
            agent_trace=agent_trace,
        )
        yield {"type": "token", "data": "❌ 未找到相关内容"}
        return

    # Step 3: 流式生成
    generate_start = timer.start_stage()
    history_text = conversation.format_history()
    context_docs = prepare_docs_for_context(question, docs, hybrid=hybrid, settings=_get_settings())
    context = _format_docs(context_docs)
    prompt_txt = load_prompt("rag_summary_prompt")
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
            agent_trace=agent_trace,
            error="LLM 模型未连接",
        )
        yield {"type": "token", "data": LLM_STREAM_DISCONNECTED_MESSAGE}
        return

    for text in stream_answer_tokens(
        llm,
        prompt_template=prompt_txt,
        context=context,
        question=question,
        history_text=history_text,
        verified_evidence_summary=verified_evidence_summary,
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
        agent_trace=agent_trace,
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
