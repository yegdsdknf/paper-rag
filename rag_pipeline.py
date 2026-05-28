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

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from utils.config_loader import load_config
from utils.ollama_client import create_chat_ollama
from utils.prompt_loader import load_prompt
from hybrid_retriever import HybridRetriever
from langchain_chroma import Chroma
from reranker import apply_rerank

config = load_config()
CHROMA_DB_DIR = config['persist_directory']
EMBEDDING_MODEL = config['embedding_model']
LLM_MODEL = config['llm_model']
TEMPERATURE = config['temperature']
LLM_NUM_CTX = config.get("llm_num_ctx", 4096)
LLM_NUM_PREDICT = config.get("llm_num_predict", 1024)
TOP_K_RESULTS = config['k']

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
    device = _get_embedding_device()
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device, "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_store = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )
    hybrid = HybridRetriever(
        vector_store=vector_store,
        top_k=config["k"],
        default_vector_weight=config["default_vector_weight"],
        default_bm25_weight=config["default_bm25_weight"],
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


def _deduplicate_docs(docs):
    """同一来源同一页只保留一个 chunk，减少重复 token 消耗"""
    seen = set()
    unique = []
    for doc in docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page", -1))
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def _retrieve(hybrid: HybridRetriever, query: str) -> list:
    """纯检索，不生成答案（已去重）"""
    retriever = hybrid.get_retriever(query)
    docs = retriever.invoke(query)
    return _deduplicate_docs(docs)


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
) -> str:
    """用检索到的文档 + 可选多轮历史生成最终答案（非流式）"""
    context = _format_docs(docs)
    prompt_txt = load_prompt("rag_summary_prompt")

    llm = _get_llm(llm_model, temperature)
    if llm is None:
        return "❌ LLM 模型未连接，请检查 Ollama 服务"

    instruction = "请严格按“结论 -> 证据 -> 限制”的顺序回答。"
    full_prompt = history_text + instruction + "\n" + prompt_txt.format(context=context, question=question)
    response = llm.invoke(full_prompt)
    text = response.content if hasattr(response, "content") else str(response)
    return text.strip()


# ── 路由 ──────────────────────────────────────────────

def is_comparison_question(question: str) -> bool:
    """检测是否对比型问题"""
    comparison_signals = [
        "vs", "versus", "difference", "differences",
        "compare", "comparison", "between",
        "不同", "比较", "相比", "之间", "对比", "区别", "差别",
    ]
    q_lower = question.lower()
    return any(sig in q_lower for sig in comparison_signals)


def is_overview_question(question: str) -> bool:
    """检测是否基础介绍类问题 → 不适合 HyDE（HyDE 生成偏技术细节）"""
    overview_signals = [
        "是什么", "什么是", "介绍", "简介", "定义",
        "what is", "definition", "overview", "introduction",
    ]
    q_lower = question.lower()
    return any(sig in q_lower for sig in overview_signals)


def _route_retrieve(
    hybrid: HybridRetriever,
    question: str,
    llm_model: str = LLM_MODEL,
    temperature: float = TEMPERATURE,
) -> tuple[list, str]:
    """
    统一路由检索入口（所有调用方使用同一逻辑）

    Returns:
        (docs, strategy) — 去重后的文档列表 + 使用的策略名 ("mixed" / "hyde")
    """
    if is_comparison_question(question) or is_overview_question(question):
        print("🔀 使用标准混合检索")
        docs = _retrieve(hybrid, question)
        docs = apply_rerank(
            question,
            docs,
            enabled=config.get("enable_rerank", False),
            model_name=config.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
            top_k=config.get("rerank_top_k", TOP_K_RESULTS),
            device=_get_embedding_device(),
        )
        return docs, "mixed"
    else:
        print("🧠 使用 HyDE 增强检索")
        docs = _retrieve_with_hyde(hybrid, question, llm_model, temperature)
        return docs, "hyde"


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
    answer = _generate_answer(question, docs, llm_model=llm_model, temperature=temperature)
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
    answer = _generate_answer(question, docs, llm_model=llm_model, temperature=temperature)
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
    standalone_q = conversation.reformulate(question)
    if standalone_q != question:
        print(f'🔄 改写追问: "{standalone_q}"')

    docs, _ = _route_retrieve(hybrid, standalone_q, llm_model, temperature)

    if not docs:
        return "❌ 未找到相关内容", []

    history_text = conversation.format_history()
    answer = _generate_answer(
        question,
        docs,
        history_text,
        llm_model=llm_model,
        temperature=temperature,
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
    # Step 1: 改写追问（多轮时）
    if conversation.history:
        standalone_q = conversation.reformulate(question)
        if standalone_q != question:
            yield {"type": "rewrite", "data": standalone_q}
    else:
        standalone_q = question

    # Step 2: 路由检索（统一入口）
    docs, strategy = _route_retrieve(hybrid, standalone_q, llm_model, temperature)
    yield {"type": "route", "data": strategy}
    yield {"type": "docs", "data": docs}

    if not docs:
        yield {"type": "token", "data": "❌ 未找到相关内容"}
        return

    # Step 3: 流式生成
    history_text = conversation.format_history()
    context = _format_docs(docs)
    prompt_txt = load_prompt("rag_summary_prompt")
    instruction = "请严格按“结论 -> 证据 -> 限制”的顺序回答。"
    full_prompt = history_text + instruction + "\n" + prompt_txt.format(context=context, question=question)

    llm = _get_llm(llm_model, temperature)
    if llm is None:
        yield {"type": "token", "data": "❌ LLM 模型未连接"}
        return

    for chunk in llm.stream(full_prompt):
        text = chunk.content if hasattr(chunk, "content") else str(chunk)
        if text:
            yield {"type": "token", "data": text}


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
