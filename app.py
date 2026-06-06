"""
论文知识库问答系统 — Streamlit Web 界面
用法：streamlit run app.py
"""
import os, sys, streamlit as st

from paper_rag.ui import (
    TokenStreamBuffer,
    build_feedback_payload,
    build_source_view_models,
    clear_conversation_state,
    format_runtime_error,
    render_streamlit_error,
    render_streamlit_startup_failure,
    save_feedback_from_payload,
    save_uploaded_pdfs,
)
from utils.console import configure_runtime_env

configure_runtime_env()

st.set_page_config(page_title="论文知识库问答", page_icon="📚", layout="wide")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ═══════════════════════════════════════════════════════
# 缓存层
# ═══════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _imports():
    from rag_pipeline import build_hybrid_retriever, ask_stream
    from conversation import ConversationManager
    from utils.config_loader import load_config
    import build_knowledge as bk
    return build_hybrid_retriever, ask_stream, ConversationManager, load_config, bk

@st.cache_resource(show_spinner="正在加载向量数据库和模型...")
def _init_from_cache(_db_version: int = 0):
    """缓存键绑定 _db_version，入库后递增版本即可触发重建"""
    build, ask, CM, load_cfg, bk = _imports()
    cfg = load_cfg()
    return build(), cfg, ask, CM


def _init():
    """统一入口：从 session_state 取版本号，传给缓存函数"""
    ver = st.session_state.get("db_version", 0)
    return _init_from_cache(ver)


def _render_sources(docs, question: str):
    for source in build_source_view_models(docs, question):
        label = source.title
        if source.score_label:
            label = f"{label} · {source.score_label}"
        st.caption(label)
        st.markdown(source.highlight_html, unsafe_allow_html=True)
        with st.expander("查看原始片段"):
            st.text(source.raw_preview)


# ═══════════════════════════════════════════════════════
# 会话状态与初始化
# ═══════════════════════════════════════════════════════

if "db_version" not in st.session_state:
    st.session_state.db_version = 0
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_feedback_payload" not in st.session_state:
    st.session_state.last_feedback_payload = None

try:
    hybrid, cfg, ask_stream, CM = _init()
except Exception as e:
    st.title("📚 论文知识库问答系统")
    render_streamlit_startup_failure(st, e, project_root=PROJECT_ROOT)
    st.stop()


# ═══════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.title("📚 论文知识库")
    model_options = {
        "默认演示": cfg["llm_model"],
        "reasoning 对照": cfg.get("llm_model_reasoning", cfg["llm_model"]),
    }
    selected_mode = st.selectbox("LLM 模式", list(model_options.keys()), index=0)
    selected_model = model_options[selected_mode]
    st.session_state.selected_llm_model = selected_model
    st.caption(f"📌 {selected_mode} · {selected_model}  |  🧩 {cfg['embedding_model']}  |  k={cfg['k']}")
    try:
        from paper_rag.config import RagSettings
        from paper_rag.indexing import load_index_manifest, resolve_index_version

        index_settings = RagSettings.from_mapping(cfg)
        manifest = load_index_manifest(index_settings)
        version = resolve_index_version(index_settings)
        detail = f" · {manifest.get('chunk_count', 0)} chunks" if manifest else " · 未找到 manifest"
        st.caption(f"🗂️ index={version}{detail}")
    except Exception:
        st.caption("🗂️ index=unknown")

    st.divider()
    st.subheader("📤 上传论文")
    up = st.file_uploader("拖拽 PDF", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

    if up and st.button("🚀 一键入库", type="primary", use_container_width=True):
        _, _, _, _, bk = _imports()
        papers_dir = os.path.join(PROJECT_ROOT, "papers")
        save_uploaded_pdfs(up, papers_dir)
        bk.main()
        # 递增版本 → 下次 _init() 自动重建检索器（无需清空其他缓存）
        st.session_state.db_version = st.session_state.get("db_version", 0) + 1
        st.success("入库完成 — 检索器已自动刷新")
        st.rerun()

    st.divider()
    if st.button("🧹 清空对话", use_container_width=True):
        clear_conversation_state(st.session_state)
        st.rerun()

    st.divider()
    st.caption("💡 对比/概述 → 混合检索")
    st.caption("🧠 其他 → HyDE 增强")
    st.caption("⚡ 流式输出")

if not st.session_state.get("conversation"):
    st.session_state.conversation = CM(
        cfg["llm_model"],
        cfg["temperature"],
        cfg.get("llm_num_ctx"),
        cfg.get("llm_num_predict"),
    )

# ═══════════════════════════════════════════════════════
# 主界面
# ═══════════════════════════════════════════════════════

st.title("📚 论文知识库问答系统")
st.caption("混合检索 + HyDE 增强  |  中英文多轮对话  |  流式输出")

# ── 历史消息 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 参考来源"):
                _render_sources(msg["sources"], msg.get("question", msg.get("content", "")))

# ── 输入 + 流式回答 ──
if q := st.chat_input("💬 输入问题..."):
    conv = st.session_state.conversation
    llm_model = st.session_state.get("selected_llm_model", cfg["llm_model"])
    conv.update_model(llm_model, cfg["temperature"])

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        try:
            docs = []
            answer = ""
            route = ""

            placeholder = st.empty()
            stream_buffer = TokenStreamBuffer(max_chunks=8, max_interval_sec=0.08)
            for event in ask_stream(hybrid, conv, q, llm_model=llm_model, temperature=cfg["temperature"]):
                if event["type"] == "rewrite":
                    st.caption(f"🔄 改写追问：_{event['data']}_")
                elif event["type"] == "route":
                    route = event["data"]
                    label = "混合检索" if event["data"].startswith("mixed") else "HyDE 增强"
                    st.caption(f"🔀 检索策略：**{label}**")
                elif event["type"] == "docs":
                    docs = event["data"]
                    if docs:
                        st.caption(f"📄 检索到 {len(docs)} 个片段")
                elif event["type"] == "token":
                    chunk = stream_buffer.append(event["data"])
                    if chunk:
                        answer += chunk
                        placeholder.markdown(answer + "▌")
            tail = stream_buffer.flush()
            if tail:
                answer += tail
            placeholder.markdown(answer)

            if docs:
                with st.expander("📎 参考来源"):
                    _render_sources(docs, q)

            conv.add_turn(q, answer)
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": docs, "question": q,
            })
            st.session_state.last_feedback_payload = build_feedback_payload(q, answer, docs, route, llm_model)
        except Exception as e:
            if "stream_buffer" in locals():
                tail = stream_buffer.flush()
                if tail:
                    answer += tail
                    placeholder.markdown(answer)
            render_streamlit_error(st, format_runtime_error(e, cfg))

payload = st.session_state.get("last_feedback_payload")
if payload:
    with st.expander("📝 记录失败样本 / 反馈"):
        note = st.text_area(
            "备注",
            placeholder="例如：答非所问、页码不对、证据不足、回答自相矛盾……",
            key="feedback_note",
        )
        if st.button("保存到反馈集", use_container_width=True):
            try:
                path = save_feedback_from_payload(payload, note)
                st.success(f"已保存：{path}")
            except ValueError as e:
                st.warning(str(e))
