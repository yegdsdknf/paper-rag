"""
论文知识库问答系统 — Streamlit Web 界面
用法：streamlit run app.py
"""
import os, sys, streamlit as st

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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


# ═══════════════════════════════════════════════════════
# 侧边栏
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.title("📚 论文知识库")
    _, cfg, _, _ = _init()
    st.caption(f"📌 {cfg['llm_model']}  |  🧩 {cfg['embedding_model']}  |  k={cfg['k']}")

    st.divider()
    st.subheader("📤 上传论文")
    up = st.file_uploader("拖拽 PDF", type="pdf", accept_multiple_files=True, label_visibility="collapsed")

    if up and st.button("🚀 一键入库", type="primary", use_container_width=True):
        _, _, _, _, bk = _imports()
        papers_dir = os.path.join(PROJECT_ROOT, "papers")
        os.makedirs(papers_dir, exist_ok=True)
        for f in up:
            with open(os.path.join(papers_dir, f.name), "wb") as fh:
                fh.write(f.getbuffer())
        bk.main()
        # 递增版本 → 下次 _init() 自动重建检索器（无需清空其他缓存）
        st.session_state.db_version = st.session_state.get("db_version", 0) + 1
        st.success("入库完成 — 检索器已自动刷新")
        st.rerun()

    st.divider()
    if st.button("🧹 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation = None
        st.rerun()

    st.divider()
    st.caption("💡 对比/概述 → 混合检索")
    st.caption("🧠 其他 → HyDE 增强")
    st.caption("⚡ 流式输出")

# ═══════════════════════════════════════════════════════
# 会话状态
# ═══════════════════════════════════════════════════════

if "db_version" not in st.session_state:
    st.session_state.db_version = 0
if "messages" not in st.session_state:
    st.session_state.messages = []
if not st.session_state.get("conversation"):
    _, cfg, _, CM = _init()
    st.session_state.conversation = CM(cfg["llm_model"], cfg["temperature"])

# ═══════════════════════════════════════════════════════
# 主界面
# ═══════════════════════════════════════════════════════

st.title("📚 论文知识库问答系统")
st.caption("混合检索 + HyDE 增强  |  中英文多轮对话  |  流式输出")

try:
    hybrid, cfg, ask_stream, _ = _init()
except Exception as e:
    st.error(f"初始化失败：{e}")
    st.stop()

# ── 历史消息 ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 参考来源"):
                for d in msg["sources"]:
                    src = d.metadata.get("source", "?")
                    pg = d.metadata.get("page", "?")
                    fn = src.replace("\\", "/").split("/")[-1]
                    st.caption(f"[{fn}] · p{pg}")
                    st.text(d.page_content[:400])

# ── 输入 + 流式回答 ──
if q := st.chat_input("💬 输入问题..."):
    conv = st.session_state.conversation

    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    with st.chat_message("assistant"):
        try:
            docs = []
            answer = ""

            placeholder = st.empty()
            for event in ask_stream(hybrid, conv, q):
                if event["type"] == "rewrite":
                    st.caption(f"🔄 改写追问：_{event['data']}_")
                elif event["type"] == "route":
                    label = "混合检索" if event["data"] == "mixed" else "HyDE 增强"
                    st.caption(f"🔀 检索策略：**{label}**")
                elif event["type"] == "docs":
                    docs = event["data"]
                    if docs:
                        st.caption(f"📄 检索到 {len(docs)} 个片段")
                elif event["type"] == "token":
                    answer += event["data"]
                    placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)

            if docs:
                with st.expander("📎 参考来源"):
                    for d in docs:
                        src = d.metadata.get("source", "?")
                        pg = d.metadata.get("page", "?")
                        fn = src.replace("\\", "/").split("/")[-1]
                        st.caption(f"[{fn}] · p{pg}")
                        st.text(d.page_content[:400])

            conv.add_turn(q, answer)
            st.session_state.messages.append({
                "role": "assistant", "content": answer, "sources": docs,
            })
        except Exception as e:
            st.error(str(e))
