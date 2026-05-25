# 模块一 入库
# 负责：读取 PDF → 文本切分 → 向量化 → 分批存入 Chroma
import os

try:
    import torch
except ImportError:
    torch = None

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.dedup_manager import DedupManager, add_with_dedup
from utils.config_loader import load_config

config = load_config()
PDF_DIR = "./papers"
CHROMA_DB_DIR = config["persist_directory"]
CHUNK_SIZE = config["chunk_size"]
CHUNK_OVERLAP = config["chunk_overlap"]
EMBEDDING_MODEL = config["embedding_model"]
BATCH_SIZE = 32

# 从配置文件读取跳过页面（config.yaml 中未配置则为空字典）
SKIP_PAGES = config.get("skip_pages", {})

def _get_embedding_device() -> str:
    """优先使用 GPU；如果不可用则回退到 CPU。"""
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pdfs(file_paths: list[str]) -> list[Document]:
    documents = []
    for fp in file_paths:
        loader = PyPDFLoader(fp)
        docs = loader.load()
        fname_lower = os.path.basename(fp).lower()
        for key, pages in SKIP_PAGES.items():
            if key in fname_lower:
                before = len(docs)
                page_set = set(pages)
                docs = [d for d in docs if d.metadata.get("page", 0) not in page_set]
                print(f"  🧹 {os.path.basename(fp)} — 跳过 {before - len(docs)} 页附录")
        documents.extend(docs)
        print(f"  ✅ {os.path.basename(fp)} — {len(docs)} 页")
    print(f"📄 成功加载 {len(documents)} 页文档")
    return documents


def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=config["separators"],
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"🔪 切分为 {len(chunks)} 个文本块（chunk_size={chunk_size}）")
    return chunks


def build_vector_store(chunks, embedding_model: str, persist_dir: str, batch_size: int = BATCH_SIZE):
    device = _get_embedding_device()
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    try:
        test_vec = embeddings.embed_query("test")
        print(f"✅ Embedding 模型 {embedding_model} 就绪，维度 {len(test_vec)}（device={device}）")
    except Exception as e:
        print(f"❌ Embedding 连接失败：{e}")
        return None

    vector_store = None
    total = len(chunks)
    batch_count = (total + batch_size - 1) // batch_size

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        batch_no = i // batch_size + 1

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir,
            )
        else:
            vector_store.add_documents(batch)

        print(f"  📦 批次 {batch_no}/{batch_count}: {len(batch)} chunks ({min(i + batch_size, total)}/{total})")

    print(f"💾 向量数据库已保存到 {persist_dir}")
    print(f"📊 共入库 {total} 个文本块")
    return vector_store


def get_new_papers() -> list[str]:
    if not os.path.exists(PDF_DIR):
        print(f"❌ 目录不存在：{PDF_DIR}")
        return []

    dedup = DedupManager()
    papers = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not papers:
        print("📭 papers/ 目录下没有 PDF 文件")
        return []

    new_papers = []
    for paper in papers:
        file_path = os.path.join(PDF_DIR, paper)
        if add_with_dedup(file_path, dedup):
            new_papers.append(file_path)
            dedup.add_record(file_path)
        else:
            print(f"⏭️  跳过已入库：{paper}")

    return new_papers


def main():
    print("=" * 50)
    print("🔨 论文知识库构建工具")
    print("=" * 50)

    new_papers = get_new_papers()
    if not new_papers:
        print("📭 没有新论文需要入库")
        return

    print(f"📄 发现 {len(new_papers)} 篇新论文，开始加载...")
    documents = load_pdfs(new_papers)
    if not documents:
        return

    print(f"\n✂️ 文本切分...")
    chunks = split_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)

    print(f"\n🧠 分批向量化并入库（batch_size={BATCH_SIZE}）...")
    vs = build_vector_store(chunks, EMBEDDING_MODEL, CHROMA_DB_DIR)

    if vs:
        print("\n✅ 论文入库完成！")
        print(f"   新增论文：{len(new_papers)} 篇")
        for fp in new_papers:
            print(f"     · {os.path.basename(fp)}")
        print(f"   入库块数：{len(chunks)}")
    else:
        print("\n❌ 入库失败")


if __name__ == "__main__":
    main()
