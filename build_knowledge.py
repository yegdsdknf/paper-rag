# 模块一 入库
# 负责：读取 PDF → 文本切分 → 向量化 → 分批存入 Chroma
import argparse
import os
import re
import time
from dataclasses import dataclass, replace

try:
    import torch
except ImportError:
    torch = None

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from utils.dedup_manager import DedupManager, add_with_dedup
from utils.config_loader import load_config
from paper_rag.config import RagSettings
from paper_rag.indexing import build_index_manifest, collect_vision_summary_docs, save_index_manifest
from paper_rag.indexing.chunking import split_documents as split_documents_by_strategy
from paper_rag.indexing.pdf_text import analyze_pdf_text_quality
from paper_rag.retrieval.hybrid import HybridRetriever

config = load_config()
settings = RagSettings.from_mapping(config)
PDF_DIR = "./papers"
CHROMA_DB_DIR = settings.persist_directory
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
EMBEDDING_MODEL = settings.embedding_model
BATCH_SIZE = 64
DEFAULT_DEDUP_RECORD_PATH = "./data/md5_records.json"

# 从配置文件读取跳过页面（config.yaml 中未配置则为空字典）
SKIP_PAGES = settings.skip_pages


@dataclass(frozen=True)
class BuildPlan:
    settings: RagSettings
    dedup_record_path: str
    batch_size: int = BATCH_SIZE
    rebuild: bool = False


def _slugify_experiment_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return slug or "experiment"


def parse_build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 Paper RAG 知识库")
    parser.add_argument("--experiment", help="实验名称；启用后使用独立 Chroma 目录、collection 和去重记录")
    parser.add_argument("--chunk-strategy", choices=[
        "recursive_character",
        "section_aware",
        "semantic",
        "hybrid_section_semantic",
    ])
    parser.add_argument("--chunk-schema-version")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="向量化和 Chroma 写入批大小")
    parser.add_argument("--rebuild", action="store_true", help="忽略实验去重记录，并在写入前重置目标 collection")
    args = parser.parse_args(argv)
    if args.rebuild and not args.experiment:
        parser.error("--rebuild 必须和 --experiment 一起使用，避免误重置默认知识库")
    if args.batch_size <= 0:
        parser.error("--batch-size 必须是正整数")
    return args


def resolve_build_plan(base_settings: RagSettings, args: argparse.Namespace) -> BuildPlan:
    next_settings = base_settings
    if args.chunk_strategy:
        next_settings = replace(next_settings, chunk_strategy=args.chunk_strategy)
    if args.chunk_schema_version:
        next_settings = replace(next_settings, chunk_schema_version=args.chunk_schema_version)

    dedup_record_path = DEFAULT_DEDUP_RECORD_PATH
    if args.experiment:
        slug = _slugify_experiment_name(args.experiment)
        next_settings = replace(
            next_settings,
            persist_directory=f"./chroma_db_experiments/{slug}",
            collection_name=f"langchain_{slug}",
        )
        dedup_record_path = f"./data/md5_records_{slug}.json"

    return BuildPlan(
        settings=next_settings,
        dedup_record_path=dedup_record_path,
        batch_size=int(args.batch_size),
        rebuild=bool(args.rebuild),
    )

def _get_embedding_device() -> str:
    """优先使用 GPU；如果不可用则回退到 CPU。"""
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pdfs_with_quality(
    file_paths: list[str],
    active_settings: RagSettings = settings,
) -> tuple[list[Document], list[dict[str, object]]]:
    documents = []
    quality_reports: list[dict[str, object]] = []
    for fp in file_paths:
        loader = PyPDFLoader(fp)
        docs = loader.load()
        fname_lower = os.path.basename(fp).lower()
        for key, pages in active_settings.skip_pages.items():
            if key in fname_lower:
                before = len(docs)
                page_set = set(pages)
                docs = [d for d in docs if d.metadata.get("page", 0) not in page_set]
                print(f"  🧹 {os.path.basename(fp)} — 跳过 {before - len(docs)} 页附录")
        docs, noisy_pages = analyze_pdf_text_quality(docs)
        if noisy_pages:
            pages = ", ".join(str(item["page"]) for item in noisy_pages)
            print(f"  🧹 {os.path.basename(fp)} — 跳过 {len(noisy_pages)} 页 PDF 编码噪声页：{pages}")
            quality_reports.extend(noisy_pages)
        documents.extend(docs)
        print(f"  ✅ {os.path.basename(fp)} — {len(docs)} 页")
    print(f"📄 成功加载 {len(documents)} 页文档")
    return documents, quality_reports


def load_pdfs(file_paths: list[str], active_settings: RagSettings = settings) -> list[Document]:
    documents, _quality_reports = load_pdfs_with_quality(file_paths, active_settings)
    return documents


def _create_embeddings(embedding_model: str):
    device = _get_embedding_device()
    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": device, "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_documents(
    documents,
    chunk_size,
    chunk_overlap,
    embeddings=None,
    source_file_hashes=None,
    active_settings: RagSettings = settings,
):
    chunks = split_documents_by_strategy(
        documents,
        active_settings,
        embeddings=embeddings,
        source_file_hashes=source_file_hashes,
    )
    print(
        f"🔪 切分为 {len(chunks)} 个文本块"
        f"（strategy={active_settings.chunk_strategy}, chunk_size={chunk_size}, overlap={chunk_overlap}）"
    )
    return chunks


def build_vector_store(
    chunks,
    embedding_model: str,
    persist_dir: str,
    batch_size: int = BATCH_SIZE,
    collection_name: str = "langchain",
    embeddings=None,
):
    device = _get_embedding_device()
    embeddings = embeddings or _create_embeddings(embedding_model)
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
        batch = prepare_chunks_for_chroma(chunks[i : i + batch_size])
        batch_no = i // batch_size + 1
        batch_start = time.perf_counter()
        first = batch[0].metadata if batch else {}
        last = batch[-1].metadata if batch else {}
        print(
            "  ▶️  开始批次 "
            f"{batch_no}/{batch_count}: {len(batch)} chunks "
            f"({i + 1}-{min(i + batch_size, total)}/{total}) "
            f"from {first.get('source_file') or os.path.basename(str(first.get('source', '')))}:"
            f"{first.get('page', '?')} -> "
            f"{last.get('source_file') or os.path.basename(str(last.get('source', '')))}:"
            f"{last.get('page', '?')}",
            flush=True,
        )

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name=collection_name,
            )
        else:
            vector_store.add_documents(batch)

        elapsed = time.perf_counter() - batch_start
        print(
            f"  📦 完成批次 {batch_no}/{batch_count}: {len(batch)} chunks "
            f"({min(i + batch_size, total)}/{total}) · {elapsed:.1f}s",
            flush=True,
        )

    print(f"💾 向量数据库已保存到 {persist_dir}")
    print(f"📊 共入库 {total} 个文本块")
    return vector_store


def prepare_chunks_for_chroma(chunks: list[Document]) -> list[Document]:
    """Chroma metadata 只稳定支持标量；列表/字典用 JSON 字符串保留信息。"""
    prepared: list[Document] = []
    for chunk in chunks:
        metadata = {}
        for key, value in dict(chunk.metadata).items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            elif isinstance(value, (list, tuple, dict)):
                metadata[key] = json_dumps_metadata(value)
            else:
                metadata[key] = str(value)
        prepared.append(Document(page_content=chunk.page_content, metadata=metadata))
    return prepared


def json_dumps_metadata(value) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def get_new_papers(dedup_record_path: str = DEFAULT_DEDUP_RECORD_PATH, rebuild: bool = False) -> list[str]:
    if not os.path.exists(PDF_DIR):
        print(f"❌ 目录不存在：{PDF_DIR}")
        return []

    dedup = DedupManager(record_path=dedup_record_path)
    papers = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not papers:
        print("📭 papers/ 目录下没有 PDF 文件")
        return []

    new_papers = []
    for paper in papers:
        file_path = os.path.join(PDF_DIR, paper)
        if rebuild or add_with_dedup(file_path, dedup):
            new_papers.append(file_path)
        else:
            print(f"⏭️  跳过已入库：{paper}")

    return new_papers


def mark_papers_indexed(file_paths: list[str], dedup_record_path: str = DEFAULT_DEDUP_RECORD_PATH) -> None:
    dedup = DedupManager(record_path=dedup_record_path)
    for file_path in file_paths:
        dedup.add_record(file_path)


def reset_collection_if_requested(plan: BuildPlan) -> None:
    if not plan.rebuild:
        return
    try:
        import chromadb

        client = chromadb.PersistentClient(path=plan.settings.persist_directory)
        try:
            client.delete_collection(plan.settings.collection_name)
            print(f"♻️  已重置 collection：{plan.settings.collection_name}")
        except Exception as exc:
            message = str(exc).lower()
            if "not found" in message or "does not exist" in message:
                print(f"ℹ️  collection 不存在，将创建新 collection：{plan.settings.collection_name}")
                return
            raise RuntimeError(f"重置 Chroma collection 失败: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"实验重建失败，未写入向量库：{type(exc).__name__}: {exc}") from exc


def prebuild_bm25_cache(vector_store, active_settings: RagSettings) -> None:
    """入库成功后预构建 BM25 磁盘缓存，减少下一次 Web 启动等待。"""
    try:
        hybrid = HybridRetriever(
            vector_store=vector_store,
            top_k=active_settings.k,
            default_vector_weight=active_settings.default_vector_weight,
            default_bm25_weight=active_settings.default_bm25_weight,
            persist_directory=active_settings.persist_directory,
            collection_name=active_settings.collection_name,
            chunk_schema_version=active_settings.chunk_schema_version,
            index_manifest_filename=active_settings.index_manifest_filename,
        )
        hybrid.build_bm25_retriever()
        print("✅ BM25 缓存已预构建")
    except Exception as exc:
        print(f"⚠️  BM25 缓存预构建失败，将在首次查询时重建：{type(exc).__name__}: {exc}")


def main(argv: list[str] | None = None):
    args = parse_build_args(argv)
    plan = resolve_build_plan(settings, args)
    active_settings = plan.settings

    print("=" * 50)
    print("🔨 论文知识库构建工具")
    print("=" * 50)
    if args.experiment:
        print(f"🧪 实验模式：{args.experiment}")
        print(f"   persist_directory: {active_settings.persist_directory}")
        print(f"   collection_name: {active_settings.collection_name}")
        print(f"   dedup_record_path: {plan.dedup_record_path}")
    print(f"   chunk_strategy: {active_settings.chunk_strategy}")
    print(f"   chunk_schema_version: {active_settings.chunk_schema_version}")

    reset_collection_if_requested(plan)

    new_papers = get_new_papers(plan.dedup_record_path, rebuild=plan.rebuild)
    if not new_papers:
        print("📭 没有新论文需要入库")
        return

    print(f"📄 发现 {len(new_papers)} 篇新论文，开始加载...")
    file_hashes = {fp: DedupManager.compute_md5(fp) for fp in new_papers}
    source_file_hashes = {os.path.basename(fp): digest for fp, digest in file_hashes.items()}

    documents, quality_reports = load_pdfs_with_quality(new_papers, active_settings)

    embeddings = None
    if documents and active_settings.chunk_strategy in {"semantic", "hybrid_section_semantic"}:
        embeddings = _create_embeddings(EMBEDDING_MODEL)

    chunks: list[Document] = []
    if documents:
        print(f"\n✂️ 文本切分...")
        chunks = split_documents(
            documents,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            embeddings=embeddings,
            source_file_hashes=source_file_hashes,
            active_settings=active_settings,
        )
    vision_docs, vision_stats = collect_vision_summary_docs(
        pdf_paths=new_papers,
        settings=active_settings,
        file_hashes=file_hashes,
        quality_reports=quality_reports,
    )
    if active_settings.enable_vision_analysis:
        print(
            "\n👁️  视觉入库："
            f"selected={vision_stats['pages_selected']}, "
            f"generated={vision_stats['generated']}, "
            f"cache_hits={vision_stats['cache_hits']}, "
            f"quality={vision_stats['quality_flags']}, "
            f"reasons={vision_stats['trigger_reasons']}"
        )
    if vision_docs:
        chunks.extend(vision_docs)
        print(f"🔎 追加 {len(vision_docs)} 个 vision_summary chunk")

    print(f"\n🧠 分批向量化并入库（batch_size={plan.batch_size}）...")
    vs = build_vector_store(
        chunks,
        EMBEDDING_MODEL,
        active_settings.persist_directory,
        batch_size=plan.batch_size,
        collection_name=active_settings.collection_name,
        embeddings=embeddings,
    )

    if vs:
        manifest = build_index_manifest(
            settings=active_settings,
            source_files=new_papers,
            file_hashes=file_hashes,
            chunk_count=len(chunks),
            embedding_device=_get_embedding_device(),
            vision_stats=vision_stats,
        )
        manifest_file = save_index_manifest(manifest, active_settings)
        prebuild_bm25_cache(vs, active_settings)
        mark_papers_indexed(new_papers, plan.dedup_record_path)
        print("\n✅ 论文入库完成！")
        print(f"   新增论文：{len(new_papers)} 篇")
        for fp in new_papers:
            print(f"     · {os.path.basename(fp)}")
        print(f"   入库块数：{len(chunks)}")
        print(f"   索引版本：{manifest['index_version']}")
        print(f"   Manifest：{manifest_file}")
    else:
        print("\n❌ 入库失败")


if __name__ == "__main__":
    main()
