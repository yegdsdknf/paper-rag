"""兼容薄壳：新代码请从 paper_rag.retrieval.hybrid 导入。"""

from paper_rag.config import RagSettings
from paper_rag.indexing.manifest import load_index_manifest as _load_index_manifest
from paper_rag.retrieval.hybrid import (
    PRECISE_ANCHORS,
    SEMANTIC_ANCHORS,
    HybridRetriever as _PackageHybridRetriever,
    SemanticWeightDecider,
)
from utils.config_loader import load_config

config = load_config()
load_index_manifest = _load_index_manifest


class HybridRetriever(_PackageHybridRetriever):
    """根目录兼容包装；真实实现位于 paper_rag.retrieval.hybrid。"""

    def _load_index_manifest(self) -> dict | None:
        if not self.persist_directory:
            return None
        try:
            settings = RagSettings.from_mapping(
                {
                    **config,
                    "persist_directory": self.persist_directory,
                    "collection_name": self.collection_name,
                    "chunk_schema_version": self.chunk_schema_version,
                    "index_manifest_filename": self.index_manifest_filename,
                }
            )
            return load_index_manifest(settings)
        except Exception as exc:
            print(f"[WARN] BM25 cache manifest unavailable; using doc count only: {type(exc).__name__}: {exc}")
            return None


__all__ = [
    "PRECISE_ANCHORS",
    "SEMANTIC_ANCHORS",
    "HybridRetriever",
    "SemanticWeightDecider",
    "load_index_manifest",
]
