from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class ConfigError(ValueError):
    """配置缺失或类型不符合预期时抛出，避免在业务链路深处才失败。"""


@dataclass(frozen=True)
class RagSettings:
    persist_directory: str
    embedding_model: str
    llm_model: str
    temperature: float
    k: int
    chunk_size: int
    chunk_overlap: int
    separators: list[str]
    chunk_strategy: str = "recursive_character"
    chunk_schema_version: str = "v1"
    collection_name: str = "langchain"
    index_manifest_filename: str = "index_manifest.json"
    llm_model_reasoning: str = "deepseek-r1:7b"
    llm_num_ctx: int = 4096
    llm_num_predict: int = 1024
    default_vector_weight: float = 0.5
    default_bm25_weight: float = 0.5
    enable_rerank: bool = False
    reranker_model: str = ""
    rerank_top_k: int = 6
    enable_query_expansion: bool = False
    query_expansion_model: str | None = None
    query_expansion_variants: int = 2
    query_expansion_max_multiplier: int = 3
    enable_context_compression: bool = False
    context_compression_max_sentences: int = 3
    enable_parent_retrieval: bool = False
    parent_max_chars_per_page: int = 2500
    enable_query_logging: bool = False
    query_log_path: str = "logs/query_runs.jsonl"
    semantic_similarity_threshold: float = 0.7
    skip_pages: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RagSettings":
        required = [
            "persist_directory",
            "embedding_model",
            "llm_model",
            "temperature",
            "k",
            "chunk_size",
            "chunk_overlap",
            "separators",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise ConfigError(f"缺少必要配置项: {', '.join(missing)}")

        return cls(
            persist_directory=str(data["persist_directory"]),
            embedding_model=str(data["embedding_model"]),
            llm_model=str(data["llm_model"]),
            temperature=float(data["temperature"]),
            k=int(data["k"]),
            chunk_size=int(data["chunk_size"]),
            chunk_overlap=int(data["chunk_overlap"]),
            separators=[str(item) for item in data["separators"]],
            chunk_strategy=str(data.get("chunk_strategy", "recursive_character")),
            chunk_schema_version=str(data.get("chunk_schema_version", "v1")),
            collection_name=str(data.get("collection_name", "langchain")),
            index_manifest_filename=str(data.get("index_manifest_filename", "index_manifest.json")),
            llm_model_reasoning=str(data.get("llm_model_reasoning", "deepseek-r1:7b")),
            llm_num_ctx=int(data.get("llm_num_ctx", 4096)),
            llm_num_predict=int(data.get("llm_num_predict", 1024)),
            default_vector_weight=float(data.get("default_vector_weight", 0.5)),
            default_bm25_weight=float(data.get("default_bm25_weight", 0.5)),
            enable_rerank=bool(data.get("enable_rerank", False)),
            reranker_model=str(data.get("reranker_model", "")),
            rerank_top_k=int(data.get("rerank_top_k", 6)),
            enable_query_expansion=bool(data.get("enable_query_expansion", False)),
            query_expansion_model=(
                str(data["query_expansion_model"]) if data.get("query_expansion_model") else None
            ),
            query_expansion_variants=int(data.get("query_expansion_variants", 2)),
            query_expansion_max_multiplier=int(data.get("query_expansion_max_multiplier", 3)),
            enable_context_compression=bool(data.get("enable_context_compression", False)),
            context_compression_max_sentences=int(data.get("context_compression_max_sentences", 3)),
            enable_parent_retrieval=bool(data.get("enable_parent_retrieval", False)),
            parent_max_chars_per_page=int(data.get("parent_max_chars_per_page", 2500)),
            enable_query_logging=bool(data.get("enable_query_logging", False)),
            query_log_path=str(data.get("query_log_path", "logs/query_runs.jsonl")),
            semantic_similarity_threshold=float(data.get("semantic_similarity_threshold", 0.7)),
            skip_pages=_normalize_skip_pages(data.get("skip_pages", {})),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "k": self.k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": list(self.separators),
            "chunk_strategy": self.chunk_strategy,
            "chunk_schema_version": self.chunk_schema_version,
            "embedding_model": self.embedding_model,
            "index_manifest_filename": self.index_manifest_filename,
            "llm_model": self.llm_model,
            "llm_model_reasoning": self.llm_model_reasoning,
            "temperature": self.temperature,
            "llm_num_ctx": self.llm_num_ctx,
            "llm_num_predict": self.llm_num_predict,
            "default_vector_weight": self.default_vector_weight,
            "default_bm25_weight": self.default_bm25_weight,
            "enable_rerank": self.enable_rerank,
            "reranker_model": self.reranker_model,
            "rerank_top_k": self.rerank_top_k,
            "enable_query_expansion": self.enable_query_expansion,
            "query_expansion_model": self.query_expansion_model,
            "query_expansion_variants": self.query_expansion_variants,
            "query_expansion_max_multiplier": self.query_expansion_max_multiplier,
            "enable_context_compression": self.enable_context_compression,
            "context_compression_max_sentences": self.context_compression_max_sentences,
            "enable_parent_retrieval": self.enable_parent_retrieval,
            "parent_max_chars_per_page": self.parent_max_chars_per_page,
            "enable_query_logging": self.enable_query_logging,
            "query_log_path": self.query_log_path,
            "semantic_similarity_threshold": self.semantic_similarity_threshold,
            "skip_pages": dict(self.skip_pages),
        }


def _normalize_skip_pages(value: Any) -> dict[str, list[int]]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, list[int]] = {}
    for key, pages in value.items():
        if not isinstance(pages, list):
            continue
        normalized[str(key)] = [int(page) for page in pages]
    return normalized
