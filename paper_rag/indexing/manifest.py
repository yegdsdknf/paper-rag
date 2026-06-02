from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from paper_rag.config import get_setting


DEFAULT_INDEX_MANIFEST_FILENAME = "index_manifest.json"


def _compatibility_payload(settings: Any) -> dict[str, Any]:
    """只纳入会影响索引兼容性的配置，避免时间戳导致版本漂移。"""
    return {
        "collection_name": get_setting(settings, "collection_name", "langchain"),
        "embedding_model": get_setting(settings, "embedding_model", ""),
        "chunk_strategy": get_setting(settings, "chunk_strategy", "recursive_character"),
        "chunk_schema_version": get_setting(settings, "chunk_schema_version", "v1"),
        "chunk_size": int(get_setting(settings, "chunk_size", 0)),
        "chunk_overlap": int(get_setting(settings, "chunk_overlap", 0)),
        "separators": list(get_setting(settings, "separators", []) or []),
        "section_heading_detection": bool(get_setting(settings, "section_heading_detection", True)),
        "section_min_chars": int(get_setting(settings, "section_min_chars", 120)),
        "section_max_chars": int(get_setting(settings, "section_max_chars", 900)),
        "preserve_abstract": bool(get_setting(settings, "preserve_abstract", True)),
        "references_policy": get_setting(settings, "references_policy", "keep_with_metadata"),
        "semantic_similarity_threshold": float(get_setting(settings, "semantic_similarity_threshold", 0.7)),
        "semantic_min_chars": int(get_setting(settings, "semantic_min_chars", 180)),
        "semantic_max_chars": int(get_setting(settings, "semantic_max_chars", 900)),
        "enable_vision_analysis": bool(get_setting(settings, "enable_vision_analysis", False)),
        "vision_model": get_setting(settings, "vision_model", "qwen2.5vl:3b"),
        "vision_prompt_version": get_setting(settings, "vision_prompt_version", "v1"),
        "vision_trigger_policy": get_setting(settings, "vision_trigger_policy", "noisy_or_figure_page"),
        "vision_visual_density_threshold": float(get_setting(settings, "vision_visual_density_threshold", 0.35)),
    }


def build_index_version(settings: Any) -> str:
    payload = json.dumps(_compatibility_payload(settings), ensure_ascii=False, sort_keys=True)
    return f"idx_{sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def manifest_path(settings: Any) -> Path:
    filename = get_setting(settings, "index_manifest_filename", DEFAULT_INDEX_MANIFEST_FILENAME)
    return Path(get_setting(settings, "persist_directory", "./chroma_db")) / str(filename)


def build_index_manifest(
    settings: Any,
    source_files: list[str],
    file_hashes: Mapping[str, str] | None,
    chunk_count: int,
    embedding_device: str,
    created_at: str | None = None,
    vision_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    created = created_at or datetime.now(timezone.utc).isoformat()
    hashes = file_hashes or {}
    sources = []
    for file_path in source_files:
        path = Path(file_path)
        sources.append(
            {
                "filename": path.name,
                "path": str(path),
                "file_hash": hashes.get(file_path) or hashes.get(str(path)) or "",
            }
        )

    manifest = {
        "index_version": build_index_version(settings),
        "created_at": created,
        "collection_name": get_setting(settings, "collection_name", "langchain"),
        "persist_directory": get_setting(settings, "persist_directory", "./chroma_db"),
        "embedding_model": get_setting(settings, "embedding_model", ""),
        "embedding_device": embedding_device,
        "chunk_strategy": get_setting(settings, "chunk_strategy", "recursive_character"),
        "chunk_schema_version": get_setting(settings, "chunk_schema_version", "v1"),
        "chunk_size": int(get_setting(settings, "chunk_size", 0)),
        "chunk_overlap": int(get_setting(settings, "chunk_overlap", 0)),
        "separators": list(get_setting(settings, "separators", []) or []),
        "section_heading_detection": bool(get_setting(settings, "section_heading_detection", True)),
        "section_min_chars": int(get_setting(settings, "section_min_chars", 120)),
        "section_max_chars": int(get_setting(settings, "section_max_chars", 900)),
        "preserve_abstract": bool(get_setting(settings, "preserve_abstract", True)),
        "references_policy": get_setting(settings, "references_policy", "keep_with_metadata"),
        "semantic_similarity_threshold": float(get_setting(settings, "semantic_similarity_threshold", 0.7)),
        "semantic_min_chars": int(get_setting(settings, "semantic_min_chars", 180)),
        "semantic_max_chars": int(get_setting(settings, "semantic_max_chars", 900)),
        "chunk_count": int(chunk_count),
        "source_files": sources,
    }
    manifest["vision_analysis"] = {
        "enabled": bool(get_setting(settings, "enable_vision_analysis", False)),
        "model": get_setting(settings, "vision_model", "qwen2.5vl:3b"),
        "prompt_version": get_setting(settings, "vision_prompt_version", "v1"),
        "cache_dir": get_setting(settings, "vision_cache_dir", "./data/vision_cache"),
        "force_refresh": bool(get_setting(settings, "vision_force_refresh", False)),
        "trigger_policy": get_setting(settings, "vision_trigger_policy", "noisy_or_figure_page"),
        "visual_density_threshold": float(get_setting(settings, "vision_visual_density_threshold", 0.35)),
        "max_pages_per_doc": int(get_setting(settings, "vision_max_pages_per_doc", 20)),
        "force_pages": dict(get_setting(settings, "vision_force_pages", {}) or {}),
        "stats": dict(vision_stats or {}),
    }
    return manifest


def save_index_manifest(manifest: Mapping[str, Any], settings: Any) -> Path:
    path = manifest_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_index_manifest(settings: Any) -> dict[str, Any] | None:
    path = manifest_path(settings)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_index_version(settings: Any) -> str:
    manifest = load_index_manifest(settings)
    if manifest and manifest.get("index_version"):
        return str(manifest["index_version"])
    return build_index_version(settings)
