from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping


CACHE_FORMAT_VERSION = 1
RETRIEVER_FILENAME = "bm25_retriever.pkl"
METADATA_FILENAME = "bm25_metadata.json"


def _source_fingerprint(manifest: Mapping[str, Any] | None) -> str:
    if not manifest:
        return ""

    sources = manifest.get("source_files") or []
    normalized = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        normalized.append(
            {
                "filename": str(source.get("filename", "")),
                "file_hash": str(source.get("file_hash", "")),
                "path": str(source.get("path", "")),
            }
        )
    payload = json.dumps(sorted(normalized, key=lambda item: item["filename"]), ensure_ascii=False, sort_keys=True)
    return sha256(payload.encode("utf-8")).hexdigest() if normalized else ""


def build_bm25_cache_metadata(
    *,
    persist_directory: str,
    collection_name: str,
    chunk_schema_version: str,
    doc_count: int,
    top_k: int,
    manifest: Mapping[str, Any] | None = None,
    bm25_class: str = "BM25Retriever",
) -> dict[str, Any]:
    manifest_count = manifest.get("chunk_count") if manifest else None
    manifest_schema = manifest.get("chunk_schema_version") if manifest else None
    return {
        "collection_name": str(collection_name),
        "persist_directory": str(persist_directory),
        "chunk_schema_version": str(manifest_schema or chunk_schema_version),
        "chunk_count": int(manifest_count if manifest_count is not None else doc_count),
        "top_k": int(top_k),
        "source_fingerprint": _source_fingerprint(manifest),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cache_format_version": CACHE_FORMAT_VERSION,
        "bm25_class": str(bm25_class),
    }


def bm25_cache_dir(persist_directory: str | Path) -> Path:
    return Path(persist_directory) / "bm25_cache"


def _metadata_matches(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    keys = [
        "collection_name",
        "persist_directory",
        "chunk_schema_version",
        "chunk_count",
        "top_k",
        "source_fingerprint",
        "cache_format_version",
    ]
    return all(stored.get(key) == expected.get(key) for key in keys)


def load_bm25_cache(cache_dir: str | Path, expected_metadata: Mapping[str, Any]) -> Any | None:
    path = Path(cache_dir)
    metadata_path = path / METADATA_FILENAME
    retriever_path = path / RETRIEVER_FILENAME
    if not metadata_path.exists() or not retriever_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] BM25 cache metadata unreadable; rebuilding: {type(exc).__name__}: {exc}")
        return None

    if not _metadata_matches(metadata, expected_metadata):
        print("[WARN] BM25 cache metadata mismatch; rebuilding")
        return None

    try:
        with retriever_path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"[WARN] BM25 cache unreadable; rebuilding: {type(exc).__name__}: {exc}")
        return None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def save_bm25_cache(cache_dir: str | Path, retriever: Any, metadata: Mapping[str, Any]) -> None:
    path = Path(cache_dir)
    _atomic_write_bytes(path / RETRIEVER_FILENAME, pickle.dumps(retriever))
    _atomic_write_text(
        path / METADATA_FILENAME,
        json.dumps(dict(metadata), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
