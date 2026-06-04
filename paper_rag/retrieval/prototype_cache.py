from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np


CACHE_FORMAT_VERSION = 1
DEFAULT_ANCHOR_VERSION = "semantic-weight-anchors-v1"


def prototype_cache_path(
    cache_dir: str | os.PathLike[str] = "data/prototypes",
    embedding_model_id: str = "unknown",
    anchor_version: str = DEFAULT_ANCHOR_VERSION,
) -> Path:
    key = f"{CACHE_FORMAT_VERSION}|{embedding_model_id}|{anchor_version}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{digest}.npz"


def load_prototype_cache(
    cache_dir: str | os.PathLike[str],
    embedding_model_id: str,
    anchor_version: str = DEFAULT_ANCHOR_VERSION,
) -> tuple[np.ndarray, np.ndarray] | None:
    path = prototype_cache_path(cache_dir, embedding_model_id, anchor_version)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["cache_format_version"]) != CACHE_FORMAT_VERSION:
                return None
            if str(payload["embedding_model_id"]) != embedding_model_id:
                return None
            if str(payload["anchor_version"]) != anchor_version:
                return None
            precise = np.asarray(payload["prototype_precise"], dtype=float)
            semantic = np.asarray(payload["prototype_semantic"], dtype=float)
            if precise.shape != semantic.shape or precise.ndim != 1 or precise.size == 0:
                return None
            return precise, semantic
    except Exception as exc:
        print(f"[WARN] Prototype cache unreadable; rebuilding: {type(exc).__name__}: {exc}")
        return None


def save_prototype_cache(
    cache_dir: str | os.PathLike[str],
    embedding_model_id: str,
    prototype_precise: np.ndarray,
    prototype_semantic: np.ndarray,
    anchor_version: str = DEFAULT_ANCHOR_VERSION,
) -> Path:
    path = prototype_cache_path(cache_dir, embedding_model_id, anchor_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez(
        tmp_path,
        cache_format_version=np.array(CACHE_FORMAT_VERSION),
        embedding_model_id=np.array(embedding_model_id),
        anchor_version=np.array(anchor_version),
        embedding_dim=np.array(int(np.asarray(prototype_precise).size)),
        prototype_precise=np.asarray(prototype_precise, dtype=float),
        prototype_semantic=np.asarray(prototype_semantic, dtype=float),
    )
    os.replace(tmp_path, path)
    return path


def embedding_model_id(embeddings: Any) -> str:
    for attr in ("model_name", "model_id", "model"):
        value = getattr(embeddings, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(embeddings).__name__
