from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


class Reranker:
    """本地优先的 Cross-Encoder 精排器，按需加载模型。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        model: Any | None = None,
    ):
        self.model_name = model_name
        self.device = device
        self.model = model
        self.unavailable_error: str | None = None

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model
        if self.unavailable_error is not None:
            raise RuntimeError(self.unavailable_error)

        # 强制离线加载，避免基准评估时因为模型下载阻塞。
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

        model_path = self.model_name
        if not Path(self.model_name).exists():
            try:
                from huggingface_hub import snapshot_download

                model_path = snapshot_download(self.model_name, local_files_only=True)
            except Exception as exc:
                self.unavailable_error = f"local model not found: {self.model_name}"
                raise RuntimeError(self.unavailable_error) from exc

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers 未安装，无法启用 rerank") from exc

        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device

        # 精排阶段只使用本地文件，缺模型时交给上层降级处理。
        kwargs["model_kwargs"] = {"local_files_only": True}
        kwargs["processor_kwargs"] = {"local_files_only": True}
        try:
            self.model = CrossEncoder(model_path, **kwargs)
            return self.model
        except Exception as exc:
            # 缓存失败原因，避免每次 mixed 查询都重复尝试加载缺失模型。
            self.unavailable_error = f"{type(exc).__name__}: {exc}"
            raise

    def rerank(self, query: str, docs: list[Document], top_k: int | None = None) -> list[Document]:
        if not docs:
            return []

        model = self._load_model()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = [float(score) for score in model.predict(pairs)]

        ranked: list[tuple[float, int, Document]] = []
        for index, (score, doc) in enumerate(zip(scores, docs)):
            metadata = dict(doc.metadata)
            metadata["rerank_score"] = score
            ranked.append((score, index, Document(page_content=doc.page_content, metadata=metadata)))

        # 分数完全相同时保留原检索顺序，确保精排结果可复现。
        ranked.sort(key=lambda item: (-item[0], item[1]))
        limit = top_k if top_k is not None else len(ranked)
        return [doc for _, _, doc in ranked[:limit]]


_reranker_cache: dict[tuple[str, str | None], Reranker] = {}


def get_reranker(model_name: str, device: str | None = None) -> Reranker:
    key = (model_name, device)
    if key not in _reranker_cache:
        _reranker_cache[key] = Reranker(model_name=model_name, device=device)
    return _reranker_cache[key]


def apply_rerank(
    query: str,
    docs: list[Document],
    enabled: bool,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    top_k: int | None = None,
    device: str | None = None,
) -> list[Document]:
    if not enabled:
        return docs

    try:
        reranker = get_reranker(model_name=model_name, device=device)
        return reranker.rerank(query, docs, top_k=top_k)
    except Exception as exc:
        print(f"[WARN] Rerank unavailable; keeping original order: {type(exc).__name__}: {exc}")
        return docs
