from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable


@dataclass(frozen=True)
class QueryVariantFilterResult:
    variants: list[str]
    rejections: list[dict[str, Any]]


def _response_text(response: Any) -> str:
    return response.content if hasattr(response, "content") else str(response)


def _parse_query_variants(text: str, original: str, n_variants: int) -> list[str]:
    variants: list[str] = []
    seen = {original.strip().casefold()}
    label_only = re.compile(
        r"^(查询|query|variant|改写|改写查询|检索词)\s*\d*\s*[:：]?$",
        re.IGNORECASE,
    )
    label_prefix = re.compile(
        r"^(查询|query|variant|改写|改写查询|检索词)\s*\d+\s*[:：]\s*",
        re.IGNORECASE,
    )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[\).、]\s*", "", line)
        line = label_prefix.sub("", line)
        line = line.strip().strip("\"'`")
        # 小模型偶尔输出“查询1:”这类标签行，不能让它消耗一次召回。
        if not line or label_only.match(line):
            continue

        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        variants.append(line)
        if len(variants) >= n_variants:
            break

    return variants


def expand_query(original: str, llm: Any, n_variants: int = 2) -> list[str]:
    """生成检索用 query 变体；返回值不包含原始 query。"""
    if llm is None or n_variants <= 0:
        return []

    prompt = f"""
你是论文 RAG 检索查询改写器。请把用户问题改写成 {n_variants} 个互补检索 query。

要求：
- 保留原始问题的核心语义，不要回答问题。
- 生成的 query 要互补但不发散：可补充术语、缩写、英文表达，但不能换成无关主题。
- 优先补充中英文术语、缩写全称、论文常用同义表达。
- 每行只输出一个 query。
- 不要输出解释、标题或多余文本。

示例：
原问题：BERT 为什么叫双向 Transformer？
可输出：BERT bidirectional Transformer masked language modeling evidence
可输出：BERT 双向编码器 Transformer 预训练目标

反例：
不要输出：Transformer 教程
不要输出：深度学习历史

原始问题：
{original}
""".strip()

    response = llm.invoke(prompt)
    return _parse_query_variants(_response_text(response), original, n_variants)


def filter_query_variants(
    original: str,
    variants: list[str],
    *,
    embed_fn: Callable[[str], list[float]] | None = None,
    enabled: bool = False,
    min_similarity: float = 0.35,
    max_similarity: float = 0.98,
) -> QueryVariantFilterResult:
    if not enabled or embed_fn is None or not variants:
        return QueryVariantFilterResult(variants=list(variants), rejections=[])

    try:
        original_vector = embed_fn(original)
    except Exception as exc:
        return QueryVariantFilterResult(
            variants=list(variants),
            rejections=[{"variant": "", "reason": "embedding_unavailable", "error": f"{type(exc).__name__}: {exc}"}],
        )

    kept: list[str] = []
    rejections: list[dict[str, Any]] = []
    for variant in variants:
        try:
            similarity = _cosine_similarity(original_vector, embed_fn(variant))
        except Exception as exc:
            rejections.append({"variant": variant, "reason": "embedding_unavailable", "error": f"{type(exc).__name__}: {exc}"})
            kept.append(variant)
            continue

        rounded = round(similarity, 4)
        if similarity >= max_similarity:
            rejections.append({"variant": variant, "reason": "too_similar", "similarity": rounded})
        elif similarity < min_similarity:
            rejections.append({"variant": variant, "reason": "too_distant", "similarity": rounded})
        else:
            kept.append(variant)

    return QueryVariantFilterResult(variants=kept, rejections=rejections)


def query_variant_embed_fn_from_hybrid(hybrid: Any) -> Callable[[str], list[float]] | None:
    embeddings = getattr(getattr(hybrid, "weight_decider", None), "embeddings", None)
    embed_query = getattr(embeddings, "embed_query", None)
    if callable(embed_query):
        return embed_query
    return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    return dot / (norm_a * norm_b + 1e-8)
