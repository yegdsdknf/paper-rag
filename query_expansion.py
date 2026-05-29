from __future__ import annotations

import re
from typing import Any


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
- 优先补充中英文术语、缩写全称、论文常用同义表达。
- 每行只输出一个 query。
- 不要输出解释、标题或多余文本。

原始问题：
{original}
""".strip()

    response = llm.invoke(prompt)
    return _parse_query_variants(_response_text(response), original, n_variants)
