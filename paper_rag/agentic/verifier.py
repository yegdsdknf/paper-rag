from __future__ import annotations

import contextlib
import io
import re
from typing import Any

from langchain_core.documents import Document

from paper_rag.agentic.json_utils import parse_json_object
from paper_rag.agentic.schema import (
    VALID_STATUSES,
    VerifiedEvidence,
    docs_to_agent_sources,
    normalize_verified_evidence,
)
from utils.prompt_loader import load_prompt


_TERM_RE = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[\u4e00-\u9fff]+")
_CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")
_ASCII_ALPHA_RE = re.compile(r"^[A-Za-z]+$")
_MAX_EVIDENCE_DOCS = 4
_MAX_EVIDENCE_CHARS = 800


def verify_goal(goal: dict[str, Any], docs: list[Document], llm: Any | None = None) -> VerifiedEvidence:
    claim = str(goal.get("claim") or "").strip()
    query = str(goal.get("query") or "").strip()
    fallback_status = _keyword_status(claim or query, docs)
    status = fallback_status
    missing_terms: list[str] = []

    if llm is not None and docs:
        try:
            prompt_template = _load_verifier_prompt()
            if not prompt_template:
                raise ValueError("agent verifier prompt is empty")

            prompt = prompt_template.format(
                claim=claim,
                query=query,
                evidence=_format_evidence(docs[:_MAX_EVIDENCE_DOCS]),
            )
            parsed = parse_json_object(_response_text(llm.invoke(prompt)))
            parsed_status = str(parsed.get("status") or "").strip()
            if parsed_status in VALID_STATUSES:
                status = parsed_status
            missing_terms = _clean_missing_terms(parsed.get("missing_terms"))
        except Exception:
            status = fallback_status
            missing_terms = []

    raw_result: dict[str, Any] = {
        "goal_id": str(goal.get("id") or "").strip(),
        "claim": claim,
        "status": status,
        "supporting_sources": docs_to_agent_sources(docs[:_MAX_EVIDENCE_DOCS]) if status != "unsupported" else [],
        "missing_terms": missing_terms,
    }
    return normalize_verified_evidence(raw_result, goal_id=raw_result["goal_id"])


def _keyword_status(text: str, docs: list[Document]) -> str:
    terms = _extract_terms(text)
    if not terms or not docs:
        return "unsupported"

    evidence_terms = set(_extract_terms("\n".join(str(doc.page_content or "") for doc in docs)))
    hits = sum(1 for term in terms if term in evidence_terms)
    if hits == len(terms):
        return "supported"
    if hits > 0:
        return "partial"
    return "unsupported"


def _extract_terms(text: str) -> list[str]:
    seen: set[str] = set()
    terms: list[str] = []
    for match in _TERM_RE.findall(str(text or "")):
        for term in _expand_match_terms(match):
            if term in seen:
                continue
            seen.add(term)
            terms.append(term)
    return terms


def _expand_match_terms(match: str) -> list[str]:
    if _CJK_RE.fullmatch(match):
        return _cjk_ngrams(match)

    if _ASCII_ALPHA_RE.fullmatch(match) and len(match) < 3 and not match.isupper():
        return []
    return [match.lower()]


def _cjk_ngrams(text: str) -> list[str]:
    chars = list(text)
    if len(chars) < 2:
        return chars

    terms: list[str] = []
    max_n = min(4, len(chars))
    for size in range(2, max_n + 1):
        for start in range(0, len(chars) - size + 1):
            terms.append("".join(chars[start : start + size]))
    return terms


def _format_evidence(docs: list[Document]) -> str:
    chunks: list[str] = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source") or metadata.get("source_file") or "unknown"
        page = metadata.get("page", "unknown")
        content = " ".join(str(doc.page_content or "").split())[:_MAX_EVIDENCE_CHARS]
        chunks.append(f"[{index}] source={source} page={page}\n{content}")
    return "\n\n".join(chunks)


def _clean_missing_terms(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    cleaned: list[str] = []
    for item in value:
        if item is None:
            continue
        term = str(item).strip()
        if term:
            cleaned.append(term)
    return cleaned


def _response_text(response: Any) -> str:
    return str(response.content if hasattr(response, "content") else response)


def _load_verifier_prompt() -> str:
    with contextlib.redirect_stdout(io.StringIO()):
        return load_prompt("agent_verifier_prompt")
