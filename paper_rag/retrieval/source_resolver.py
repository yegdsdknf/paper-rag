from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paper_rag.indexing.manifest import load_index_manifest


def normalize_source_signal(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def source_filename(metadata: dict[str, Any]) -> str:
    return os.path.basename(str(metadata.get("source", "")))


def source_stem_signal(filename: str) -> str:
    return normalize_source_signal(os.path.splitext(os.path.basename(filename))[0])


def _alias_candidates(filename: str, front_page_text: str = "") -> set[str]:
    stem = Path(filename).stem
    candidates = {
        stem,
        stem.replace("-", " "),
        stem.replace("_", " "),
        stem.replace("-", ""),
        stem.replace("_", ""),
    }
    compact = normalize_source_signal(stem)
    if compact:
        candidates.add(compact)

    # 首页标题常比文件名更接近用户表达；只取前两行，避免把摘要里的泛词都变成别名。
    for line in front_page_text.splitlines()[:2]:
        clean = line.strip(" \t\r\n:-")
        if 3 <= len(clean) <= 120:
            candidates.add(clean)

    return {candidate for candidate in candidates if normalize_source_signal(candidate)}


@dataclass(frozen=True)
class SourceEntry:
    filename: str
    aliases: tuple[str, ...]
    front_page_text: str = ""


class SourceResolver:
    def __init__(self, sources: list[SourceEntry]):
        self.sources = sources

    @classmethod
    def from_hybrid(cls, hybrid: Any, settings: Any | None = None) -> "SourceResolver":
        rows = _rows_from_vector_store(hybrid)
        if rows:
            return cls(_entries_from_rows(rows))

        manifest = load_index_manifest(settings) if settings is not None else None
        if manifest:
            return cls(_entries_from_manifest(manifest))

        return cls([])

    def resolve_source_files(self, question: str) -> list[str]:
        matched = self.explicit_source_matches(question)
        if matched:
            return matched

        origin = self.resolve_origin_source(question)
        return [origin] if origin else []

    def resolve_origin_source(self, question: str, require_question_signal: bool = True) -> str | None:
        q_signal = normalize_source_signal(question)
        if require_question_signal and "transformer" not in q_signal:
            return None

        scored: list[tuple[int, str]] = []
        for entry in self.sources:
            text = f"{Path(entry.filename).stem}\n{entry.front_page_text}".lower()
            score = 0
            # 用首页中的自描述识别“原始架构论文”，避免把所有 Transformer 家族论文都拉进来。
            if "we propose" in text and "transformer" in text:
                score += 4
            if "new simple network architecture" in text and "transformer" in text:
                score += 3
            if "the transformer" in text:
                score += 2
            if "attention" in text and "transformer" in text:
                score += 1
            if score:
                scored.append((score, entry.filename))

        if not scored:
            return None
        return sorted(scored, key=lambda item: (-item[0], item[1]))[0][1]

    def explicit_source_matches(self, question: str) -> list[str]:
        q_signal = normalize_source_signal(question)
        matches: list[tuple[int, str]] = []
        for entry in self.sources:
            positions = []
            for alias in entry.aliases:
                alias_signal = normalize_source_signal(alias)
                short_model_name = len(alias_signal) == 2 and any(char.isdigit() for char in alias_signal)
                if len(alias_signal) < 3 and not short_model_name:
                    continue
                pos = q_signal.find(alias_signal)
                if pos >= 0:
                    positions.append(pos)
            if positions:
                matches.append((min(positions), entry.filename))

        ordered = []
        seen = set()
        for _pos, filename in sorted(matches, key=lambda item: item[0]):
            if filename not in seen:
                seen.add(filename)
                ordered.append(filename)

        if len(ordered) >= 2 and _asks_for_commonality(question):
            origin = self.resolve_origin_source(question, require_question_signal=False)
            if origin and origin not in seen:
                ordered.append(origin)
        return ordered


def _asks_for_commonality(question: str) -> bool:
    q_lower = question.lower()
    return any(signal in q_lower for signal in ["共同点", "相同", "共性", "similarity", "similarities", "common"])


def _rows_from_vector_store(hybrid: Any) -> list[tuple[str, dict[str, Any]]]:
    vector_store = getattr(hybrid, "vector_store", None)
    if vector_store is None:
        return []

    try:
        stored = vector_store.get(include=["documents", "metadatas"])
    except Exception:
        return []

    return [
        (str(content), dict(metadata))
        for content, metadata in zip(stored.get("documents", []), stored.get("metadatas", []))
        if metadata and metadata.get("source")
    ]


def _entries_from_rows(rows: list[tuple[str, dict[str, Any]]]) -> list[SourceEntry]:
    by_source: dict[str, dict[str, Any]] = {}
    for content, metadata in rows:
        filename = source_filename(metadata)
        if not filename:
            continue
        page = int(metadata.get("page") or 0)
        current = by_source.setdefault(filename, {"front_page_chunks": []})
        if page == 0:
            current["front_page_chunks"].append(content)

    return [
        SourceEntry(
            filename=filename,
            aliases=tuple(
                sorted(_alias_candidates(filename, " ".join(payload["front_page_chunks"])[:4000]))
            ),
            front_page_text=" ".join(payload["front_page_chunks"])[:4000],
        )
        for filename, payload in sorted(by_source.items())
    ]


def _entries_from_manifest(manifest: dict[str, Any]) -> list[SourceEntry]:
    entries: list[SourceEntry] = []
    for item in manifest.get("source_files", []) or []:
        filename = str(item.get("filename") or os.path.basename(str(item.get("path", ""))))
        if not filename:
            continue
        entries.append(SourceEntry(filename=filename, aliases=tuple(sorted(_alias_candidates(filename)))))
    return entries
