from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping

from langchain_core.documents import Document

from paper_rag.config import get_setting
from utils.ollama_client import generate_vision_summary


VISION_PROMPT = """请分析这页论文图像，输出结构化结果：
1. 页面类型：figure/table/equation/mixed/unknown
2. 图表编号或标题：
3. 主要内容：
4. 关键指标/数值/趋势：
5. 可用于回答的问题：
6. 局限：无法辨认或不确定的内容

要求：
- 使用中文说明。
- 保留英文模型名、数据集名、指标名和数字。
- 不要编造看不清的内容。
- 如果内容主要是图表，请优先总结趋势、对比关系和关键数值。
"""


@dataclass(frozen=True)
class PageVisionSignal:
    source_file: str
    page: int
    visual_density: float = 0.0
    quality_flags: list[str] = field(default_factory=list)
    source_path: str = ""


@dataclass(frozen=True)
class SelectedVisionPage:
    source_file: str
    page: int
    trigger_reasons: list[str]
    visual_density: float
    quality_flags: list[str]
    source_path: str = ""


class VisionCache:
    def __init__(self, cache_dir: str | os.PathLike[str]):
        self.cache_dir = Path(cache_dir)
        self.summary_dir = self.cache_dir / "summaries"
        self.image_dir = self.cache_dir / "images"

    def _cache_path(self, source_hash: str, page: int, model: str, prompt_version: str) -> Path:
        model_key = _slug(model)
        prompt_key = _slug(prompt_version)
        return self.summary_dir / f"{source_hash}_p{int(page):04d}_{model_key}_{prompt_key}.json"

    def load(self, source_hash: str, page: int, model: str, prompt_version: str) -> dict[str, Any] | None:
        path = self._cache_path(source_hash, page, model, prompt_version)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["cache_file"] = str(path)
        return payload

    def save(
        self,
        source_hash: str,
        page: int,
        model: str,
        prompt_version: str,
        summary: str,
        image_path: str,
    ) -> Path:
        path = self._cache_path(source_hash, page, model, prompt_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "image_path": image_path,
            "model": model,
            "prompt_version": prompt_version,
            "page": int(page),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path


def select_vision_pages(signals: list[PageVisionSignal], settings: Any) -> list[SelectedVisionPage]:
    threshold = float(get_setting(settings, "vision_visual_density_threshold", 0.35))
    max_pages = int(get_setting(settings, "vision_max_pages_per_doc", 20))
    force_pages = dict(get_setting(settings, "vision_force_pages", {}) or {})
    policy = str(get_setting(settings, "vision_trigger_policy", "noisy_or_figure_page"))
    selected: list[SelectedVisionPage] = []

    by_source: dict[str, list[PageVisionSignal]] = {}
    for signal in signals:
        by_source.setdefault(signal.source_file.lower(), []).append(signal)

    for source_key, source_signals in by_source.items():
        candidates: list[SelectedVisionPage] = []
        for signal in sorted(source_signals, key=lambda item: item.page):
            reasons = _trigger_reasons(signal, force_pages, threshold, policy)
            if not reasons:
                continue
            candidates.append(
                SelectedVisionPage(
                    source_file=signal.source_file,
                    page=signal.page,
                    trigger_reasons=reasons,
                    visual_density=signal.visual_density,
                    quality_flags=signal.quality_flags,
                    source_path=signal.source_path,
                )
            )
        candidates.sort(key=_selection_priority)
        selected.extend(candidates[:max_pages])

    return selected


def collect_vision_summary_docs(
    pdf_paths: list[str],
    settings: Any,
    file_hashes: Mapping[str, str] | None,
    quality_reports: list[Mapping[str, Any]] | None = None,
    render_page: Callable[[str, int, str], str] | None = None,
    density_detector: Callable[[str, int], float] | None = None,
    summarizer: Callable[[str, str, str], str] | None = None,
) -> tuple[list[Document], dict[str, Any]]:
    stats = _empty_stats(bool(get_setting(settings, "enable_vision_analysis", False)))
    quality_reports = quality_reports or []
    stats["quality_flags"] = _quality_flag_counts(quality_reports)
    if not stats["enabled"]:
        return [], stats

    cache = VisionCache(str(get_setting(settings, "vision_cache_dir", "./data/vision_cache")))
    render_page = render_page or render_pdf_page
    density_detector = density_detector or detect_pdf_page_visual_density
    summarizer = summarizer or _default_summarizer

    signals = _collect_page_signals(pdf_paths, settings, quality_reports, density_detector)
    selected = select_vision_pages(signals, settings)
    stats["pages_selected"] = len(selected)

    docs: list[Document] = []
    for page in selected:
        source_path = page.source_path or _find_source_path(pdf_paths, page.source_file)
        source_hash = _source_hash(source_path, file_hashes)
        model = str(get_setting(settings, "vision_model", "qwen2.5vl:3b"))
        prompt_version = str(get_setting(settings, "vision_prompt_version", "v1"))
        cached = None
        if not bool(get_setting(settings, "vision_force_refresh", False)):
            cached = cache.load(source_hash, page.page, model, prompt_version)

        if cached:
            summary = str(cached["summary"])
            image_path = str(cached.get("image_path", ""))
            cache_hit = True
            stats["cache_hits"] += 1
        else:
            image_path = render_page(source_path, page.page, str(cache.image_dir))
            summary = summarizer(model, VISION_PROMPT, image_path)
            cache.save(source_hash, page.page, model, prompt_version, summary, image_path)
            cache_hit = False
            stats["generated"] += 1

        for reason in page.trigger_reasons:
            stats["trigger_reasons"][reason] = stats["trigger_reasons"].get(reason, 0) + 1

        docs.append(
            build_vision_summary_document(
                source_path=source_path,
                source_file_hash=source_hash,
                page=page.page,
                summary=summary,
                image_path=image_path,
                model=model,
                prompt_version=prompt_version,
                cache_hit=cache_hit,
                trigger_reasons=page.trigger_reasons,
                quality_flags=_vision_quality_flags(page.quality_flags),
            )
        )

    return docs, stats


def build_vision_summary_document(
    source_path: str,
    source_file_hash: str,
    page: int,
    summary: str,
    image_path: str,
    model: str,
    prompt_version: str,
    cache_hit: bool,
    trigger_reasons: list[str],
    quality_flags: list[str],
) -> Document:
    source_file = os.path.basename(str(source_path).replace("\\", "/"))
    content_hash = sha256(summary.encode("utf-8")).hexdigest()[:12]
    doc_id = f"doc_{source_file_hash}"
    metadata = {
        "source": source_path,
        "source_file": source_file,
        "page": int(page),
        "doc_id": doc_id,
        "block_type": "vision_summary",
        "chunk_strategy": "vision_summary",
        "chunk_schema_version": prompt_version,
        "vision_model": model,
        "vision_prompt_version": prompt_version,
        "image_path": image_path,
        "vision_cache_hit": bool(cache_hit),
        "vision_trigger_reason": list(trigger_reasons),
        "quality_flags": list(quality_flags),
        "content_hash": content_hash,
        "chunk_id": f"{doc_id}:vision_summary:{prompt_version}:{int(page)}:{content_hash}",
        "paper_region": "vision",
    }
    return Document(page_content=summary, metadata=metadata)


def render_pdf_page(pdf_path: str, page: int, image_dir: str) -> str:
    fitz = _import_fitz()
    output_dir = Path(image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_file = Path(pdf_path).stem
    image_path = output_dir / f"{_slug(source_file)}_p{int(page):04d}.png"
    with fitz.open(pdf_path) as pdf:
        pix = pdf.load_page(int(page)).get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pix.save(str(image_path))
    return str(image_path)


def detect_pdf_page_visual_density(pdf_path: str, page: int) -> float:
    fitz = _import_fitz()
    with fitz.open(pdf_path) as pdf:
        return _visual_density_from_page(pdf.load_page(int(page)))


def image_to_data_url(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    mime = "image/png" if suffix != ".jpg" and suffix != ".jpeg" else "image/jpeg"
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _default_summarizer(model: str, prompt: str, image_path: str) -> str:
    return generate_vision_summary(model=model, prompt=prompt, image_data_url=image_to_data_url(image_path))


def _collect_page_signals(
    pdf_paths: list[str],
    settings: Any,
    quality_reports: list[Mapping[str, Any]],
    density_detector: Callable[[str, int], float],
) -> list[PageVisionSignal]:
    signal_map: dict[tuple[str, int], PageVisionSignal] = {}
    force_pages = dict(get_setting(settings, "vision_force_pages", {}) or {})

    for pdf_path in pdf_paths:
        source_file = os.path.basename(str(pdf_path).replace("\\", "/"))
        pages = set(_matching_force_pages(source_file, force_pages))
        pages.update(_quality_pages(source_file, quality_reports))
        pages.update(_all_pages_if_available(pdf_path))
        for page in sorted(pages):
            quality_flags = _quality_flags_for(source_file, page, quality_reports)
            try:
                density = float(density_detector(pdf_path, page))
            except Exception:
                density = 0.0
            signal_map[(source_file.lower(), int(page))] = PageVisionSignal(
                source_file=source_file,
                page=int(page),
                visual_density=density,
                quality_flags=quality_flags,
                source_path=pdf_path,
            )

    return list(signal_map.values())


def _trigger_reasons(
    signal: PageVisionSignal,
    force_pages: Mapping[str, list[int]],
    threshold: float,
    policy: str = "noisy_or_figure_page",
) -> list[str]:
    reasons: list[str] = []
    forced = int(signal.page) in _matching_force_pages(signal.source_file, force_pages)
    if forced:
        reasons.append("forced_page")
    if policy == "forced_only":
        return reasons
    if "unicode_escape_noise" in signal.quality_flags:
        reasons.append("unicode_escape_noise")
    if policy == "noisy_or_forced":
        return reasons
    if signal.visual_density >= threshold:
        reasons.append("figure_dense_page")
    return reasons


def _selection_priority(page: SelectedVisionPage) -> tuple[int, int]:
    if "forced_page" in page.trigger_reasons:
        priority = 0
    elif "unicode_escape_noise" in page.trigger_reasons:
        priority = 1
    else:
        priority = 2
    return priority, page.page


def _matching_force_pages(source_file: str, force_pages: Mapping[str, list[int]]) -> list[int]:
    source_lower = source_file.lower()
    pages: list[int] = []
    for key, values in force_pages.items():
        key_lower = str(key).lower()
        if key_lower == source_lower or key_lower in source_lower:
            pages.extend(int(value) for value in values)
    return pages


def _quality_pages(source_file: str, quality_reports: list[Mapping[str, Any]]) -> list[int]:
    return [
        int(report.get("page", -1))
        for report in quality_reports
        if str(report.get("source", "")).lower() == source_file.lower()
    ]


def _quality_flag_counts(quality_reports: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for report in quality_reports:
        flags = report.get("quality_flags", [report.get("reason")])
        for flag in flags:
            if not flag:
                continue
            flag_key = str(flag)
            counts[flag_key] = counts.get(flag_key, 0) + 1
    return counts


def _quality_flags_for(
    source_file: str,
    page: int,
    quality_reports: list[Mapping[str, Any]],
) -> list[str]:
    flags: list[str] = []
    for report in quality_reports:
        if str(report.get("source", "")).lower() != source_file.lower():
            continue
        if int(report.get("page", -1)) != int(page):
            continue
        for flag in report.get("quality_flags", [report.get("reason")]):
            if flag and flag not in flags:
                flags.append(str(flag))
    return flags


def _vision_quality_flags(source_flags: list[str]) -> list[str]:
    flags = list(source_flags)
    if "vision_generated" not in flags:
        flags.append("vision_generated")
    return flags


def _all_pages_if_available(pdf_path: str) -> list[int]:
    path = Path(pdf_path)
    if not path.exists():
        return []
    try:
        fitz = _import_fitz()
        with fitz.open(str(path)) as pdf:
            return list(range(pdf.page_count))
    except Exception:
        return []


def _visual_density_from_page(page: Any) -> float:
    rect = page.rect
    page_area = max(float(rect.width * rect.height), 1.0)
    visual_area = 0.0
    for image in page.get_image_info(xrefs=True):
        bbox = image.get("bbox")
        if bbox:
            visual_area += _rect_area(bbox)
    for drawing in page.get_drawings():
        drawing_rect = drawing.get("rect")
        if drawing_rect:
            visual_area += max(float(drawing_rect.width * drawing_rect.height), 0.0)
    return min(1.0, visual_area / page_area)


def _rect_area(bbox: Any) -> float:
    if len(bbox) != 4:
        return 0.0
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def _find_source_path(pdf_paths: list[str], source_file: str) -> str:
    for path in pdf_paths:
        if os.path.basename(str(path).replace("\\", "/")).lower() == source_file.lower():
            return path
    return source_file


def _source_hash(source_path: str, file_hashes: Mapping[str, str] | None) -> str:
    hashes = file_hashes or {}
    path = Path(source_path)
    return (
        hashes.get(source_path)
        or hashes.get(str(path))
        or hashes.get(path.name)
        or sha256(str(source_path).encode("utf-8")).hexdigest()[:12]
    )


def _empty_stats(enabled: bool) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "pages_selected": 0,
        "generated": 0,
        "cache_hits": 0,
        "quality_flags": {},
        "trigger_reasons": {},
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()


def _import_fitz():
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF 未安装，请安装 pymupdf 后启用视觉入库。") from exc
    return fitz
