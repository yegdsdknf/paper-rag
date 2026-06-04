from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.error import URLError
from urllib.request import urlopen

from paper_rag.config.settings import RagSettings
from utils.config_loader import load_config


DiagnosticStatus = Literal["OK", "WARN", "ERROR"]


@dataclass(frozen=True)
class DiagnosticCheck:
    id: str
    title: str
    status: DiagnosticStatus
    message: str
    suggestion: str = ""
    elapsed_sec: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticReport:
    status: DiagnosticStatus
    exit_code: int
    summary: dict[str, int]
    checks: list[DiagnosticCheck]
    settings: dict[str, Any]


def _elapsed(start: float) -> float:
    return round(time.perf_counter() - start, 4)


def _timed_check(
    check_id: str,
    title: str,
    fn: Callable[[], tuple[DiagnosticStatus, str, str, dict[str, Any] | None]],
) -> DiagnosticCheck:
    start = time.perf_counter()
    status, message, suggestion, details = fn()
    return DiagnosticCheck(
        check_id,
        title,
        status,
        message,
        suggestion=suggestion,
        elapsed_sec=_elapsed(start),
        details=details or {},
    )


def _settings_snapshot(settings: RagSettings) -> dict[str, Any]:
    return {
        "persist_directory": settings.persist_directory,
        "collection_name": settings.collection_name,
        "chunk_schema_version": settings.chunk_schema_version,
        "embedding_model": settings.embedding_model,
        "llm_model": settings.llm_model,
        "enable_rerank": settings.enable_rerank,
        "reranker_model": settings.reranker_model,
        "enable_query_expansion": settings.enable_query_expansion,
        "enable_context_compression": settings.enable_context_compression,
        "enable_parent_retrieval": settings.enable_parent_retrieval,
    }


def build_report(checks: list[DiagnosticCheck], settings_snapshot: dict[str, Any]) -> DiagnosticReport:
    summary: dict[str, int] = {}
    for check in checks:
        summary[check.status] = summary.get(check.status, 0) + 1

    if summary.get("ERROR", 0):
        status: DiagnosticStatus = "ERROR"
        exit_code = 1
    elif summary.get("WARN", 0):
        status = "WARN"
        exit_code = 0
    else:
        status = "OK"
        exit_code = 0

    return DiagnosticReport(
        status=status,
        exit_code=exit_code,
        summary=summary,
        checks=checks,
        settings=settings_snapshot,
    )


def report_to_dict(report: DiagnosticReport) -> dict[str, Any]:
    return {
        "status": report.status,
        "exit_code": report.exit_code,
        "summary": report.summary,
        "settings": report.settings,
        "checks": [
            {
                "id": check.id,
                "title": check.title,
                "status": check.status,
                "message": check.message,
                "suggestion": check.suggestion,
                "elapsed_sec": check.elapsed_sec,
                "details": check.details,
            }
            for check in report.checks
        ],
    }


def render_json(report: DiagnosticReport) -> str:
    return json.dumps(report_to_dict(report), ensure_ascii=False, indent=2)


def _summary_text(summary: dict[str, int]) -> str:
    parts = []
    for status in ("OK", "WARN", "ERROR"):
        count = summary.get(status, 0)
        if count:
            parts.append(f"{count} {status}")
    return ", ".join(parts) if parts else "0 checks"


def render_text(report: DiagnosticReport) -> str:
    lines = [
        f"Paper-RAG Doctor: {report.status}",
        f"Summary: {_summary_text(report.summary)}",
    ]

    if report.settings:
        lines.append("")
        lines.append("Settings:")
        for key in ("persist_directory", "collection_name", "embedding_model", "llm_model", "enable_rerank"):
            if key in report.settings:
                lines.append(f"- {key}: {report.settings[key]}")

    errors = [check for check in report.checks if check.status == "ERROR"]
    warnings = [check for check in report.checks if check.status == "WARN"]

    if errors:
        lines.append("")
        lines.append("Blocking issues:")
        for check in errors:
            lines.append(f"- {check.title}: {check.message}")
            if check.suggestion:
                lines.append(f"  Suggestion: {check.suggestion}")

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for check in warnings:
            lines.append(f"- {check.title}: {check.message}")
            if check.suggestion:
                lines.append(f"  Suggestion: {check.suggestion}")

    lines.append("")
    lines.append("Checks:")
    for check in report.checks:
        lines.append(f"[{check.status}] {check.title} ({check.elapsed_sec:.2f}s) - {check.message}")
    return "\n".join(lines)


def _default_path_exists(path: str | Path) -> bool:
    return Path(path).exists()


def _default_manifest_loader(settings: RagSettings) -> dict[str, Any] | None:
    path = Path(settings.persist_directory) / settings.index_manifest_filename
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _default_chroma_count_loader(settings: RagSettings) -> int:
    import chromadb

    client = chromadb.PersistentClient(path=settings.persist_directory)
    collection = client.get_collection(settings.collection_name)
    return int(collection.count())


def _default_ollama_tags_loader(_settings: RagSettings) -> list[str]:
    with urlopen("http://localhost:11434/api/tags", timeout=2) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [str(model.get("name", "")) for model in payload.get("models", [])]


def _embedding_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _default_embedding_checker(settings: RagSettings) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

    from langchain_community.embeddings import HuggingFaceBgeEmbeddings

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": _embedding_device(), "local_files_only": True},
        encode_kwargs={"normalize_embeddings": True},
    )
    embeddings.embed_query("doctor check")


def _default_reranker_locator(settings: RagSettings) -> bool:
    if not settings.reranker_model:
        return False
    path = Path(settings.reranker_model)
    if path.exists():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(settings.reranker_model, local_files_only=True)
        return True
    except Exception:
        return False


def _default_papers_pdf_counter(project_root: Path) -> int:
    papers_dir = project_root / "papers"
    if not papers_dir.exists():
        return 0
    return sum(1 for _ in papers_dir.glob("*.pdf"))


def run_diagnostics(
    *,
    config_loader: Callable[[], dict[str, Any]] = load_config,
    path_exists: Callable[[str | Path], bool] = _default_path_exists,
    manifest_loader: Callable[[RagSettings], dict[str, Any] | None] = _default_manifest_loader,
    chroma_count_loader: Callable[[RagSettings], int] = _default_chroma_count_loader,
    ollama_tags_loader: Callable[[RagSettings], list[str]] = _default_ollama_tags_loader,
    embedding_checker: Callable[[RagSettings], None] = _default_embedding_checker,
    reranker_locator: Callable[[RagSettings], bool] = _default_reranker_locator,
    papers_pdf_counter: Callable[[Path], int] = _default_papers_pdf_counter,
    project_root: Path | None = None,
    slow_threshold_sec: float = 10.0,
) -> DiagnosticReport:
    total_start = time.perf_counter()
    project_root = project_root or Path.cwd()
    checks: list[DiagnosticCheck] = []

    start = time.perf_counter()
    try:
        settings = RagSettings.from_mapping(config_loader())
    except Exception as exc:
        checks.append(
            DiagnosticCheck(
                "config.load",
                "配置文件",
                "ERROR",
                f"配置加载失败: {exc}",
                suggestion="检查 config.yaml 是否存在，并确认必要字段完整且类型正确。",
                elapsed_sec=_elapsed(start),
            )
        )
        return build_report(checks, {})

    checks.append(
        DiagnosticCheck(
            "config.load",
            "配置文件",
            "OK",
            "配置已加载",
            elapsed_sec=_elapsed(start),
        )
    )

    persist_ok = path_exists(settings.persist_directory)
    checks.append(
        DiagnosticCheck(
            "path.persist_directory",
            "向量库目录",
            "OK" if persist_ok else "ERROR",
            f"向量库目录存在: {settings.persist_directory}" if persist_ok else f"向量库目录不存在: {settings.persist_directory}",
            suggestion="" if persist_ok else "先运行 python main.py build 构建向量库。",
        )
    )

    if persist_ok:
        def chroma_check() -> tuple[DiagnosticStatus, str, str, dict[str, Any] | None]:
            try:
                count = chroma_count_loader(settings)
            except Exception as exc:
                return "ERROR", f"Chroma collection 不可读: {exc}", "确认 collection_name 是否正确，或重新运行 python main.py build。", {}
            if count <= 0:
                return "ERROR", "Chroma collection 为空", "重新运行 python main.py build 构建知识库。", {"count": count}
            return "OK", f"Chroma collection 可读，文档数 {count}", "", {"count": count}

        checks.append(_timed_check("chroma.collection", "Chroma Collection", chroma_check))
    else:
        checks.append(
            DiagnosticCheck(
                "chroma.collection",
                "Chroma Collection",
                "WARN",
                "因向量库目录不存在，跳过 Chroma collection 检查。",
                suggestion="先运行 python main.py build 构建向量库。",
            )
        )

    def manifest_check() -> tuple[DiagnosticStatus, str, str, dict[str, Any] | None]:
        manifest = manifest_loader(settings)
        if manifest is None:
            return "WARN", "未找到 index_manifest.json", "重新运行 python main.py build 以生成索引元信息。", {}
        manifest_schema = str(manifest.get("chunk_schema_version", ""))
        if manifest_schema != settings.chunk_schema_version:
            return (
                "WARN",
                f"manifest schema={manifest_schema} 与当前配置 schema={settings.chunk_schema_version} 不一致",
                "确认 config.yaml 是否切换了分块策略；必要时重新运行 python main.py build。",
                {"manifest_schema": manifest_schema},
            )
        return "OK", "索引 manifest 与当前 chunk schema 一致", "", {"chunk_schema_version": manifest_schema}

    checks.append(_timed_check("index.manifest", "索引 Manifest", manifest_check))

    ollama_models: list[str] = []

    def ollama_service_check() -> tuple[DiagnosticStatus, str, str, dict[str, Any] | None]:
        nonlocal ollama_models
        try:
            ollama_models = ollama_tags_loader(settings)
        except (URLError, TimeoutError, OSError, Exception) as exc:
            return "ERROR", f"Ollama 服务不可用: {exc}", "启动 Ollama 服务，例如运行 ollama serve。", {}
        return "OK", f"Ollama 服务可用，发现 {len(ollama_models)} 个模型", "", {"model_count": len(ollama_models)}

    service_check = _timed_check("ollama.service", "Ollama 服务", ollama_service_check)
    checks.append(service_check)
    if service_check.status == "OK":
        model_found = settings.llm_model in ollama_models
        checks.append(
            DiagnosticCheck(
                "ollama.model",
                "Ollama 模型",
                "OK" if model_found else "ERROR",
                f"已找到 LLM 模型: {settings.llm_model}" if model_found else f"未找到 LLM 模型: {settings.llm_model}",
                suggestion="" if model_found else f"运行 ollama pull {settings.llm_model} 下载模型。",
                details={"available_models": ollama_models},
            )
        )
    else:
        checks.append(
            DiagnosticCheck(
                "ollama.model",
                "Ollama 模型",
                "WARN",
                "因 Ollama 服务不可用，跳过模型列表检查。",
                suggestion="先启动 Ollama 服务，再重新运行 python main.py doctor。",
            )
        )

    def embedding_check() -> tuple[DiagnosticStatus, str, str, dict[str, Any] | None]:
        try:
            embedding_checker(settings)
        except Exception as exc:
            return "ERROR", f"Embedding 模型不可用: {exc}", "确认本地 embedding 模型已下载，或将 config.yaml 中 embedding_model 改为可用本地路径。", {}
        return "OK", f"Embedding 模型可用: {settings.embedding_model}", "", {"embedding_model": settings.embedding_model}

    checks.append(_timed_check("embedding.model", "Embedding 模型", embedding_check))

    if not settings.enable_rerank:
        checks.append(
            DiagnosticCheck(
                "reranker.model",
                "Reranker 模型",
                "OK",
                "Rerank 未启用，跳过模型检查。",
            )
        )
    else:
        def reranker_check() -> tuple[DiagnosticStatus, str, str, dict[str, Any] | None]:
            if reranker_locator(settings):
                return "OK", f"Reranker 模型可定位: {settings.reranker_model}", "", {}
            return "WARN", f"Reranker 模型不可定位: {settings.reranker_model}", "确认 reranker_model 路径，或关闭 enable_rerank。", {}

        checks.append(_timed_check("reranker.model", "Reranker 模型", reranker_check))

    def papers_check() -> tuple[DiagnosticStatus, str, str, dict[str, Any] | None]:
        count = papers_pdf_counter(project_root)
        if count <= 0:
            return "WARN", "papers/ 目录为空或没有 PDF", "将论文 PDF 放入 papers/ 后运行 python main.py build。", {"pdf_count": count}
        return "OK", f"papers/ 中发现 {count} 个 PDF", "", {"pdf_count": count}

    checks.append(_timed_check("path.papers", "论文目录", papers_check))
    total_elapsed = _elapsed(total_start)
    if total_elapsed > slow_threshold_sec:
        checks.append(
            DiagnosticCheck(
                "diagnostics.elapsed",
                "诊断耗时",
                "WARN",
                f"诊断总耗时 {total_elapsed:.2f}s，超过目标 {slow_threshold_sec:.2f}s。",
                suggestion="查看各检查项 elapsed_sec，优先定位耗时最高的模型或服务检查。",
                elapsed_sec=total_elapsed,
                details={"total_elapsed_sec": total_elapsed, "slow_threshold_sec": slow_threshold_sec},
            )
        )
    return build_report(checks, _settings_snapshot(settings))


def run_doctor_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Paper-RAG startup diagnostics.")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON to stdout.")
    args = parser.parse_args(argv)

    if args.json:
        with contextlib.redirect_stdout(io.StringIO()):
            report = run_diagnostics()
        print(render_json(report))
    else:
        report = run_diagnostics()
        print(render_text(report))
    return report.exit_code
