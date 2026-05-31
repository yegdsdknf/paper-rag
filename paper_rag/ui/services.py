from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from feedback import build_feedback_record, save_feedback_record


@dataclass
class StreamAnswerResult:
    answer: str
    docs: list[Any]
    route: str
    rewrite: str


def save_uploaded_pdfs(files: list[Any], papers_dir: str | Path) -> list[Path]:
    """保存 Streamlit 上传文件，返回落盘路径；不负责触发入库。"""
    target_dir = Path(papers_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for file in files:
        path = target_dir / file.name
        with path.open("wb") as fh:
            fh.write(file.getbuffer())
        saved_paths.append(path)
    return saved_paths


def collect_stream_answer(
    ask_stream_fn: Callable[..., Any],
    hybrid: Any,
    conversation: Any,
    question: str,
    llm_model: str,
    temperature: float,
) -> StreamAnswerResult:
    answer = ""
    docs: list[Any] = []
    route = ""
    rewrite = ""

    for event in ask_stream_fn(hybrid, conversation, question, llm_model=llm_model, temperature=temperature):
        event_type = event["type"]
        if event_type == "rewrite":
            rewrite = event["data"]
        elif event_type == "route":
            route = event["data"]
        elif event_type == "docs":
            docs = event["data"]
        elif event_type == "token":
            answer += event["data"]

    return StreamAnswerResult(answer=answer, docs=docs, route=route, rewrite=rewrite)


def build_feedback_payload(
    question: str,
    answer: str,
    docs: list[Any],
    route: str,
    llm_model: str,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "sources": docs,
        "route": route,
        "llm_model": llm_model,
    }


def save_feedback_from_payload(
    payload: dict[str, Any],
    note: str,
    save_fn: Callable[..., Path] = save_feedback_record,
) -> Path:
    if not note.strip():
        raise ValueError("请先写一条备注，方便后续人工标注。")

    record = build_feedback_record(
        question=payload["question"],
        answer=payload["answer"],
        sources=payload["sources"],
        note=note,
        route=payload.get("route"),
        llm_model=payload.get("llm_model"),
    )
    return save_fn(record)

