from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReformulationResult:
    standalone_question: str
    rewritten: bool


def reformulate_question(
    conversation: Any,
    question: str,
    *,
    require_history: bool,
) -> ReformulationResult:
    """统一多轮改写决策，保留非流式和流式入口的历史判断差异。"""
    if require_history and not getattr(conversation, "history", []):
        return ReformulationResult(standalone_question=question, rewritten=False)

    standalone_question = conversation.reformulate(question)
    return ReformulationResult(
        standalone_question=standalone_question,
        rewritten=standalone_question != question,
    )
