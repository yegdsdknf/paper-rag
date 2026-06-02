from __future__ import annotations

import contextlib
import io
from typing import Any

from paper_rag.agentic.json_utils import parse_json_object
from paper_rag.agentic.schema import EvidenceGoal, normalize_goal
from utils.prompt_loader import load_prompt


_MAX_GOALS = 4


def _rule_goals(standalone_question: str, source_hints: list[str], task_type: str) -> list[dict[str, Any]]:
    if task_type == "compare" and source_hints:
        return [
            {
                "goal_type": "compare_dimension",
                "claim": f"比较 {source} 中与问题相关的证据",
                "query": f"{standalone_question} {source}",
                "source_hint": source,
                "page_hint": None,
            }
            for source in source_hints[:_MAX_GOALS]
        ]

    if task_type == "figure":
        goal_type = "figure_evidence"
    elif task_type == "evidence":
        goal_type = "page_evidence"
    else:
        goal_type = "method_overview"

    return [
        {
            "goal_type": goal_type,
            "claim": standalone_question,
            "query": standalone_question,
            "source_hint": source_hints[0] if source_hints else "",
            "page_hint": None,
        }
    ]


def _response_text(response: Any) -> str:
    return str(response.content if hasattr(response, "content") else response)


def _load_planner_prompt() -> str:
    with contextlib.redirect_stdout(io.StringIO()):
        return load_prompt("agent_planner_prompt")


def plan_evidence_goals(
    question: str,
    standalone_question: str,
    source_hints: list[str],
    task_type: str,
    llm: Any | None = None,
) -> list[EvidenceGoal]:
    raw_goals = _rule_goals(standalone_question, source_hints, task_type)

    if llm is not None:
        try:
            prompt_template = _load_planner_prompt()
            if not prompt_template:
                raise ValueError("agent planner prompt is empty")

            prompt = prompt_template.format(
                question=question,
                standalone_question=standalone_question,
                task_type=task_type,
                source_hints=source_hints,
            )
            parsed = parse_json_object(_response_text(llm.invoke(prompt)))
            llm_goals = parsed.get("goals")
            if isinstance(llm_goals, list) and llm_goals:
                raw_goals = [goal for goal in llm_goals if isinstance(goal, dict)]
                if not raw_goals:
                    raw_goals = _rule_goals(standalone_question, source_hints, task_type)
        except Exception:
            raw_goals = _rule_goals(standalone_question, source_hints, task_type)

    allowed_sources = set(source_hints)
    return [
        normalize_goal(goal, index, allowed_sources=allowed_sources)
        for index, goal in enumerate(raw_goals[:_MAX_GOALS])
    ]
