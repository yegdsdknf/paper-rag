from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, StateGraph

from paper_rag.agentic.collector import collect_for_goal
from paper_rag.agentic.context import assemble_agentic_context
from paper_rag.agentic.planner import plan_evidence_goals
from paper_rag.agentic.schema import AgenticRagState
from paper_rag.agentic.verifier import verify_goal


def build_agentic_graph():
    graph = StateGraph(AgenticRagState)

    graph.add_node("plan", _plan_node)
    graph.add_node("collect", _collect_node)
    graph.add_node("verify", _verify_node)
    graph.add_node("repair", _repair_node)
    graph.add_node("assemble", _assemble_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "collect")
    graph.add_edge("collect", "verify")
    graph.add_conditional_edges("verify", _should_repair, {"repair": "repair", "assemble": "assemble"})
    graph.add_edge("repair", "collect")
    graph.add_edge("assemble", END)

    return graph.compile()


def run_agentic_rag(
    question: str,
    standalone_question: str,
    task_type: str,
    source_hints: list[str],
    hybrid: Any,
    router: Any,
    planner_llm: Any | None = None,
    verifier_llm: Any | None = None,
    llm_model: str = "",
    temperature: float = 0.0,
    max_repair_rounds: int = 1,
) -> AgenticRagState:
    state: AgenticRagState = {
        "question": question,
        "standalone_question": standalone_question,
        "task_type": task_type,
        "source_hints": list(source_hints),
        "repair_rounds": 0,
        "max_repair_rounds": max_repair_rounds,
        "llm_model": llm_model,
        "temperature": temperature,
        "_hybrid": hybrid,
        "_router": router,
        "_planner_llm": planner_llm,
        "_verifier_llm": verifier_llm,
        "_agent_start": time.perf_counter(),
    }
    return build_agentic_graph().invoke(state)


def _plan_node(state: AgenticRagState) -> AgenticRagState:
    goals = plan_evidence_goals(
        state.get("question", ""),
        state.get("standalone_question", ""),
        state.get("source_hints", []),
        state.get("task_type", ""),
        llm=state.get("_planner_llm"),
    )
    return {"goals": goals}


def _collect_node(state: AgenticRagState) -> AgenticRagState:
    collected_docs = []
    collected_docs_by_goal = {}
    routes: list[str] = []
    for index, goal in enumerate(state.get("goals", [])):
        docs, route = collect_for_goal(
            goal,
            hybrid=state.get("_hybrid"),
            router=state.get("_router"),
            llm_model=state.get("llm_model", ""),
            temperature=state.get("temperature", 0.0),
            repair_round=int(state.get("repair_rounds", 0) or 0),
        )
        goal_id = _goal_id(goal, index)
        collected_docs_by_goal[goal_id] = docs
        collected_docs.extend(docs)
        routes.append(route)

    return {
        "collected_docs": collected_docs,
        "collected_docs_by_goal": collected_docs_by_goal,
        "route": routes[0] if routes else "agentic_mixed",
    }


def _verify_node(state: AgenticRagState) -> AgenticRagState:
    verified_evidence = [
        verify_goal(
            goal,
            state.get("collected_docs_by_goal", {}).get(_goal_id(goal, index), []),
            llm=state.get("_verifier_llm"),
        )
        for index, goal in enumerate(state.get("goals", []))
    ]
    missing_goal_ids = [
        str(item.get("goal_id") or "")
        for item in verified_evidence
        if item.get("status") == "unsupported" and str(item.get("goal_id") or "")
    ]
    return {
        "verified_evidence": verified_evidence,
        "missing_goal_ids": missing_goal_ids,
    }


def _should_repair(state: AgenticRagState) -> str:
    if state.get("missing_goal_ids") and state.get("repair_rounds", 0) < state.get("max_repair_rounds", 1):
        return "repair"
    return "assemble"


def _repair_node(state: AgenticRagState) -> AgenticRagState:
    return {"repair_rounds": state.get("repair_rounds", 0) + 1}


def _goal_id(goal: dict[str, Any], index: int) -> str:
    return str(goal.get("id") or f"g{index + 1}").strip()


def _assemble_node(state: AgenticRagState) -> AgenticRagState:
    result = assemble_agentic_context(
        state.get("collected_docs", []),
        state.get("verified_evidence", []),
        state.get("task_type", ""),
    )
    missing_goal_ids = state.get("missing_goal_ids", [])
    agent_start = state.get("_agent_start")
    elapsed = time.perf_counter() - agent_start if isinstance(agent_start, (int, float)) else 0.0
    return {
        "final_docs": result.final_docs,
        "verified_summary": result.verified_summary,
        "agent_trace": {
            "enabled": True,
            "plan": state.get("goals", []),
            "verification": state.get("verified_evidence", []),
            "repair_rounds": state.get("repair_rounds", 0),
            "repair_success": not missing_goal_ids,
            "agent_elapsed_sec": round(elapsed, 4),
        },
    }
