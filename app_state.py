from __future__ import annotations

from collections.abc import MutableMapping


def clear_conversation_state(state: MutableMapping) -> None:
    """清空一次对话相关状态，保留模型选择和数据库版本等全局设置。"""
    state["messages"] = []
    state["conversation"] = None
    state["last_feedback_payload"] = None
    state.pop("feedback_note", None)
