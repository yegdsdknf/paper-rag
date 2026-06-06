"""兼容薄壳：新代码请从 paper_rag.ui.state 导入。"""

__compat_replacement__ = "paper_rag.ui.state"

from paper_rag.ui.state import clear_conversation_state

__all__ = ["clear_conversation_state"]
