"""兼容薄壳：新代码请从 paper_rag.observability.feedback 导入。"""

__compat_replacement__ = "paper_rag.observability.feedback"

from paper_rag.observability.feedback import (
    DEFAULT_FEEDBACK_PATH,
    build_feedback_record,
    save_feedback_record,
)

__all__ = [
    "DEFAULT_FEEDBACK_PATH",
    "build_feedback_record",
    "save_feedback_record",
]
