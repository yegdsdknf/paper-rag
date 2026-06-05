"""兼容薄壳：新代码请从 paper_rag.ui.services 导入。"""

from paper_rag.ui.services import (
    StreamAnswerResult,
    build_feedback_payload,
    collect_stream_answer,
    save_feedback_from_payload,
    save_uploaded_pdfs,
)

__all__ = [
    "StreamAnswerResult",
    "build_feedback_payload",
    "collect_stream_answer",
    "save_feedback_from_payload",
    "save_uploaded_pdfs",
]
