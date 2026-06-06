"""兼容薄壳：新代码请从 paper_rag.observability.query_logger 导入。"""

__compat_replacement__ = "paper_rag.observability.query_logger"

from paper_rag.observability.query_logger import (
    DEFAULT_QUERY_LOG_PATH,
    build_query_log_record,
    save_query_log_record,
)

__all__ = [
    "DEFAULT_QUERY_LOG_PATH",
    "build_query_log_record",
    "save_query_log_record",
]
