from paper_rag.observability.sources import source_from_doc, sources_from_docs

__all__ = [
    "build_feedback_record",
    "build_query_log_record",
    "save_feedback_record",
    "save_query_log_record",
    "source_from_doc",
    "sources_from_docs",
    "TraceTimer",
    "write_query_log",
]


def __getattr__(name):
    if name in {"build_feedback_record", "save_feedback_record"}:
        from paper_rag.observability.feedback import build_feedback_record, save_feedback_record

        return {
            "build_feedback_record": build_feedback_record,
            "save_feedback_record": save_feedback_record,
        }[name]
    if name in {"build_query_log_record", "save_query_log_record"}:
        from paper_rag.observability.query_logger import (
            build_query_log_record,
            save_query_log_record,
        )

        return {
            "build_query_log_record": build_query_log_record,
            "save_query_log_record": save_query_log_record,
        }[name]
    if name == "write_query_log":
        from paper_rag.observability.service import write_query_log

        return write_query_log
    if name == "TraceTimer":
        from paper_rag.observability.trace import TraceTimer

        return TraceTimer
    raise AttributeError(name)
