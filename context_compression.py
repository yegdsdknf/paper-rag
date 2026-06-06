"""兼容薄壳：新代码请从 paper_rag.generation.context_compression 导入。"""

__compat_replacement__ = "paper_rag.generation.context_compression"

from paper_rag.generation.context_compression import compress_chunk, compress_documents

__all__ = ["compress_chunk", "compress_documents"]
