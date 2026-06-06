"""兼容薄壳：新代码请从 paper_rag.generation.parent_retrieval 导入。"""

__compat_replacement__ = "paper_rag.generation.parent_retrieval"

from paper_rag.generation.parent_retrieval import expand_parent_pages

__all__ = ["expand_parent_pages"]
