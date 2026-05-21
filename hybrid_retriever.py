"""
混合检索器：向量检索 + BM25 关键词检索 + 动态权重
"""
import re
import numpy as np
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from utils.config_loader import load_config

config = load_config()


class SemanticWeightDecider:
    """
    基于语义的查询意图分类 → 语言无关的动态权重
    原理：用 bge-m3 多语言向量空间做意图匹配。
    「BERT和ViT的区别」和「What is the difference between BERT and ViT」
    在向量空间中方向相同 → 分到同一个原型 → 同一个权重。
    """
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self._build_prototypes()

    def _build_prototypes(self):
        precise_anchors = [
            "BERT 和 ViT 的 Transformer 架构有什么区别",
            "F1 分数的计算公式",
            "LoRA 的秩 r 取多少合适",
            "GPT-4 的参数量是多少",
            "SQuAD 数据集的基准结果",
            "混合检索中 BM25 和向量检索的权重分配",
            "What is the difference between BERT and ViT Transformer architecture",
            "F1 score formula and calculation steps",
            "How many parameters does GPT-4 have",
            "bge-m3 vs text-embedding-3 benchmark comparison",
        ]
        semantic_anchors = [
            "什么是注意力机制",
            "为什么要做预训练",
            "如何理解知识蒸馏的思想",
            "Transformer 的核心创新是什么",
            "大语言模型的工作原理",
            "对比学习的直觉解释",
            "RAG 为什么能减少幻觉",
            "What is attention mechanism in deep learning",
            "Why do we need pretraining for language models",
            "What is the core innovation of Transformer architecture",
            "Explain the intuition behind contrastive learning",
            "Why does RAG reduce hallucinations in LLMs",
        ]
        precise_embs = np.array(self.embeddings.embed_documents(precise_anchors))
        semantic_embs = np.array(self.embeddings.embed_documents(semantic_anchors))
        self.prototype_precise = np.mean(precise_embs, axis=0)
        self.prototype_semantic = np.mean(semantic_embs, axis=0)

    def semantic_nudge(self, query: str) -> float:
        """
        计算语义倾向微调值（±0.12），叠加在正则基线上

        原理：bge-m3 原型空间虽近，但 sim 差值仍有细粒度信号。
        不做绝对权重决策，只输出方向+强度。

        Returns:
            float in [-0.12, +0.12]
            >0 → 偏精确，BM25 应增强
            <0 → 偏语义，向量应增强
        """
        q_vec = np.array(self.embeddings.embed_query(query))
        sim_precise = self._cosine(q_vec, self.prototype_precise)
        sim_semantic = self._cosine(q_vec, self.prototype_semantic)

        # sim 差值归一化到 [-1, 1]，缩放到 [-0.12, 0.12]
        diff = sim_precise - sim_semantic
        max_diff = max(abs(sim_precise), abs(sim_semantic), 1e-8)
        normalized = diff / max_diff
        return float(normalized * 0.12)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class HybridRetriever:
    """
    混合检索器

    融合方式：
    - 向量检索（语义匹配）
    - BM25 检索（关键词精确匹配）
    - 语义动态权重：基于 bge-m3 多语言意图分类（语言无关），降级为字符级正则
    """

    def __init__(
        self,
        vector_store: Chroma,
        top_k: int = config["k"],
        default_vector_weight: float = config["default_vector_weight"],
        default_bm25_weight: float = config["default_bm25_weight"],
        embedding_model = None
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.default_vector_weight = default_vector_weight
        self.default_bm25_weight = default_bm25_weight
        self.bm25_retriever = None      # 缓存
        self._bm25_doc_count = 0        # 过期校验（内部变量）
        if embedding_model is not None:
            self.weight_decider = SemanticWeightDecider(embedding_model)
        else:
            self.weight_decider = None

    def _get_chroma_doc_count(self) -> int:
        """安全获取 Chroma 文档数量（避免访问私有属性）"""
        try:
            # 优先用公开方法，降级到私有属性
            all_ids = self.vector_store.get(include=[])["ids"]
            return len(all_ids)
        except Exception:
            # 极端情况降级到私有属性
            try:
                return self.vector_store._collection.count()
            except Exception:
                return -1  # 强制触发 BM25 重建

    def build_bm25_retriever(self) -> BM25Retriever:
        """从向量库中提取所有文档，构建 BM25 检索器，加入 lazy_load 缓存"""
        current_count = self._get_chroma_doc_count()
        if self.bm25_retriever and current_count == self._bm25_doc_count:
            return self.bm25_retriever      # 缓存命中
        # 重建
        all_docs = self.vector_store.get(include=["documents", "metadatas"])
        documents = [
            Document(page_content=c, metadata=all_docs["metadatas"][i])
            for i, c in enumerate(all_docs["documents"])
        ]
        self.bm25_retriever = BM25Retriever.from_documents(documents=documents, k=self.top_k)
        self._bm25_doc_count = current_count
        return self.bm25_retriever

    @staticmethod
    def compute_dynamic_weights(query: str):
        """基于查询的统计特征自动判断，不需要关键词白名单（仅识别专有名词与缩写）"""

        # ── 信号检测 ──────────────────────
        # 信号 1：数学/公式符号（√、²、³、=、± 等） → 精确术语查询
        formula_chars = "√²³ᵀ⁻⁺→∑∏∫∂±="
        has_formula = any(c in query for c in formula_chars)

        # 信号 2：全大写缩写词（字符级正则，不依赖 split）
        # 匹配 2+ 连续大写字母，前后不能是字母（避免匹配长单词内部）
        acronyms = re.findall(r'(?<![a-zA-Z])[A-Z]{2,}(?![a-zA-Z])', query)
        has_acronym = len(acronyms) > 0

        # 信号 3：数字
        has_number = any(c.isdigit() for c in query)

        # 信号 4：首字母大写的专有名词（去掉前面已经匹配到的全大写缩写）
        # 匹配首字母大写、长度≥2 的驼峰/标题词
        proper_nouns = re.findall(r'(?<![a-zA-Z])[A-Z][a-zA-Z]+(?![a-zA-Z])', query)
        # 排除已经被信号2匹配到的全大写缩写
        proper_nouns = [p for p in proper_nouns if p.upper() != p]
        # 排除句首词
        if proper_nouns and query.strip().startswith(proper_nouns[0]):
            proper_nouns = proper_nouns[1:]
        proper_count = len(proper_nouns)

        # ── 综合评分（连续映射，替代硬分档）──
        precision_score = (
            (2 if has_formula else 0) +
            (1 if has_acronym else 0) +
            (1 if has_number else 0) +
            (proper_count * 0.5)
        )

        # 连续映射：score 0→bm25=0.3, score 4+→bm25=0.7
        bm25_weight = min(0.7, 0.3 + 0.1 * precision_score)
        vector_weight = 1.0 - bm25_weight
        return vector_weight, bm25_weight

    def _get_weights(self, query: str):
        """
        获取动态权重：正则信号（主） + 语义微调（辅）

        正则信号负责捕捉：公式符号、缩写词、数字、专有名词 → 粗粒度区分
        语义微调负责捕捉：查询在 precise/semantic 原型空间的倾向 → ±0.12 细调
        """
        vec_w, bm25_w = self.compute_dynamic_weights(query)

        if self.weight_decider is not None:
            nudge = self.weight_decider.semantic_nudge(query)
            bm25_w = max(0.20, min(0.80, bm25_w + nudge))
            vec_w = 1.0 - bm25_w

        return vec_w, bm25_w

    def get_retriever(self, query: str = None) -> BaseRetriever:
        """
        获取混合检索器

        Args:
            query: 用户查询（用于计算动态权重），为 None 时使用默认权重

        Returns:
            EnsembleRetriever（混合检索器）或纯向量检索器（降级方案）
        """
        # 1. 向量检索器
        vector_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        # 2. BM25 检索器
        bm25_retriever = self.build_bm25_retriever()

        # 3. 如果 BM25 构建失败，降级为纯向量检索
        if bm25_retriever is None:
            print("⚠️  BM25 构建失败，降级为纯向量检索")
            return vector_retriever

        # 4. 计算动态权重
        if query:
            weights = self._get_weights(query)
        else:
            weights = [self.default_vector_weight, self.default_bm25_weight]
        print(f"⚖️  混合检索权重：向量={weights[0]:.1f}, BM25={weights[1]:.1f}")

        # 5. 融合为 EnsembleRetriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=weights,
        )

        return ensemble_retriever
