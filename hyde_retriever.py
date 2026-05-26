"""
HyDE 检索器：先让 LLM 生成假设性文档，再用它去检索

参考：LangChain-RAG-FastAPI-Service 的 rag_service.py

⚠️ 此模块当前未被任何文件引用，HyDE 的实际实现在 rag_pipeline.py 的 _retrieve_with_hyde() 中。
保留此文件作为未来迁移到 async 方案的参考。
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from utils.ollama_client import create_chat_ollama

from utils.config_loader import load_config
from utils.prompt_loader import load_prompt

config = load_config()
DEFAULT_HYDE_MODEL = config.get("llm_model_reasoning", "deepseek-r1:7b")


class HydeRetriever:
    """
    HyDE 检索器

    流程：
        1. 用户问题 → LLM → 生成一段假设性回答
        2. 用假设性回答代替原始问题，去向量库检索
        3. 返回检索结果
    """
    def __init__(self, base_retriever, llm_model: str = DEFAULT_HYDE_MODEL, temperature: float = 0.1):
        self.base_retriever = base_retriever
        self.llm = create_chat_ollama(model=llm_model, temperature=temperature)
        # 从文件加载 HyDE prompt 模板
        hyde_prompt_text = load_prompt("hyde_prompt")
        self.hyde_prompt = PromptTemplate(
            template=hyde_prompt_text,
            input_variables=["query"]
        )

        self.hyde_chain = self.hyde_prompt | self.llm | StrOutputParser()

    async def generate_hypothetical_document(self, query: str) -> str:
        """
            使用 HyDE 技术生成假设性文档

            Args:
                query: 用户原始问题

            Returns:
                假设性文档文本
        """
        print(f"🔄 HyDE：正在根据问题生成假设性文档...")
        hypothetical_doc = await self.hyde_chain.ainvoke({"query": query})
        print(f"✅ HyDE：假设性文档生成完成（{len(hypothetical_doc)} 字符）")

        return hypothetical_doc

    async def retriever(self, query: str) -> list:
        """
            使用 HyDE 技术检索文档

            Args:
                query: 用户原始问题

            Returns:
                检索到的 Document 列表
        """
        # Step 1: 生成假设性文档
        hypothetical_doc = await self.generate_hypothetical_document(query)
        # Step 2: 用假设性文档替代原始查询进行检索
        print(f"🔍 HyDE：使用假设性文档进行检索...")
        documents = await self.base_retriever.ainvoke(hypothetical_doc)

        print(f"📄 HyDE：检索到 {len(documents)} 个相关文档")
        for i, doc in enumerate(documents):
            print(f"  [{i + 1}] {doc.page_content[:80]}...")

        return documents
