import unittest

from langchain_core.documents import Document


class FakeRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.invocations = []

    def invoke(self, query):
        self.invocations.append(query)
        return list(self.docs)


class FakeHybrid:
    def __init__(self, mapping):
        self.mapping = mapping
        self.queries = []

    def get_retriever(self, query):
        self.queries.append(query)
        return FakeRetriever(self.mapping[query])


class FakeLLM:
    def __init__(self, response=None, fail=False):
        self.response = response
        self.fail = fail
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("offline")
        return type("Response", (), {"content": self.response})()


class PipelineRetrievalTest(unittest.TestCase):
    def test_retrieve_documents_invokes_hybrid_and_deduplicates(self):
        from paper_rag.pipeline.retrieval import retrieve_documents

        docs = [
            Document(page_content="first", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="duplicate", metadata={"source": "a.pdf", "page": 1}),
            Document(page_content="second", metadata={"source": "b.pdf", "page": 2}),
        ]
        hybrid = FakeHybrid({"query": docs})

        results = retrieve_documents(hybrid, "query")

        self.assertEqual(hybrid.queries, ["query"])
        self.assertEqual([doc.metadata["source"] for doc in results], ["a.pdf", "b.pdf"])

    def test_retrieve_with_hyde_uses_generated_document_for_retrieval(self):
        from paper_rag.pipeline.retrieval import retrieve_documents, retrieve_with_hyde

        hyde_doc = "hypothetical evidence"
        result_doc = Document(page_content="retrieved", metadata={"source": "paper.pdf", "page": 3})
        hybrid = FakeHybrid({hyde_doc: [result_doc]})
        llm = FakeLLM(response=hyde_doc)

        docs = retrieve_with_hyde(
            hybrid,
            "question",
            llm_model="qwen2.5:3b",
            temperature=0.0,
            load_prompt_fn=lambda name: "hyde prompt: {query}",
            get_llm_fn=lambda model, temperature: llm,
            retrieve_fn=retrieve_documents,
        )

        self.assertEqual(docs, [result_doc])
        self.assertEqual(hybrid.queries, [hyde_doc])
        self.assertEqual(llm.prompts, ["hyde prompt: question"])

    def test_retrieve_with_hyde_falls_back_when_llm_unavailable_or_fails(self):
        from paper_rag.pipeline.retrieval import retrieve_documents, retrieve_with_hyde

        fallback_doc = Document(page_content="fallback", metadata={"source": "paper.pdf", "page": 1})
        hybrid = FakeHybrid({"question": [fallback_doc]})

        docs_without_llm = retrieve_with_hyde(
            hybrid,
            "question",
            llm_model="qwen2.5:3b",
            temperature=0.0,
            load_prompt_fn=lambda name: "hyde prompt: {query}",
            get_llm_fn=lambda model, temperature: None,
            retrieve_fn=retrieve_documents,
        )
        docs_after_failure = retrieve_with_hyde(
            hybrid,
            "question",
            llm_model="qwen2.5:3b",
            temperature=0.0,
            load_prompt_fn=lambda name: "hyde prompt: {query}",
            get_llm_fn=lambda model, temperature: FakeLLM(fail=True),
            retrieve_fn=retrieve_documents,
        )

        self.assertEqual(docs_without_llm, [fallback_doc])
        self.assertEqual(docs_after_failure, [fallback_doc])
        self.assertEqual(hybrid.queries, ["question", "question"])


if __name__ == "__main__":
    unittest.main()
