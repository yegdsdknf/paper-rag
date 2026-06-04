import unittest


class FakeCuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeTorch:
    def __init__(self, available):
        self.cuda = FakeCuda(available)


class FakeLLM:
    def __init__(self, fail=False):
        self.fail = fail
        self.invocations = []

    def invoke(self, prompt):
        self.invocations.append(prompt)
        if self.fail:
            raise RuntimeError("offline")
        return "pong"


class FakeSettings:
    embedding_model = "fake/bge"
    persist_directory = "./fake-db"
    collection_name = "fake_collection"
    k = 7
    default_vector_weight = 0.65
    default_bm25_weight = 0.35
    chunk_schema_version = "v9"
    index_manifest_filename = "manifest.json"


class FakeEmbeddings:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeEmbeddings.instances.append(self)


class FakeVectorStore:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeVectorStore.instances.append(self)


class FakeHybridRetriever:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeHybridRetriever.instances.append(self)


class RuntimeModelsTest(unittest.TestCase):
    def test_select_embedding_device_prefers_cuda_when_available(self):
        from paper_rag.runtime.models import select_embedding_device

        self.assertEqual(select_embedding_device(FakeTorch(True)), "cuda")
        self.assertEqual(select_embedding_device(FakeTorch(False)), "cpu")
        self.assertEqual(select_embedding_device(None), "cpu")

    def test_get_cached_llm_reuses_successful_connection_by_generation_settings(self):
        from paper_rag.runtime.models import get_cached_llm

        cache = {}
        created = []

        def create_llm(**kwargs):
            llm = FakeLLM()
            created.append((kwargs, llm))
            return llm

        first = get_cached_llm(
            cache,
            create_llm,
            model="qwen2.5:3b",
            temperature=0.1,
            num_ctx=2048,
            num_predict=512,
        )
        second = get_cached_llm(
            cache,
            create_llm,
            model="qwen2.5:3b",
            temperature=0.1,
            num_ctx=2048,
            num_predict=512,
        )

        self.assertIs(first, second)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0][1].invocations, ["ping"])
        self.assertEqual(
            created[0][0],
            {"model": "qwen2.5:3b", "temperature": 0.1, "num_ctx": 2048, "num_predict": 512},
        )

    def test_get_cached_llm_caches_none_when_ping_fails(self):
        from paper_rag.runtime.models import get_cached_llm

        cache = {}
        errors = []
        created = []

        def create_llm(**_kwargs):
            llm = FakeLLM(fail=True)
            created.append(llm)
            return llm

        first = get_cached_llm(
            cache,
            create_llm,
            model="missing",
            temperature=0.0,
            num_ctx=1024,
            num_predict=128,
            on_error=lambda exc: errors.append(str(exc)),
        )
        second = get_cached_llm(
            cache,
            create_llm,
            model="missing",
            temperature=0.0,
            num_ctx=1024,
            num_predict=128,
        )

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(created), 1)
        self.assertEqual(errors, ["offline"])

    def test_build_hybrid_retriever_wires_embeddings_vector_store_and_hybrid(self):
        from paper_rag.runtime.models import build_hybrid_retriever

        FakeEmbeddings.instances = []
        FakeVectorStore.instances = []
        FakeHybridRetriever.instances = []

        hybrid = build_hybrid_retriever(
            FakeSettings(),
            torch_module=FakeTorch(True),
            embeddings_cls=FakeEmbeddings,
            vector_store_cls=FakeVectorStore,
            hybrid_retriever_cls=FakeHybridRetriever,
        )

        embeddings = FakeEmbeddings.instances[0]
        vector_store = FakeVectorStore.instances[0]
        self.assertIs(hybrid, FakeHybridRetriever.instances[0])
        self.assertEqual(
            embeddings.kwargs,
            {
                "model_name": "fake/bge",
                "model_kwargs": {"device": "cuda", "local_files_only": True},
                "encode_kwargs": {"normalize_embeddings": True},
            },
        )
        self.assertEqual(
            vector_store.kwargs,
            {
                "persist_directory": "./fake-db",
                "embedding_function": embeddings,
                "collection_name": "fake_collection",
            },
        )
        self.assertEqual(
            hybrid.kwargs,
            {
                "vector_store": vector_store,
                "top_k": 7,
                "default_vector_weight": 0.65,
                "default_bm25_weight": 0.35,
                "embedding_model": embeddings,
                "persist_directory": "./fake-db",
                "collection_name": "fake_collection",
                "chunk_schema_version": "v9",
                "index_manifest_filename": "manifest.json",
            },
        )


if __name__ == "__main__":
    unittest.main()
