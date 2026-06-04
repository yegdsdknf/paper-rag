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


if __name__ == "__main__":
    unittest.main()
