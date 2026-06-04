import tempfile
import unittest
from pathlib import Path

import numpy as np


class FakeEmbeddings:
    model_name = "fake/bge"

    def __init__(self):
        self.embed_documents_calls = 0

    def embed_documents(self, texts):
        self.embed_documents_calls += 1
        vectors = []
        for index, _text in enumerate(texts):
            vectors.append([float(index + 1), float(len(texts))])
        return vectors

    def embed_query(self, text):
        return [1.0, 1.0]


class SemanticPrototypeCacheTest(unittest.TestCase):
    def test_semantic_weight_decider_reuses_persisted_prototypes(self):
        from hybrid_retriever import SemanticWeightDecider

        with tempfile.TemporaryDirectory() as tmp:
            first_embeddings = FakeEmbeddings()
            first = SemanticWeightDecider(first_embeddings, prototype_cache_dir=tmp)

            second_embeddings = FakeEmbeddings()
            second = SemanticWeightDecider(second_embeddings, prototype_cache_dir=tmp)

            self.assertEqual(first_embeddings.embed_documents_calls, 2)
            self.assertEqual(second_embeddings.embed_documents_calls, 0)
            self.assertTrue(np.array_equal(first.prototype_precise, second.prototype_precise))
            self.assertTrue(np.array_equal(first.prototype_semantic, second.prototype_semantic))
            self.assertEqual(len(list(Path(tmp).glob("*.npz"))), 1)

    def test_corrupt_prototype_cache_is_ignored_and_overwritten(self):
        from hybrid_retriever import SemanticWeightDecider
        from paper_rag.retrieval.prototype_cache import prototype_cache_path

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = prototype_cache_path(tmp, "fake/bge")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(b"not an npz")

            embeddings = FakeEmbeddings()
            decider = SemanticWeightDecider(embeddings, prototype_cache_dir=tmp)

            self.assertEqual(embeddings.embed_documents_calls, 2)
            self.assertTrue(cache_path.read_bytes().startswith(b"PK"))
            self.assertEqual(decider.prototype_precise.shape, decider.prototype_semantic.shape)


if __name__ == "__main__":
    unittest.main()
