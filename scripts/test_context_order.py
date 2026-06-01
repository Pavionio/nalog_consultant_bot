import unittest

from src.rag.core import RAGConfig, build_context, reorder_context_chunks


class ContextOrderTests(unittest.TestCase):
    def test_rerank_123_preserves_order(self):
        chunks = [{"text": str(i), "payload": {}} for i in range(1, 7)]
        ordered = reorder_context_chunks(chunks, "rerank_123")
        self.assertEqual([x["text"] for x in ordered], ["1", "2", "3", "4", "5", "6"])

    def test_rerank_132_swaps_middle_and_last_thirds(self):
        chunks = [{"text": str(i), "payload": {}} for i in range(1, 7)]
        ordered = reorder_context_chunks(chunks, "rerank_132")
        self.assertEqual([x["text"] for x in ordered], ["1", "2", "5", "6", "3", "4"])

    def test_rerank_132_handles_remainder(self):
        chunks = [{"text": str(i), "payload": {}} for i in range(1, 6)]
        ordered = reorder_context_chunks(chunks, "rerank_132")
        self.assertEqual([x["text"] for x in ordered], ["1", "2", "5", "3", "4"])

    def test_build_context_uses_configured_order(self):
        chunks = [{"text": str(i), "payload": {"external_id": str(i)}} for i in range(1, 7)]
        cfg = RAGConfig(context_order="rerank_132", max_context_chars=1000, max_chunk_chars=100)
        context, sources = build_context(cfg, chunks)
        self.assertIn("[3] 5", context)
        self.assertEqual([x["external_id"] for x in sources], ["1", "2", "5", "6", "3", "4"])


if __name__ == "__main__":
    unittest.main()
