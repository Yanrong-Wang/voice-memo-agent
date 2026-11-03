import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from backend.rag import (
    RAGChunkInput,
    RAGRecordingInput,
    RAGSource,
    RerankSelection,
    VoiceMemoRAG,
)


class KeywordEmbeddings:
    vocabulary = ("rag", "vector", "interview", "react", "migration")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        normalized = text.lower()
        vector = [float(normalized.count(word)) for word in self.vocabulary]
        vector.append(1.0)
        return vector


class VoiceMemoRAGTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        chat_model = RunnableLambda(
            lambda _: AIMessage(content="RAG was discussed in the interview. [S1]")
        )
        self.rag = VoiceMemoRAG(
            persist_directory=Path(self.temporary_directory.name),
            embeddings=KeywordEmbeddings(),
            chat_model=chat_model,
            min_score=0,
        )
        self.recordings = [
            RAGRecordingInput(
                id="recording-1",
                title="AI Engineer Interview Review",
                transcript="We discussed RAG, vector retrieval, and evaluation.",
                duration=120,
                created_at="2026-06-15T10:00:00Z",
                chunks=[
                    RAGChunkInput(
                        id="chunk_001",
                        text="We discussed RAG, vector retrieval, and evaluation.",
                        start_time=30,
                        end_time=55,
                    )
                ],
                tags=["interview", "AI"],
            ),
            RAGRecordingInput(
                id="recording-2",
                title="React Native Migration Notes",
                transcript="Plan the React migration and navigation changes.",
                duration=90,
                created_at="2026-06-14T10:00:00Z",
            ),
        ]
        self.rag.sync_recordings(self.recordings)

    def tearDown(self):
        del self.rag
        self.temporary_directory.cleanup()

    def test_search_returns_timestamped_source(self):
        results = self.rag.search("RAG vector interview", top_k=2)

        self.assertTrue(results)
        self.assertEqual(results[0].recording_id, "recording-1")
        self.assertEqual(results[0].start_time, 30)
        self.assertEqual(results[0].end_time, 55)

    def test_answer_uses_retrieved_context_and_returns_sources(self):
        answer = self.rag.answer("What did I say about RAG?", top_k=2)

        self.assertIn("[S1]", answer.answer)
        self.assertEqual(answer.sources[0].recording_id, "recording-1")

    def test_sync_removes_recordings_missing_from_source_of_truth(self):
        self.rag.sync_recordings(self.recordings[:1])

        status = self.rag.status()
        self.assertEqual(status["recordings"], 1)
        self.assertEqual(status["chunks"], 1)

    def test_recording_index_is_isolated_by_owner(self):
        owner_a = RAGRecordingInput(
            id="shared-id",
            title="Owner A memo",
            transcript="The interview covered RAG and vector evaluation.",
            duration=30,
            created_at="2026-06-15T10:00:00Z",
        )
        owner_b = RAGRecordingInput(
            id="shared-id",
            title="Owner B memo",
            transcript="The product team planned a React migration.",
            duration=30,
            created_at="2026-06-15T10:00:00Z",
        )

        self.rag.sync_recordings([owner_a], owner_id="owner-a")
        self.rag.sync_recordings([owner_b], owner_id="owner-b")

        owner_a_results = self.rag.search("RAG vector", owner_id="owner-a")
        owner_b_results = self.rag.search("React migration", owner_id="owner-b")
        self.assertEqual(owner_a_results[0].recording_title, "Owner A memo")
        self.assertEqual(owner_b_results[0].recording_title, "Owner B memo")

        self.rag.sync_recordings([], owner_id="owner-a")
        self.assertEqual(self.rag.status(owner_id="owner-a")["recordings"], 0)
        self.assertEqual(self.rag.status(owner_id="owner-b")["recordings"], 1)

    def test_timestamp_format(self):
        self.assertEqual(VoiceMemoRAG.format_timestamp(65), "01:05")
        self.assertEqual(VoiceMemoRAG.format_timestamp(3_661), "01:01:01")

    def test_pgvector_requires_connection_string(self):
        with self.assertRaisesRegex(ValueError, "POSTGRES_CONNECTION_STRING"):
            VoiceMemoRAG(
                persist_directory=Path(self.temporary_directory.name) / "postgres",
                vector_store_provider="pgvector",
                embeddings=KeywordEmbeddings(),
                chat_model=RunnableLambda(
                    lambda _: AIMessage(content="unused")
                ),
            )

    def test_hybrid_fusion_boosts_exact_lexical_matches(self):
        semantic_match = RAGSource(
            recording_id="semantic",
            recording_title="Semantic match",
            chunk_id="chunk_1",
            text="A related discussion without the exact product term.",
            start_time=0,
            end_time=10,
            score=0.42,
        )
        exact_match = RAGSource(
            recording_id="exact",
            recording_title="Exact match",
            chunk_id="chunk_1",
            text="The team chose React Native for the migration.",
            start_time=20,
            end_time=30,
            score=0.38,
        )
        lexical_match = RAGSource(
            **{**exact_match.__dict__, "score": 1.0}
        )

        results = VoiceMemoRAG._fuse_sources(
            [semantic_match, exact_match],
            [lexical_match],
            top_k=2,
            min_score=0.2,
        )

        self.assertEqual(results[0].recording_id, "exact")
        self.assertGreater(results[0].score, results[1].score)

    def test_reranker_can_prefer_the_answer_bearing_passage(self):
        first = RAGSource(
            recording_id="recording-1",
            recording_title="Feature discussion",
            chunk_id="context",
            text="The team wants to combine multiple functions.",
            start_time=0,
            end_time=20,
            score=0.45,
        )
        second = RAGSource(
            recording_id="recording-1",
            recording_title="Feature discussion",
            chunk_id="answer",
            text="Palm Pilots were the example of a multifunction device.",
            start_time=20,
            end_time=40,
            score=0.44,
        )
        self.rag.rerank_chain = RunnableLambda(
            lambda _: RerankSelection(ordered_indexes=[2, 1])
        )

        results = self.rag._rerank(
            "What device was the example?",
            [first, second],
            top_k=2,
        )

        self.assertEqual(results[0].chunk_id, "answer")
        self.assertEqual(results[0].start_time, 20)

    def test_reranker_can_reject_all_candidates(self):
        candidate = RAGSource(
            recording_id="recording-1",
            recording_title="Unrelated memo",
            chunk_id="context",
            text="This passage does not answer the question.",
            start_time=0,
            end_time=10,
            score=0.2,
        )
        self.rag.rerank_chain = RunnableLambda(
            lambda _: RerankSelection(ordered_indexes=[])
        )

        self.assertEqual(self.rag._rerank("A different topic", [candidate], 1), [])


if __name__ == "__main__":
    unittest.main()
