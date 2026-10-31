import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from backend.rag import RAGChunkInput, RAGRecordingInput, VoiceMemoRAG


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


if __name__ == "__main__":
    unittest.main()
