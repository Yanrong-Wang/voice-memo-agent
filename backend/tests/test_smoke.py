from io import BytesIO
import unittest
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from backend.app import (
    aligned_transcript_chunks,
    app,
    copy_upload_with_limit,
    split_text,
    transcript_chunks,
    transcription_request_kwargs,
    validate_audio_upload,
)


class BackendSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_is_available_without_an_api_key(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_shared_token_authentication_rejects_missing_token(self):
        with patch("backend.app.JWT_JWKS_URL", ""), patch(
            "backend.app.API_BEARER_TOKEN",
            "test-secret",
        ):
            denied = self.client.post(
                "/agent/confirm",
                json={"session_id": "auth-check", "confirm": True},
            )
            allowed = self.client.post(
                "/agent/confirm",
                headers={"Authorization": "Bearer test-secret"},
                json={"session_id": "auth-check", "confirm": True},
            )

        self.assertEqual(denied.status_code, 401)
        self.assertEqual(allowed.status_code, 200)

    def test_text_chunking_preserves_short_text(self):
        self.assertEqual(split_text("hello world", size=20), ["hello world"])

    def test_transcript_chunk_has_estimated_timing(self):
        chunks = transcript_chunks("one two three four", duration=12)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_time, 0)
        self.assertEqual(chunks[0].end_time, 12)

    def test_transcript_chunking_splits_long_speech_on_sentences(self):
        text = " ".join(
            [
                "The team reviewed retrieval quality and citation accuracy.",
                "They compared vector search with exact keyword matching.",
                "The next step is to evaluate real meeting audio.",
            ]
            * 4
        )

        chunks = transcript_chunks(text, duration=48)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].start_time, 0)
        self.assertEqual(chunks[-1].end_time, 48)
        self.assertTrue(
            all(left.end_time <= right.start_time for left, right in zip(chunks, chunks[1:]))
        )

    def test_upload_copy_rejects_files_over_the_limit(self):
        with self.assertRaises(HTTPException) as context:
            copy_upload_with_limit(BytesIO(b"12345"), BytesIO(), limit=4)

        self.assertEqual(context.exception.status_code, 413)

    def test_upload_validation_rejects_non_audio_files(self):
        upload = UploadFile(
            file=BytesIO(b"not audio"),
            filename="notes.txt",
            headers=Headers({"content-type": "text/plain"}),
        )

        with self.assertRaises(HTTPException) as context:
            validate_audio_upload(upload, duration=10)

        self.assertEqual(context.exception.status_code, 415)

    def test_transcription_prompt_is_sent_only_when_configured(self):
        audio_file = object()
        with patch("backend.app.TRANSCRIPTION_PROMPT", "Preserve pgvector."):
            options = transcription_request_kwargs(audio_file)
        self.assertEqual(options["prompt"], "Preserve pgvector.")

        with patch("backend.app.TRANSCRIPTION_PROMPT", ""):
            options = transcription_request_kwargs(audio_file)
        self.assertNotIn("prompt", options)

    def test_word_alignment_produces_real_chunk_boundaries(self):
        transcript = "Hello world. This is a timestamp test."
        words = [
            {"word": "Hello", "start": 1.2, "end": 1.6},
            {"word": "world", "start": 1.7, "end": 2.1},
            {"word": "This", "start": 4.0, "end": 4.2},
            {"word": "is", "start": 4.3, "end": 4.4},
            {"word": "a", "start": 4.5, "end": 4.6},
            {"word": "timestamp", "start": 4.7, "end": 5.2},
            {"word": "test", "start": 5.3, "end": 5.8},
        ]

        chunks = aligned_transcript_chunks(transcript, words)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_time, 1.2)
        self.assertEqual(chunks[0].end_time, 5.8)
        self.assertEqual(chunks[0].timing_source, "word")

    def test_cjk_alignment_uses_character_level_timestamps(self):
        transcript = "这并不是告别。这是新篇章。"
        words = [
            {"word": "这并不是", "start": 0.2, "end": 1.0},
            {"word": "告别", "start": 1.1, "end": 1.5},
            {"word": "这是", "start": 1.8, "end": 2.2},
            {"word": "新篇章", "start": 2.3, "end": 3.2},
        ]

        chunks = aligned_transcript_chunks(transcript, words)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_time, 0.2)
        self.assertEqual(chunks[0].end_time, 3.2)
        self.assertEqual(chunks[0].timing_source, "word")

    def test_confirm_without_pending_action_is_safe(self):
        response = self.client.post(
            "/agent/confirm",
            json={"session_id": "missing-session", "confirm": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["success"])


if __name__ == "__main__":
    unittest.main()
