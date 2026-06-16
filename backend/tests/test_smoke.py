import unittest

from fastapi.testclient import TestClient

from backend.app import app, split_text, transcript_chunks


class BackendSmokeTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_is_available_without_an_api_key(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_text_chunking_preserves_short_text(self):
        self.assertEqual(split_text("hello world", size=20), ["hello world"])

    def test_transcript_chunk_has_estimated_timing(self):
        chunks = transcript_chunks("one two three four", duration=12)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start_time, 0)
        self.assertEqual(chunks[0].end_time, 12)

    def test_confirm_without_pending_action_is_safe(self):
        response = self.client.post(
            "/agent/confirm",
            json={"session_id": "missing-session", "confirm": True},
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["success"])


if __name__ == "__main__":
    unittest.main()
