import unittest

from backend.session_store import MemorySessionStore


class MemorySessionStoreTests(unittest.TestCase):
    def test_saved_state_is_isolated_from_caller_mutation(self):
        store = MemorySessionStore()
        state = {"history": [{"role": "user", "content": "hello"}]}

        store.save("session-1", state)
        state["history"][0]["content"] = "changed"

        self.assertEqual(store.load("session-1")["history"][0]["content"], "hello")


if __name__ == "__main__":
    unittest.main()
