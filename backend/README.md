# AI backend

This local FastAPI service keeps the OpenAI API key out of the iPhone app. Audio
is uploaded only when analysis is requested. The iPhone remains the source of
truth for recordings and metadata. LangChain splits timestamped transcripts,
OpenAI Embeddings indexes them in PostgreSQL with pgvector, and a retrieval
chain generates answers with source citations. Chroma remains available as a
zero-setup fallback.

## Setup

```bash
cd backend
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Python 3.9 or newer is supported by the pinned LangChain 0.x dependency range.

Edit `.env` and set `OPENAI_API_KEY`, then start the service:

```bash
docker compose up -d
```

The included Compose service runs PostgreSQL 16 with the pgvector extension.
Its credentials are for local development only. For a deployed backend, replace
`POSTGRES_CONNECTION_STRING` with a managed PostgreSQL connection string.

Then start FastAPI:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

The first Agent question synchronizes the app's transcript chunks into pgvector.
Inspect the active store with:

```bash
curl http://127.0.0.1:8000/rag/status
```

For a zero-setup local run, set `VECTOR_STORE_PROVIDER=chroma`. Chroma writes to
`backend/data/chroma`, which is excluded from Git.

The iOS Simulator can use `http://127.0.0.1:8000`. A physical iPhone must use
the Mac's local hostname or LAN IP, for example
`http://your-mac-name.local:8000`, and both devices must be on the same network.
