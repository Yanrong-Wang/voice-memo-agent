# AI backend

FastAPI keeps the OpenAI API key outside the iPhone app and exposes audio
analysis, timestamp-aware RAG, and tool-calling Agent endpoints. The iPhone is
the source of truth for audio and metadata; the backend stores transcript
chunks, embeddings, job state, and Agent conversation state.

## Architecture

- OpenAI Audio Transcriptions produces the primary transcript.
- An optional timestamp pass returns word timings, which are aligned to the
  primary transcript. Failed or low-confidence alignment falls back to clearly
  marked estimated timestamps.
- OpenAI Responses returns structured summaries, key points, action items,
  titles, tags, and categories.
- LangChain splits transcript evidence and orchestrates grounded answers.
- PostgreSQL/pgvector combines vector similarity with PostgreSQL full-text
  search. An LLM reranker promotes passages containing the direct answer.
- Redis and RQ run audio analysis outside the HTTP process. Results and failures
  remain queryable after the request completes.
- PostgreSQL persists Agent history and pending confirmations for 30 days.
- Shared bearer tokens or OIDC/JWT verification, Redis rate limiting, per-user
  RAG namespaces, request IDs, and Prometheus metrics protect deployed APIs.
- Chroma remains available as a local vector-only fallback.

## Setup

```bash
cd backend
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. For specialized names, optionally set
`OPENAI_TRANSCRIPTION_PROMPT` to a short vocabulary hint.

Start PostgreSQL, Redis, and the RQ worker:

```bash
docker compose up -d --build
```

Start FastAPI in another terminal:

```bash
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs`. Health and Prometheus metrics are exposed at
`/health` and `/metrics`. `/rag/status` reports the current authenticated
user's indexed recording and chunk counts.

The iOS Simulator can use `http://127.0.0.1:8000`. A physical iPhone must use
the Mac's LAN address or `.local` hostname while both devices share a network.

## Authentication and isolation

Local development may leave authentication disabled. For deployment, configure
`JWT_JWKS_URL`, `JWT_ISSUER`, and `JWT_AUDIENCE` for an OIDC provider. The
verified JWT `sub` claim scopes Agent sessions, RAG synchronization, vector
retrieval, and full-text retrieval. `API_BEARER_TOKEN` is a simpler shared-token
option for private testing, not a multi-user identity system.

The server rejects unsupported uploads, files over 25 MB, and recordings over
four hours by default. Configure those limits and the per-identity request limit
in `.env`. Rate limiting fails open if Redis is unavailable so a Redis outage
does not take down read traffic; authentication still fails closed.

## Timestamp accuracy

`OPENAI_TIMESTAMP_MODEL=whisper-1` enables the current word-timestamp alignment
path. This is transitional because OpenAI has announced that model's shutdown
for 2027-02-26. Production deployment should replace it with a maintained
provider or a local forced aligner. Every returned chunk includes
`timing_source` (`word`, `segment`, or `estimated`) so the client never presents
an estimated jump as frame-accurate.

## Real-audio evaluation

The repeatable benchmark uses three CC BY 4.0 AMI multi-speaker meeting clips.
It measures transcription WER, literal summary fact coverage, retrieval Top-1,
MRR, and answer-bearing passage coverage. A deterministic 10 dB noise variant
exercises degraded audio. A separate Google FLEURS smoke set measures English
WER and Mandarin CER on real speech.

```bash
.venv/bin/python evaluation/download_ami_sample.py
.venv/bin/python evaluation/evaluate_ami.py
.venv/bin/python evaluation/create_noisy_variants.py
.venv/bin/python evaluation/evaluate_ami.py \
  --manifest data/eval/ami/manifest-noise-10db.json
.venv/bin/python -m pip install -r evaluation/requirements.txt
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN \
  .venv/bin/python evaluation/download_fleurs_sample.py
.venv/bin/python evaluation/evaluate_multilingual.py
```

See `evaluation/README.md` for metric interpretation and the manifest format for
additional languages and noise conditions.
