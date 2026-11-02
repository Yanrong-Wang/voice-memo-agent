# AI Voice Memo Agent

An iPhone voice-recording app built with SwiftUI and AVFoundation. It combines
Voice Memos-style recording and editing with AI transcription, structured notes,
timestamp-linked retrieval-augmented generation (RAG), and a tool-calling agent
that can organize recordings with user approval.

## Features

### Audio

- Record, pause, resume, and play local `.m4a` audio
- Display live and persisted waveforms
- Seek, skip forward or backward, trim, and delete selected audio segments
- Rename, favorite, duplicate, share, and delete recordings

### AI analysis

- Automatically transcribe and analyze newly saved recordings
- Generate a short summary, detailed summary, key points, and action items
- Suggest titles, tags, and a category
- Cache analysis with `not_started`, `processing`, `ready`, `failed`, and
  `expired` states
- Invalidate stale analysis after an audio edit

### RAG and agent

- Split transcripts into timestamp-linked chunks with LangChain
- Store embeddings and metadata in PostgreSQL with pgvector
- Search semantically across multiple recordings
- Answer questions from retrieved transcript evidence and return source cards
- Open a cited recording near the relevant audio segment
- Use OpenAI tool calling to search, summarize, rename, tag, and delete recordings
- Require human confirmation before metadata changes or deletion
- Preserve conversational references such as "the second result" or "that recording"

## Architecture

```text
iPhone SwiftUI app
  |-- AVFoundation: recording, playback, waveform sampling, and editing
  |-- RecordingStore: local .m4a files and JSON metadata
  |-- AIController: analysis state and agent conversations
  `-- URLSession: multipart audio and JSON requests
                    |
                    v
FastAPI service
  |-- OpenAI Audio Transcriptions API: speech to text
  |-- OpenAI Responses API: structured analysis and tool calling
  |-- LangChain: transcript splitting, retrieval, prompting, and RAG
  |-- OpenAI Embeddings: semantic vector representations
  `-- PostgreSQL + pgvector: persistent vector search and metadata
                    |
                    v
                 OpenAI API
```

The iPhone remains the source of truth for recordings and their metadata. When
the user searches or asks the agent a question, the app sends the current
recording context to FastAPI. The backend incrementally reconciles new, edited,
renamed, and deleted recordings with the RAG index.

The OpenAI API key is stored only in `backend/.env`; it is never embedded in the
iOS application.

## Data Flow

### Recording analysis

```text
Local .m4a audio
  -> multipart upload
  -> transcription
  -> structured summary, key points, and action items
  -> JSON response
  -> cached in the recording's local metadata
```

### RAG question answering

```text
Transcript chunks
  -> embeddings
  -> pgvector
  -> similarity search for a user question
  -> retrieved evidence
  -> grounded answer with source citations
```

## Project Structure

```text
VoiceMemoAgent/
  Models/       Recording, transcript, analysis, and agent data models
  Services/     Audio, persistence, networking, and AI orchestration
  Views/        SwiftUI recording, editing, summary, and agent screens

backend/
  app.py        FastAPI routes, OpenAI analysis, and agent tools
  rag.py        LangChain retrieval and pgvector/Chroma persistence
  tests/        Backend RAG tests
  docker-compose.yml
                Local PostgreSQL 16 and pgvector service
```

## Requirements

- macOS with Xcode 16.4 or later
- Python 3.9 or later
- Docker Desktop, unless using Chroma or a managed PostgreSQL database
- An OpenAI API key

## Setup

### 1. Start PostgreSQL with pgvector

Install and open Docker Desktop, then run this command from the project root:

```bash
docker compose -f backend/docker-compose.yml up -d
```

This starts PostgreSQL 16 with pgvector on `127.0.0.1:5432`. Database files are
stored in a named Docker volume and persist across container restarts.

You can instead use a managed PostgreSQL provider that supports pgvector. Set
its connection string in `POSTGRES_CONNECTION_STRING`.

### 2. Configure and start FastAPI

```bash
cp backend/.env.example backend/.env
open -e backend/.env
```

Set your API key:

```text
OPENAI_API_KEY=your_api_key
```

Start the backend:

```bash
bash backend/start.sh
```

The API is available at `http://127.0.0.1:8000`. Open
`http://127.0.0.1:8000/docs` for the generated API documentation.

Check the active vector store:

```bash
curl http://127.0.0.1:8000/rag/status
```

A successful pgvector setup returns a response containing
`"vector_store":"pgvector"`. To run without PostgreSQL, set
`VECTOR_STORE_PROVIDER=chroma` in `backend/.env`.

### 3. Run the iOS app

1. Open `VoiceMemoAgent.xcodeproj` in Xcode.
2. Select your development team under **Signing & Capabilities**.
3. Choose an iPhone simulator or a physical device and run the app.
4. Use `http://127.0.0.1:8000` as the AI server URL in the simulator.
5. For a physical iPhone, connect it to the same network as the Mac and use the
   Mac's `.local` address, such as `http://Amy-Mac.local:8000`.

Find the Mac's local hostname with:

```bash
scutil --get LocalHostName
```

## Testing

Run backend tests:

```bash
backend/.venv/bin/python -m unittest discover backend/tests
```

Build the iOS target without code signing:

```bash
xcodebuild \
  -project VoiceMemoAgent.xcodeproj \
  -scheme VoiceMemoAgent \
  -destination 'generic/platform=iOS' \
  -derivedDataPath /tmp/VoiceMemoAgentDerivedData \
  CODE_SIGNING_ALLOWED=NO \
  build
```

## Storage, Cost, and Current Limitations

- Audio files and recording metadata remain on the iPhone.
- Cached analysis prevents repeated model calls when a ready summary is opened.
- Model calls occur for initial analysis, explicit retries, and regeneration
  after audio edits.
- pgvector stores transcript chunks, embeddings, and metadata, but not audio.
- Local pgvector data lives in a Docker volume; managed PostgreSQL data lives
  with the selected provider.
- Source timestamps are currently estimated from each chunk's relative position
  in the transcript, not from word-level transcription timestamps.
- The current backend is designed for a single user. A multi-user deployment
  would require authentication, tenant isolation, persistent agent sessions,
  HTTPS, and production secret management.
