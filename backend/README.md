# AI backend

This local FastAPI service keeps the OpenAI API key out of the iPhone app. Audio
is uploaded only when analysis is requested. The iPhone remains the source of
truth for recordings and metadata.

## Setup

```bash
cd backend
python3 --version
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Python 3.9 or newer is supported.

Edit `.env` and set `OPENAI_API_KEY`, then start the service:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for the generated API documentation.

The iOS Simulator can use `http://127.0.0.1:8000`. A physical iPhone must use
the Mac's local hostname or LAN IP, for example
`http://your-mac-name.local:8000`, and both devices must be on the same network.
