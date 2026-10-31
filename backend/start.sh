#!/bin/zsh
set -e

cd "$(dirname "$0")"

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.txt
fi

if ! .venv/bin/python -c 'import langchain_chroma, langchain_openai, langchain_postgres' 2>/dev/null; then
  .venv/bin/python -m pip install -r requirements.txt
fi

if [[ ! -f ".env" ]]; then
  cp .env.example .env
  echo "Created backend/.env. Add your OPENAI_API_KEY, then run this script again."
  exit 1
fi

if ! grep -q '^OPENAI_API_KEY=..' .env; then
  echo "OPENAI_API_KEY is empty in backend/.env."
  exit 1
fi

if grep -q '^VECTOR_STORE_PROVIDER=pgvector' .env; then
  if ! .venv/bin/python - <<'PY'
from dotenv import dotenv_values
import psycopg

settings = dotenv_values(".env")
connection_string = settings.get("POSTGRES_CONNECTION_STRING", "")
connection_string = connection_string.replace(
    "postgresql+psycopg://",
    "postgresql://",
    1,
).replace("postgres://", "postgresql://", 1)
if not connection_string:
    raise SystemExit(1)
with psycopg.connect(connection_string, connect_timeout=3) as connection:
    connection.execute("SELECT 1")
PY
  then
    echo "pgvector is unavailable. Start it with:"
    echo "docker compose -f docker-compose.yml up -d"
    echo "Or check POSTGRES_CONNECTION_STRING in backend/.env."
    exit 1
  fi
fi

exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload
