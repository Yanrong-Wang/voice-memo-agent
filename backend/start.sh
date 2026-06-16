#!/bin/zsh
set -e

cd "$(dirname "$0")"

if [[ ! -x ".venv/bin/python" ]]; then
  python3 -m venv .venv
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

exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000 --reload
