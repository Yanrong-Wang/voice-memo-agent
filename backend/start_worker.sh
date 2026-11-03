#!/bin/zsh
set -e

cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)/.."

REDIS_URL=$(
    .venv/bin/python -c \
        'from dotenv import dotenv_values; print(dotenv_values(".env").get("REDIS_URL", "redis://127.0.0.1:6380/0"))'
)

exec .venv/bin/rq worker \
    --url "$REDIS_URL" \
    --worker-class rq.worker.SimpleWorker \
    voice-memo-analysis
