"""Redis/RQ-backed durable analysis jobs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from redis import Redis
from rq import Queue
from rq.job import Job


QUEUE_NAME = "voice-memo-analysis"


def job_data_root() -> Path:
    return Path(
        os.getenv("ANALYSIS_DATA_DIRECTORY", str(Path(__file__).parent / "data"))
    ).resolve()


def redis_connection() -> Redis:
    return Redis.from_url(
        os.getenv("REDIS_URL", "redis://127.0.0.1:6380/0"),
        decode_responses=False,
    )


def analysis_queue() -> Queue:
    return Queue(QUEUE_NAME, connection=redis_connection())


def enqueue_analysis(
    *,
    recording_id: str,
    owner_id: str,
    audio_path: Path,
    duration: float,
) -> Job:
    timeout = int(os.getenv("ANALYSIS_JOB_TIMEOUT_SECONDS", "900"))
    try:
        relative_audio_path = audio_path.resolve().relative_to(job_data_root())
    except ValueError as error:
        raise ValueError("Analysis audio must be stored under the backend data directory.") from error
    return analysis_queue().enqueue(
        "backend.job_queue.run_analysis_job",
        recording_id=recording_id,
        audio_path=str(relative_audio_path),
        duration=duration,
        job_timeout=timeout,
        result_ttl=7 * 24 * 60 * 60,
        failure_ttl=7 * 24 * 60 * 60,
        meta={"owner_id": owner_id, "recording_id": recording_id},
        description=f"Analyze voice memo {recording_id}",
    )


def fetch_job(job_id: str) -> Job:
    return Job.fetch(job_id, connection=redis_connection())


def run_analysis_job(
    *,
    recording_id: str,
    audio_path: str,
    duration: float,
) -> dict[str, Any]:
    del recording_id
    path = job_data_root() / audio_path
    try:
        from backend.app import analyze_audio

        result = analyze_audio(path, duration)
        return result.model_dump(mode="json")
    finally:
        path.unlink(missing_ok=True)
