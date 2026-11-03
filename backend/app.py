from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import secrets
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from openai import OpenAI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

try:
    from .rag import (
        RAGChunkInput,
        RAGRecordingInput,
        RAGSource,
        VoiceMemoRAG,
    )
    from .session_store import MemorySessionStore, PostgresSessionStore
except ImportError:
    from rag import RAGChunkInput, RAGRecordingInput, RAGSource, VoiceMemoRAG
    from session_store import MemorySessionStore, PostgresSessionStore

load_dotenv()

app = FastAPI(
    title="AI Voice Memo Agent",
    version="2.0.0",
    description=(
        "Transcription, structured analysis, persistent LangChain RAG, and agent tools."
    ),
)

logger = logging.getLogger("voice_memo_agent")

AGENT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-5.4-mini")
ANALYSIS_MODEL = os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-5.4-mini")
TRANSCRIPTION_MODEL = os.getenv(
    "OPENAI_TRANSCRIPTION_MODEL",
    "gpt-transcribe",
)
TRANSCRIPTION_PROMPT = os.getenv("OPENAI_TRANSCRIPTION_PROMPT", "").strip()
TIMESTAMP_MODEL = os.getenv("OPENAI_TIMESTAMP_MODEL", "").strip()
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536"))
SEMANTIC_MIN_SCORE = float(os.getenv("SEMANTIC_MIN_SCORE", "0.15"))
HYBRID_SEARCH_ENABLED = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
}
RAG_RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
}
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6380/0").strip()
ANALYSIS_QUEUE_ENABLED = os.getenv("ANALYSIS_QUEUE_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
}
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()
JWT_JWKS_URL = os.getenv("JWT_JWKS_URL", "").strip()
JWT_ISSUER = os.getenv("JWT_ISSUER", "").strip()
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "").strip()
RATE_LIMIT_REQUESTS_PER_MINUTE = int(
    os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "120")
)
SESSION_STORE_PROVIDER = os.getenv("SESSION_STORE_PROVIDER", "memory").strip().lower()
MAX_AUDIO_UPLOAD_BYTES = int(
    os.getenv("MAX_AUDIO_UPLOAD_BYTES", str(25 * 1024 * 1024))
)
MAX_RECORDING_DURATION_SECONDS = float(
    os.getenv("MAX_RECORDING_DURATION_SECONDS", str(4 * 60 * 60))
)
VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "chroma").strip().lower()
POSTGRES_CONNECTION_STRING = (
    os.getenv("POSTGRES_CONNECTION_STRING", "").strip()
    or os.getenv("DATABASE_URL", "").strip()
)
RAG_COLLECTION_NAME = os.getenv(
    "RAG_COLLECTION_NAME",
    "voice_memo_transcript_chunks",
).strip()
RAG_DATA_DIRECTORY = Path(
    os.getenv("RAG_DATA_DIRECTORY", str(Path(__file__).parent / "data" / "chroma"))
)

ALLOWED_AUDIO_CONTENT_TYPES = {
    "application/octet-stream",
    "audio/aac",
    "audio/m4a",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
    "video/mp4",
}
ALLOWED_AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".oga",
    ".ogg",
    ".wav",
    ".webm",
}


class ActionItem(BaseModel):
    task: str
    priority: Literal["low", "medium", "high"] = "medium"
    source_timestamp: Optional[str] = None


class TranscriptChunk(BaseModel):
    id: str
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    timing_source: Literal["word", "segment", "estimated"] = "estimated"


class AnalysisPayload(BaseModel):
    summary: str = Field(description="A concise one or two sentence preview.")
    detailed_summary: str = Field(description="A fuller summary for the detail page.")
    key_points: list[str]
    action_items: list[ActionItem]
    suggested_title: Optional[str] = None
    suggested_tags: list[str]
    category: Optional[str] = None


class RecordingAnalysis(AnalysisPayload):
    transcript: str
    transcript_chunks: list[TranscriptChunk]


class AgentRecording(BaseModel):
    id: str
    title: str
    created_at: datetime
    duration: float
    transcript: Optional[str] = None
    transcript_chunks: list[TranscriptChunk] = Field(default_factory=list)
    summary: Optional[str] = None
    key_points: list[str] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    suggested_title: Optional[str] = None
    suggested_tags: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    category: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    recordings: list[AgentRecording]
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    recording_id: str
    title: str
    matched_text: str
    score: float
    chunk_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class SearchReply(BaseModel):
    results: list[SearchResult]


class AgentSource(BaseModel):
    recording_id: str
    recording_title: str
    chunk_id: str
    excerpt: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    score: float


class PendingAction(BaseModel):
    type: Literal["rename_recording", "tag_recording", "delete_recording"]
    recording_id: str
    new_title: Optional[str] = None
    tags: Optional[list[str]] = None
    recording_title: Optional[str] = None


class AgentAskRequest(BaseModel):
    session_id: str
    query: str
    current_recording_id: Optional[str] = None
    recordings: list[AgentRecording]


class AgentReply(BaseModel):
    message: str
    recording_ids: list[str] = Field(default_factory=list)
    sources: list[AgentSource] = Field(default_factory=list)
    requires_confirmation: bool = False
    pending_action: Optional[PendingAction] = None


class ConfirmRequest(BaseModel):
    session_id: str
    confirm: bool


class ConfirmReply(BaseModel):
    success: bool
    message: str
    action: Optional[PendingAction] = None


class SessionTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionState(BaseModel):
    last_search_results: list[str] = Field(default_factory=list)
    current_recording_id: Optional[str] = None
    pending_action: Optional[PendingAction] = None
    history: list[SessionTurn] = Field(default_factory=list)


class AnalysisJobSubmission(BaseModel):
    job_id: str
    status: str


class AnalysisJobStatus(BaseModel):
    job_id: str
    status: str
    result: Optional[RecordingAnalysis] = None
    error: Optional[str] = None


class TitleSuggestion(BaseModel):
    title: str


class TagSuggestion(BaseModel):
    tags: list[str]


_memory_session_store = MemorySessionStore()
_session_store: Optional[Any] = None
_session_store_lock = Lock()
_jwks_client: Optional[Any] = None
_rag_service: Optional[VoiceMemoRAG] = None
_rag_lock = Lock()

HTTP_REQUESTS = Counter(
    "voice_memo_http_requests_total",
    "HTTP requests handled by the voice memo backend.",
    ["method", "status"],
)
HTTP_LATENCY = Histogram(
    "voice_memo_http_request_duration_seconds",
    "HTTP request latency for the voice memo backend.",
    ["method"],
)


def authenticated_identity(request: Request) -> str:
    global _jwks_client
    authorization = request.headers.get("authorization", "")
    if JWT_JWKS_URL:
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="A bearer token is required.")
        try:
            import jwt

            if _jwks_client is None:
                _jwks_client = jwt.PyJWKClient(JWT_JWKS_URL)
            token = authorization.removeprefix("Bearer ").strip()
            signing_key = _jwks_client.get_signing_key_from_jwt(token)
            options = {"verify_aud": bool(JWT_AUDIENCE)}
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience=JWT_AUDIENCE or None,
                issuer=JWT_ISSUER or None,
                options=options,
            )
            subject = str(claims.get("sub", "")).strip()
            if not subject:
                raise ValueError("Token has no subject claim.")
            return subject
        except Exception as error:
            raise HTTPException(status_code=401, detail="Invalid bearer token.") from error
    if API_BEARER_TOKEN:
        expected = f"Bearer {API_BEARER_TOKEN}"
        if not secrets.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=401,
                detail="Missing or invalid API bearer token.",
            )
        return "shared-api-token"
    return request.client.host if request.client else "local"


@app.middleware("http")
async def security_and_observability(request: Request, call_next: Any) -> Response:
    started_at = time.monotonic()
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    public_paths = {"/health", "/metrics", "/docs", "/openapi.json", "/redoc"}

    identity = "public"
    if request.url.path not in public_paths:
        try:
            identity = authenticated_identity(request)
        except HTTPException as error:
            response = JSONResponse(
                status_code=error.status_code,
                content={"detail": error.detail},
            )
            response.headers["x-request-id"] = request_id
            return response
    request.state.user_id = identity

    if request.url.path not in public_paths and RATE_LIMIT_REQUESTS_PER_MINUTE > 0:
        try:
            from redis import Redis

            identity_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
            window = int(time.time() // 60)
            key = f"voice-memo:rate:{identity_hash}:{window}"
            redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
            count = redis_client.incr(key)
            if count == 1:
                redis_client.expire(key, 65)
            if count > RATE_LIMIT_REQUESTS_PER_MINUTE:
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded. Try again shortly."},
                )
                response.headers["retry-after"] = "60"
                response.headers["x-request-id"] = request_id
                return response
        except Exception as error:
            logger.warning("Rate limiter unavailable: %s", error)

    try:
        response = await call_next(request)
    except Exception:
        HTTP_REQUESTS.labels(request.method, "500").inc()
        HTTP_LATENCY.labels(request.method).observe(time.monotonic() - started_at)
        logger.exception("Unhandled request error request_id=%s", request_id)
        raise

    response.headers["x-request-id"] = request_id
    HTTP_REQUESTS.labels(request.method, str(response.status_code)).inc()
    HTTP_LATENCY.labels(request.method).observe(time.monotonic() - started_at)
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.monotonic() - started_at) * 1000,
    )
    return response


def get_session_store() -> Any:
    global _session_store
    with _session_store_lock:
        if _session_store is None:
            if SESSION_STORE_PROVIDER == "postgres":
                if not POSTGRES_CONNECTION_STRING:
                    raise HTTPException(
                        status_code=503,
                        detail="PostgreSQL is required for persistent Agent sessions.",
                    )
                _session_store = PostgresSessionStore(POSTGRES_CONNECTION_STRING)
            else:
                _session_store = _memory_session_store
        return _session_store


def load_session_state(session_id: str) -> SessionState:
    payload = get_session_store().load(session_id)
    return SessionState.model_validate(payload) if payload else SessionState()


def save_session_state(session_id: str, state: SessionState) -> None:
    state.history = state.history[-12:]
    get_session_store().save(session_id, state.model_dump(mode="json"))


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the AI server.",
        )
    return OpenAI(api_key=api_key)


def get_rag_service() -> VoiceMemoRAG:
    global _rag_service
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the AI server.",
        )
    with _rag_lock:
        if _rag_service is None:
            try:
                _rag_service = VoiceMemoRAG(
                    persist_directory=RAG_DATA_DIRECTORY,
                    vector_store_provider=VECTOR_STORE_PROVIDER,
                    postgres_connection=POSTGRES_CONNECTION_STRING or None,
                    collection_name=RAG_COLLECTION_NAME,
                    api_key=api_key,
                    embedding_model=EMBEDDING_MODEL,
                    embedding_dimensions=EMBEDDING_DIMENSIONS,
                    answer_model=AGENT_MODEL,
                    min_score=SEMANTIC_MIN_SCORE,
                    hybrid_search_enabled=HYBRID_SEARCH_ENABLED,
                    rerank_enabled=RAG_RERANK_ENABLED,
                )
            except Exception as error:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"The {VECTOR_STORE_PROVIDER} vector store is unavailable. "
                        "Check the backend database configuration."
                    ),
                ) from error
        return _rag_service


def rag_recording(recording: AgentRecording) -> RAGRecordingInput:
    return RAGRecordingInput(
        id=recording.id,
        title=recording.title,
        transcript=recording.transcript,
        duration=recording.duration,
        created_at=recording.created_at.isoformat(),
        chunks=[
            RAGChunkInput(
                id=chunk.id,
                text=chunk.text,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
            )
            for chunk in recording.transcript_chunks
        ],
        summary=recording.summary,
        tags=recording.tags,
        category=recording.category,
    )


def sync_rag(
    recordings: list[AgentRecording],
    *,
    owner_id: str = "local",
) -> VoiceMemoRAG:
    rag = get_rag_service()
    rag.sync_recordings(
        [rag_recording(recording) for recording in recordings],
        owner_id=owner_id,
    )
    return rag


def agent_source(source: RAGSource) -> AgentSource:
    return AgentSource(
        recording_id=source.recording_id,
        recording_title=source.recording_title,
        chunk_id=source.chunk_id,
        excerpt=source.text,
        start_time=source.start_time,
        end_time=source.end_time,
        score=source.score,
    )


def split_text(text: str, size: int = 1_200, overlap: int = 150) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def transcript_chunks(text: str, duration: float) -> list[TranscriptChunk]:
    normalized = " ".join(text.split())
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", normalized)
        if sentence.strip()
    ]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        pieces = split_text(sentence, size=420, overlap=0)
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > 420:
                chunks.append(current)
                current = piece
            else:
                current = candidate
    if current:
        chunks.append(current)
    if not chunks and normalized:
        chunks = [normalized]

    total_chars = max(1, sum(len(chunk) for chunk in chunks))
    cursor = 0
    result: list[TranscriptChunk] = []
    for index, chunk in enumerate(chunks):
        start_ratio = cursor / total_chars
        cursor += len(chunk)
        end_ratio = cursor / total_chars
        result.append(
            TranscriptChunk(
                id=f"chunk_{index + 1:03d}",
                text=chunk,
                start_time=round(duration * start_ratio, 2) if duration > 0 else None,
                end_time=round(duration * end_ratio, 2) if duration > 0 else None,
            )
        )
    return result


def _is_cjk(character: str) -> bool:
    return (
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
    )


def _alignment_units(value: str) -> list[tuple[str, int, int]]:
    """Tokenize spaced languages by word and CJK text by character."""
    units: list[tuple[str, int, int]] = []
    index = 0
    while index < len(value):
        character = value[index]
        if _is_cjk(character):
            units.append((character.casefold(), index, index + 1))
            index += 1
            continue
        if character.isalnum():
            end = index + 1
            while end < len(value):
                candidate = value[end]
                if _is_cjk(candidate) or not (
                    candidate.isalnum() or candidate in {"'", "\u2019"}
                ):
                    break
                end += 1
            token = "".join(
                candidate
                for candidate in value[index:end].casefold()
                if candidate.isalnum()
            )
            if token:
                units.append((token, index, end))
            index = end
            continue
        index += 1
    return units


def _timed_alignment_units(
    timed_words: list[dict[str, Any]],
) -> list[tuple[str, float, float]]:
    units: list[tuple[str, float, float]] = []
    for word in timed_words:
        if word.get("start") is None or word.get("end") is None:
            continue
        word_units = _alignment_units(str(word.get("word", "")))
        if not word_units:
            continue
        start = float(word["start"])
        end = float(word["end"])
        duration = max(0.0, end - start)
        for index, (token, _, _) in enumerate(word_units):
            unit_start = start + duration * index / len(word_units)
            unit_end = start + duration * (index + 1) / len(word_units)
            units.append((token, unit_start, unit_end))
    return units


def aligned_transcript_chunks(
    transcript: str,
    timed_words: list[dict[str, Any]],
) -> list[TranscriptChunk]:
    primary_units = _alignment_units(transcript)
    timestamped_units = _timed_alignment_units(timed_words)
    if not primary_units or not timestamped_units:
        return []

    primary_tokens = [unit[0] for unit in primary_units]
    timed_tokens = [unit[0] for unit in timestamped_units]
    matcher = difflib.SequenceMatcher(
        None,
        primary_tokens,
        timed_tokens,
        autojunk=False,
    )
    if matcher.ratio() < 0.55:
        return []

    mapped: list[Optional[tuple[float, float]]] = [None] * len(primary_units)
    for operation, first_start, first_end, second_start, second_end in matcher.get_opcodes():
        if operation == "equal":
            for offset in range(first_end - first_start):
                _, start, end = timestamped_units[second_start + offset]
                mapped[first_start + offset] = (start, end)
        elif operation == "replace" and second_end > second_start:
            replacement_count = second_end - second_start
            for index in range(first_start, first_end):
                relative = (index - first_start) / max(1, first_end - first_start)
                word_index = min(
                    second_end - 1,
                    second_start + int(relative * replacement_count),
                )
                _, start, end = timestamped_units[word_index]
                mapped[index] = (start, end)

    for index, value in enumerate(mapped):
        if value is not None:
            continue
        previous = next(
            (mapped[candidate] for candidate in range(index - 1, -1, -1) if mapped[candidate]),
            None,
        )
        following = next(
            (
                mapped[candidate]
                for candidate in range(index + 1, len(mapped))
                if mapped[candidate]
            ),
            None,
        )
        if previous and following:
            midpoint = (previous[1] + following[0]) / 2
            mapped[index] = (midpoint, midpoint)
        elif previous:
            mapped[index] = (previous[1], previous[1])
        elif following:
            mapped[index] = (following[0], following[0])

    chunks: list[TranscriptChunk] = []
    chunk_start = 0
    for index, (_, _, unit_end) in enumerate(primary_units):
        is_last = index == len(primary_units) - 1
        next_start = primary_units[index + 1][1] if not is_last else len(transcript)
        trailing_text = transcript[unit_end:next_start]
        character_count = unit_end - primary_units[chunk_start][1]
        start_time = mapped[chunk_start][0] if mapped[chunk_start] else 0.0
        end_time = mapped[index][1] if mapped[index] else start_time
        sentence_end = bool(re.search(r"[.!?\u3002\uff01\uff1f]", trailing_text))
        should_close = is_last or (
            sentence_end and (character_count >= 320 or end_time - start_time >= 22)
        ) or character_count >= 650
        if not should_close:
            continue

        text_start = primary_units[chunk_start][1]
        text_end = next_start
        chunk_text = transcript[text_start:text_end].strip()
        if chunk_text:
            chunks.append(
                TranscriptChunk(
                    id=f"chunk_{len(chunks) + 1:03d}",
                    text=chunk_text,
                    start_time=round(start_time, 2),
                    end_time=round(end_time, 2),
                    timing_source="word",
                )
            )
        chunk_start = index + 1
    return chunks


def validate_audio_upload(audio: UploadFile, duration: float) -> str:
    if duration <= 0:
        raise HTTPException(status_code=422, detail="Recording duration must be positive.")
    if duration > MAX_RECORDING_DURATION_SECONDS:
        raise HTTPException(
            status_code=413,
            detail="Recording duration exceeds the configured limit.",
        )

    content_type = (audio.content_type or "application/octet-stream").lower()
    if content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio content type.")

    suffix = Path(audio.filename or "recording.m4a").suffix.lower() or ".m4a"
    if suffix not in ALLOWED_AUDIO_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported audio file extension.")
    return suffix


def copy_upload_with_limit(source: Any, destination: Any, limit: int) -> int:
    copied = 0
    while chunk := source.read(1024 * 1024):
        copied += len(chunk)
        if copied > limit:
            raise HTTPException(
                status_code=413,
                detail="Audio upload exceeds the configured size limit.",
            )
        destination.write(chunk)
    return copied


def transcription_request_kwargs(audio_file: Any) -> dict[str, Any]:
    options: dict[str, Any] = {
        "model": TRANSCRIPTION_MODEL,
        "file": audio_file,
        "response_format": "text",
    }
    if TRANSCRIPTION_PROMPT:
        options["prompt"] = TRANSCRIPTION_PROMPT
    return options


def transcribe_audio(
    client: OpenAI,
    audio_path: Path,
    duration: float,
) -> tuple[str, list[TranscriptChunk]]:
    with audio_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            **transcription_request_kwargs(audio_file)
        )
    transcript = (
        transcription
        if isinstance(transcription, str)
        else getattr(transcription, "text", str(transcription))
    ).strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="The recording contains no speech.")

    if TIMESTAMP_MODEL:
        try:
            with audio_path.open("rb") as timestamp_audio:
                timestamped = client.audio.transcriptions.create(
                    model=TIMESTAMP_MODEL,
                    file=timestamp_audio,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                    **({"prompt": TRANSCRIPTION_PROMPT} if TRANSCRIPTION_PROMPT else {}),
                )
            words = [
                {
                    "word": getattr(word, "word", ""),
                    "start": getattr(word, "start", None),
                    "end": getattr(word, "end", None),
                }
                for word in (getattr(timestamped, "words", None) or [])
            ]
            aligned = aligned_transcript_chunks(transcript, words)
            if aligned:
                return transcript, aligned
            logger.warning("Word alignment quality was insufficient; using estimated timing.")
        except Exception as error:
            logger.warning("Timestamp provider failed; using estimated timing: %s", error)
    return transcript, transcript_chunks(transcript, duration)


def timestamp_label(seconds: Optional[float]) -> str:
    value = max(0, int(seconds or 0))
    minutes, remaining = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining:02d}"
    return f"{minutes:02d}:{remaining:02d}"


def summarize_long_transcript(client: OpenAI, transcript: str) -> str:
    chunks = split_text(transcript, size=10_000, overlap=300)
    if len(chunks) == 1:
        return transcript

    summaries: list[str] = []
    for index, chunk in enumerate(chunks):
        response = client.responses.create(
            model=ANALYSIS_MODEL,
            instructions=(
                "Summarize this transcript section faithfully. Preserve decisions, "
                "names, technical terms, and action items. Use the transcript language."
            ),
            input=f"Section {index + 1} of {len(chunks)}:\n\n{chunk}",
        )
        summaries.append(response.output_text)
    return "\n\n".join(summaries)


def analyze_audio(audio_path: Path, duration: float) -> RecordingAnalysis:
    if duration < 3:
        raise HTTPException(
            status_code=422,
            detail="Recording is too short to summarize.",
        )

    client = get_client()
    transcript, chunks = transcribe_audio(client, audio_path, duration)

    if len(transcript) <= 10_000:
        source = "\n".join(
            f"[{timestamp_label(chunk.start_time)}] {chunk.text}" for chunk in chunks
        )
    else:
        source = summarize_long_transcript(client, transcript)
    response = client.responses.parse(
        model=ANALYSIS_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "Analyze a voice memo without inventing facts. Return concise, "
                    "useful structured notes in the language primarily used in the "
                    "transcript. Action items must be explicit tasks actually supported "
                    "by the transcript. Set source_timestamp only to a timestamp shown "
                    "in square brackets next to supporting text. Use an empty list when "
                    "no action items exist. The detailed_summary must preserve every "
                    "named person, product or device example, numeric detail, decision, "
                    "constraint, and explicit next step present in the source, even when "
                    "those details seem secondary. Do not silently merge or omit examples."
                ),
            },
            {"role": "user", "content": source},
        ],
        text_format=AnalysisPayload,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise HTTPException(status_code=502, detail="AI analysis returned no structured data.")

    return RecordingAnalysis(
        transcript=transcript,
        transcript_chunks=chunks,
        **parsed.model_dump(),
    )


def semantic_search(
    request: SearchRequest,
    *,
    owner_id: str = "local",
) -> SearchReply:
    query = request.query.strip()
    if not query or not request.recordings:
        return SearchReply(results=[])

    rag = sync_rag(request.recordings, owner_id=owner_id)
    sources = rag.search(
        query,
        top_k=max(request.top_k * 3, request.top_k),
        owner_id=owner_id,
    )
    best_by_recording: dict[str, SearchResult] = {}

    for source in sources:
        existing = best_by_recording.get(source.recording_id)
        if existing is None or source.score > existing.score:
            best_by_recording[source.recording_id] = SearchResult(
                recording_id=source.recording_id,
                title=source.recording_title,
                matched_text=source.text,
                score=source.score,
                chunk_id=source.chunk_id,
                start_time=source.start_time,
                end_time=source.end_time,
            )

    ranked = sorted(
        best_by_recording.values(),
        key=lambda result: result.score,
        reverse=True,
    )
    return SearchReply(results=ranked[: request.top_k])


def recording_by_id(
    recordings: list[AgentRecording],
    recording_id: str,
) -> Optional[AgentRecording]:
    return next((item for item in recordings if item.id == recording_id), None)


def suggest_title(client: OpenAI, recording: AgentRecording) -> str:
    if recording.suggested_title:
        return recording.suggested_title
    source = recording.summary or recording.transcript or recording.title
    response = client.responses.parse(
        model=ANALYSIS_MODEL,
        instructions="Suggest one concise, specific voice memo title. Do not use quotes.",
        input=source[:12_000],
        text_format=TitleSuggestion,
    )
    return response.output_parsed.title if response.output_parsed else recording.title


def suggest_tags(client: OpenAI, recording: AgentRecording) -> list[str]:
    if recording.suggested_tags:
        return recording.suggested_tags
    source = recording.summary or recording.transcript or recording.title
    response = client.responses.parse(
        model=ANALYSIS_MODEL,
        instructions="Suggest 2 to 5 short tags grounded in this voice memo.",
        input=source[:12_000],
        text_format=TagSuggestion,
    )
    return response.output_parsed.tags if response.output_parsed else []


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_recordings",
        "description": "Find recordings and timestamped transcript passages relevant to a query.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query", "top_k"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "answer_recording_question",
        "description": (
            "Answer a question by retrieving timestamped evidence from one or more "
            "recordings. Use this for synthesis, comparisons, and factual questions "
            "about transcript content. Pass an empty recording_ids list to search all."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                "recording_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["query", "top_k", "recording_ids"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_recording",
        "description": "Get one recording's metadata and cached AI analysis.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"recording_id": {"type": "string"}},
            "required": ["recording_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_latest_recording",
        "description": "Get the most recently created recording.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "summarize_recording",
        "description": "Return the cached summary, key points, and action items for one recording.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"recording_id": {"type": "string"}},
            "required": ["recording_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_transcript",
        "description": "Return the cached transcript for one recording.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"recording_id": {"type": "string"}},
            "required": ["recording_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "extract_action_items",
        "description": "Return cached action items extracted from one recording.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"recording_id": {"type": "string"}},
            "required": ["recording_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "suggest_title",
        "description": "Suggest a better title grounded in a recording's content.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"recording_id": {"type": "string"}},
            "required": ["recording_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "suggest_tags",
        "description": "Suggest useful tags grounded in a recording's content.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"recording_id": {"type": "string"}},
            "required": ["recording_id"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "rename_recording",
        "description": "Prepare a recording rename. The app must ask for confirmation before applying it.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "recording_id": {"type": "string"},
                "new_title": {"type": "string"},
            },
            "required": ["recording_id", "new_title"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "tag_recording",
        "description": "Prepare tag changes. The app must ask for confirmation before applying them.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "recording_id": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["recording_id", "tags"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "delete_recording",
        "description": "Prepare deletion of one recording. Never execute without confirmation.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"recording_id": {"type": "string"}},
            "required": ["recording_id"],
            "additionalProperties": False,
        },
    },
]


def execute_read_tool(
    client: OpenAI,
    name: str,
    arguments: dict[str, Any],
    recordings: list[AgentRecording],
    state: SessionState,
    owner_id: str,
) -> tuple[dict[str, Any], list[str], list[AgentSource]]:
    if name == "search_recordings":
        reply = semantic_search(
            SearchRequest(
                query=arguments["query"],
                recordings=recordings,
                top_k=arguments["top_k"],
            ),
            owner_id=owner_id,
        )
        ids = [result.recording_id for result in reply.results]
        state.last_search_results = ids
        state.current_recording_id = ids[0] if ids else None
        sources = [
            AgentSource(
                recording_id=result.recording_id,
                recording_title=result.title,
                chunk_id=result.chunk_id or "chunk",
                excerpt=result.matched_text,
                start_time=result.start_time,
                end_time=result.end_time,
                score=result.score,
            )
            for result in reply.results
        ]
        return reply.model_dump(), ids, sources

    if name == "answer_recording_question":
        available_ids = {recording.id for recording in recordings}
        requested_ids = [
            recording_id
            for recording_id in arguments["recording_ids"]
            if recording_id in available_ids
        ]
        rag = sync_rag(recordings, owner_id=owner_id)
        answer = rag.answer(
            arguments["query"],
            top_k=arguments["top_k"],
            recording_ids=requested_ids or None,
            owner_id=owner_id,
        )
        sources = [agent_source(source) for source in answer.sources]
        ids = list(dict.fromkeys(source.recording_id for source in sources))
        state.last_search_results = ids
        state.current_recording_id = ids[0] if ids else state.current_recording_id
        return {
            "answer": answer.answer,
            "sources": [
                {
                    "label": f"S{index}",
                    **source.model_dump(),
                }
                for index, source in enumerate(sources, start=1)
            ],
        }, ids, sources

    if name == "get_latest_recording":
        if not recordings:
            return {"error": "No recordings exist."}, [], []
        latest = max(recordings, key=lambda item: item.created_at)
        state.current_recording_id = latest.id
        return {"recording_id": latest.id, "title": latest.title}, [latest.id], []

    recording_id = arguments.get("recording_id", "")
    recording = recording_by_id(recordings, recording_id)
    if recording is None:
        return {"error": "Recording not found."}, [], []
    state.current_recording_id = recording.id

    if name == "get_recording":
        return {
            "id": recording.id,
            "title": recording.title,
            "created_at": recording.created_at.isoformat(),
            "duration": recording.duration,
            "tags": recording.tags,
            "category": recording.category,
            "has_transcript": bool(recording.transcript),
            "has_summary": bool(recording.summary),
        }, [recording.id], []
    if name == "summarize_recording":
        if not recording.summary:
            return {
                "error": "No cached summary is available. Ask the user to generate analysis first."
            }, [recording.id], []
        return {
            "recording_id": recording.id,
            "summary": recording.summary,
            "key_points": recording.key_points,
            "action_items": [item.model_dump() for item in recording.action_items],
        }, [recording.id], []
    if name == "get_transcript":
        if not recording.transcript:
            return {"error": "No cached transcript is available."}, [recording.id], []
        return {
            "recording_id": recording.id,
            "transcript": recording.transcript,
        }, [recording.id], []
    if name == "extract_action_items":
        return {
            "recording_id": recording.id,
            "action_items": [item.model_dump() for item in recording.action_items],
        }, [recording.id], []
    if name == "suggest_title":
        return {"suggested_title": suggest_title(client, recording)}, [recording.id], []
    if name == "suggest_tags":
        return {"tags": suggest_tags(client, recording)}, [recording.id], []
    return {"error": f"Unsupported tool: {name}"}, [], []


def prepare_pending_action(
    name: str,
    arguments: dict[str, Any],
    recordings: list[AgentRecording],
) -> Optional[PendingAction]:
    recording_id = arguments.get("recording_id", "")
    recording = recording_by_id(recordings, recording_id)
    if recording is None:
        return None
    if name == "rename_recording":
        return PendingAction(
            type=name,
            recording_id=recording.id,
            recording_title=recording.title,
            new_title=arguments["new_title"].strip(),
        )
    if name == "tag_recording":
        return PendingAction(
            type=name,
            recording_id=recording.id,
            recording_title=recording.title,
            tags=arguments["tags"],
        )
    if name == "delete_recording":
        return PendingAction(
            type=name,
            recording_id=recording.id,
            recording_title=recording.title,
        )
    return None


def confirmation_message(action: PendingAction) -> str:
    if action.type == "rename_recording":
        return (
            f'I am ready to rename "{action.recording_title}" to '
            f'"{action.new_title}". Do you want to continue?'
        )
    if action.type == "tag_recording":
        return (
            f'I am ready to add these tags to "{action.recording_title}": '
            f'{", ".join(action.tags or [])}. Do you want to continue?'
        )
    return (
        f'I am ready to delete "{action.recording_title}". '
        "This action cannot be undone. Do you want to continue?"
    )


def run_agent(request: AgentAskRequest, *, owner_id: str = "local") -> AgentReply:
    client = get_client()
    state = load_session_state(request.session_id)
    if request.current_recording_id:
        state.current_recording_id = request.current_recording_id

    state_context = {
        "current_recording_id": state.current_recording_id,
        "last_search_results_in_order": state.last_search_results,
        "recording_count": len(request.recordings),
    }
    input_items: list[Any] = [
        {"role": turn.role, "content": turn.content} for turn in state.history[-10:]
    ]
    input_items.append({"role": "user", "content": request.query})
    last_recording_ids: list[str] = []
    last_sources: list[AgentSource] = []

    instructions = (
        "You are the AI agent inside a voice memo app. Use tools for every claim "
        "about the user's recordings. Never invent recordings or transcript content. "
        "Use answer_recording_question for questions that require transcript evidence, "
        "cross-recording synthesis, or comparisons. Preserve its [S1] citation labels "
        "in the final response. Use search_recordings when the user only wants to locate "
        "relevant recordings. Pass the user's wording to retrieval tools verbatim unless "
        "a conversational pronoun must be resolved from session state. "
        "Cached summaries must be reused. For rename, tag, or delete requests, call "
        "the corresponding modification tool; the application will request confirmation. "
        "Resolve phrases such as 'the second one', 'it', and 'this recording' using "
        f"this session state: {json.dumps(state_context, ensure_ascii=False)}. "
        "Answer in the user's language and keep the final response concise."
    )

    for _ in range(8):
        response = client.responses.create(
            model=AGENT_MODEL,
            instructions=instructions,
            tools=TOOLS,
            input=input_items,
        )
        input_items.extend(response.output)
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            message = response.output_text.strip() or "I couldn't complete this request."
            state.history.extend(
                [
                    SessionTurn(role="user", content=request.query),
                    SessionTurn(role="assistant", content=message),
                ]
            )
            save_session_state(request.session_id, state)
            return AgentReply(
                message=message,
                recording_ids=last_recording_ids,
                sources=last_sources,
            )

        for call in function_calls:
            arguments = json.loads(call.arguments)
            if call.name in {
                "rename_recording",
                "tag_recording",
                "delete_recording",
            }:
                action = prepare_pending_action(call.name, arguments, request.recordings)
                if action is None:
                    output = {"error": "Recording not found."}
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(output),
                        }
                    )
                    continue
                state.pending_action = action
                state.current_recording_id = action.recording_id
                confirmation = confirmation_message(action)
                state.history.extend(
                    [
                        SessionTurn(role="user", content=request.query),
                        SessionTurn(role="assistant", content=confirmation),
                    ]
                )
                save_session_state(request.session_id, state)
                return AgentReply(
                    message=confirmation,
                    recording_ids=[action.recording_id],
                    requires_confirmation=True,
                    pending_action=action,
                )

            output, recording_ids, sources = execute_read_tool(
                client,
                call.name,
                arguments,
                request.recordings,
                state,
                owner_id,
            )
            if recording_ids:
                last_recording_ids = recording_ids
            if sources:
                last_sources = sources
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(output, ensure_ascii=False, default=str),
                }
            )

    raise HTTPException(status_code=502, detail="The agent exceeded its tool-call limit.")


@app.get("/health")
def health() -> dict[str, Any]:
    authentication_mode = (
        "oidc_jwt" if JWT_JWKS_URL else "shared_token" if API_BEARER_TOKEN else "disabled"
    )
    return {
        "status": "ok",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "models": {
            "agent": AGENT_MODEL,
            "analysis": ANALYSIS_MODEL,
            "transcription": TRANSCRIPTION_MODEL,
            "embedding": EMBEDDING_MODEL,
        },
        "transcription_prompt_configured": bool(TRANSCRIPTION_PROMPT),
        "timestamp_source": "word" if TIMESTAMP_MODEL else "estimated",
        "analysis_queue": "redis-rq" if ANALYSIS_QUEUE_ENABLED else "synchronous",
        "authentication_required": bool(API_BEARER_TOKEN or JWT_JWKS_URL),
        "authentication_mode": authentication_mode,
        "agent": {
            "session_store": SESSION_STORE_PROVIDER,
            "confirmation_required_for_writes": True,
        },
        "rag": {
            "framework": "LangChain",
            "vector_store": VECTOR_STORE_PROVIDER,
            "retrieval": "hybrid" if HYBRID_SEARCH_ENABLED else "vector",
            "reranking": RAG_RERANK_ENABLED,
            "minimum_candidate_score": SEMANTIC_MIN_SCORE,
            "tenant_isolated": True,
            "persistent": True,
        },
    }


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/rag/status")
def rag_status(request: Request) -> dict[str, Any]:
    rag = get_rag_service()
    owner_id = getattr(request.state, "user_id", "local")
    return {
        "status": "ready",
        "vector_store": VECTOR_STORE_PROVIDER,
        **rag.status(owner_id=owner_id),
    }


@app.post(
    "/recordings/{recording_id}/analysis-jobs",
    response_model=AnalysisJobSubmission,
    status_code=202,
)
async def submit_analysis_job(
    recording_id: str,
    request: Request,
    audio: UploadFile = File(...),
    duration: float = Form(...),
    force_regenerate: bool = Form(False),
) -> AnalysisJobSubmission:
    del force_regenerate
    if not ANALYSIS_QUEUE_ENABLED:
        raise HTTPException(status_code=503, detail="The analysis queue is disabled.")

    suffix = validate_audio_upload(audio, duration)
    jobs_directory = Path(__file__).parent / "data" / "jobs"
    jobs_directory.mkdir(parents=True, exist_ok=True)
    recording_digest = hashlib.sha256(recording_id.encode("utf-8")).hexdigest()[:16]
    audio_path = jobs_directory / f"{recording_digest}-{uuid.uuid4()}{suffix}"
    owner_id = getattr(request.state, "user_id", "local")
    try:
        with audio_path.open("wb") as destination:
            copy_upload_with_limit(audio.file, destination, MAX_AUDIO_UPLOAD_BYTES)
        try:
            from .job_queue import enqueue_analysis
        except ImportError:
            from job_queue import enqueue_analysis
        job = enqueue_analysis(
            recording_id=recording_id,
            owner_id=owner_id,
            audio_path=audio_path,
            duration=duration,
        )
        return AnalysisJobSubmission(job_id=job.id, status="queued")
    except HTTPException:
        audio_path.unlink(missing_ok=True)
        raise
    except Exception as error:
        audio_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=503,
            detail="The analysis queue is unavailable.",
        ) from error
    finally:
        await audio.close()


@app.get("/analysis-jobs/{job_id}", response_model=AnalysisJobStatus)
def analysis_job_status(job_id: str, request: Request) -> AnalysisJobStatus:
    try:
        from rq.exceptions import NoSuchJobError

        try:
            from .job_queue import fetch_job
        except ImportError:
            from job_queue import fetch_job
        job = fetch_job(job_id)
        owner_id = getattr(request.state, "user_id", "local")
        if not secrets.compare_digest(str(job.meta.get("owner_id", "")), owner_id):
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        raw_status = job.get_status(refresh=True)
        status = getattr(raw_status, "value", str(raw_status))
    except NoSuchJobError as error:
        raise HTTPException(status_code=404, detail="Analysis job not found.") from error
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail="The analysis queue is unavailable.",
        ) from error

    if status == "finished":
        return AnalysisJobStatus(
            job_id=job_id,
            status=status,
            result=RecordingAnalysis.model_validate(job.result),
        )
    if status in {"failed", "stopped", "canceled"}:
        return AnalysisJobStatus(
            job_id=job_id,
            status=status,
            error="Audio analysis failed. Check the backend worker logs.",
        )
    return AnalysisJobStatus(job_id=job_id, status=status)


@app.post("/recordings/{recording_id}/analyze", response_model=RecordingAnalysis)
async def analyze_recording(
    recording_id: str,
    audio: UploadFile = File(...),
    duration: float = Form(...),
    force_regenerate: bool = Form(False),
) -> RecordingAnalysis:
    del recording_id, force_regenerate
    suffix = validate_audio_upload(audio, duration)
    temp_path = Path(tempfile.gettempdir()) / f"voice-memo-{uuid.uuid4()}{suffix}"
    try:
        with temp_path.open("wb") as destination:
            copy_upload_with_limit(audio.file, destination, MAX_AUDIO_UPLOAD_BYTES)
        return analyze_audio(temp_path, duration)
    finally:
        temp_path.unlink(missing_ok=True)
        await audio.close()


@app.post("/search", response_model=SearchReply)
def search(request: SearchRequest, http_request: Request) -> SearchReply:
    subject = getattr(http_request.state, "user_id", "local")
    return semantic_search(request, owner_id=subject)


@app.post("/agent/ask", response_model=AgentReply)
def ask_agent(request: AgentAskRequest, http_request: Request) -> AgentReply:
    subject = getattr(http_request.state, "user_id", "local")
    scoped_session_id = f"{subject}:{request.session_id}"
    return run_agent(
        request.model_copy(update={"session_id": scoped_session_id}),
        owner_id=subject,
    )


@app.post("/agent/confirm", response_model=ConfirmReply)
def confirm_agent_action(request: ConfirmRequest, http_request: Request) -> ConfirmReply:
    subject = getattr(http_request.state, "user_id", "local")
    session_id = f"{subject}:{request.session_id}"
    state = load_session_state(session_id)
    if state.pending_action is None:
        return ConfirmReply(success=False, message="There is no pending action to confirm.")
    action = state.pending_action
    state.pending_action = None

    if not request.confirm:
        message = "The action was cancelled."
        state.history.append(SessionTurn(role="assistant", content=message))
        save_session_state(session_id, state)
        return ConfirmReply(success=True, message=message)
    message = "The action was confirmed and is ready to be applied."
    state.history.append(SessionTurn(role="assistant", content=message))
    save_session_state(session_id, state)
    return ConfirmReply(
        success=True,
        message=message,
        action=action,
    )
