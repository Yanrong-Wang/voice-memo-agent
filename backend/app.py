from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

app = FastAPI(
    title="AI Voice Memo Agent",
    version="1.0.0",
    description="Transcription, structured analysis, semantic search, and agent tools.",
)

AGENT_MODEL = os.getenv("OPENAI_AGENT_MODEL", "gpt-5.4-mini")
ANALYSIS_MODEL = os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-5.4-mini")
TRANSCRIPTION_MODEL = os.getenv(
    "OPENAI_TRANSCRIPTION_MODEL",
    "gpt-4o-mini-transcribe",
)
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
SEMANTIC_MIN_SCORE = float(os.getenv("SEMANTIC_MIN_SCORE", "0.20"))


class ActionItem(BaseModel):
    task: str
    priority: Literal["low", "medium", "high"] = "medium"
    source_timestamp: Optional[str] = None


class TranscriptChunk(BaseModel):
    id: str
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


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


class SearchReply(BaseModel):
    results: list[SearchResult]


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
    requires_confirmation: bool = False
    pending_action: Optional[PendingAction] = None


class ConfirmRequest(BaseModel):
    session_id: str
    confirm: bool


class ConfirmReply(BaseModel):
    success: bool
    message: str
    action: Optional[PendingAction] = None


class SessionState(BaseModel):
    last_search_results: list[str] = Field(default_factory=list)
    current_recording_id: Optional[str] = None
    pending_action: Optional[PendingAction] = None


class TitleSuggestion(BaseModel):
    title: str


class TagSuggestion(BaseModel):
    tags: list[str]


_sessions: dict[str, SessionState] = {}
_sessions_lock = Lock()
_embedding_cache: dict[str, list[float]] = {}
_embedding_lock = Lock()


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the AI server.",
        )
    return OpenAI(api_key=api_key)


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
    chunks = split_text(text, size=1_600, overlap=0)
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
    with audio_path.open("rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=TRANSCRIPTION_MODEL,
            file=audio_file,
            response_format="text",
        )
    transcript = (
        transcription
        if isinstance(transcription, str)
        else getattr(transcription, "text", str(transcription))
    ).strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="The recording contains no speech.")

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
                    "by the transcript. Use an empty list when none exist."
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
        transcript_chunks=transcript_chunks(transcript, duration),
        **parsed.model_dump(),
    )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0
    return dot / (left_norm * right_norm)


def embeddings_for(client: OpenAI, texts: list[str]) -> list[list[float]]:
    results: list[Optional[list[float]]] = [None] * len(texts)
    missing_texts: list[str] = []
    missing_indexes: list[int] = []

    with _embedding_lock:
        for index, text in enumerate(texts):
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cached = _embedding_cache.get(key)
            if cached is None:
                missing_texts.append(text)
                missing_indexes.append(index)
            else:
                results[index] = cached

    if missing_texts:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=missing_texts,
            encoding_format="float",
        )
        with _embedding_lock:
            for target_index, item, text in zip(
                missing_indexes,
                response.data,
                missing_texts,
            ):
                vector = item.embedding
                results[target_index] = vector
                key = hashlib.sha256(text.encode("utf-8")).hexdigest()
                _embedding_cache[key] = vector

    return [vector or [] for vector in results]


def semantic_search(request: SearchRequest) -> SearchReply:
    query = request.query.strip()
    if not query or not request.recordings:
        return SearchReply(results=[])

    candidates: list[tuple[AgentRecording, str]] = []
    for recording in request.recordings:
        searchable = recording.transcript or recording.summary or " ".join(
            [recording.title, *recording.tags]
        )
        for chunk in split_text(searchable):
            candidates.append((recording, chunk))
    if not candidates:
        return SearchReply(results=[])

    client = get_client()
    vectors = embeddings_for(client, [query, *[item[1] for item in candidates]])
    query_vector = vectors[0]
    best_by_recording: dict[str, SearchResult] = {}

    for (recording, chunk), vector in zip(candidates, vectors[1:]):
        score = cosine_similarity(query_vector, vector)
        existing = best_by_recording.get(recording.id)
        if existing is None or score > existing.score:
            best_by_recording[recording.id] = SearchResult(
                recording_id=recording.id,
                title=recording.title,
                matched_text=chunk,
                score=round(score, 4),
            )

    ranked = sorted(
        best_by_recording.values(),
        key=lambda result: result.score,
        reverse=True,
    )
    return SearchReply(
        results=[
            result
            for result in ranked
            if result.score >= SEMANTIC_MIN_SCORE
        ][: request.top_k]
    )


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
        "description": "Semantic search over recording transcripts, summaries, titles, and tags.",
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
) -> tuple[dict[str, Any], list[str]]:
    if name == "search_recordings":
        reply = semantic_search(
            SearchRequest(
                query=arguments["query"],
                recordings=recordings,
                top_k=arguments["top_k"],
            )
        )
        ids = [result.recording_id for result in reply.results]
        state.last_search_results = ids
        state.current_recording_id = ids[0] if ids else None
        return reply.model_dump(), ids

    if name == "get_latest_recording":
        if not recordings:
            return {"error": "No recordings exist."}, []
        latest = max(recordings, key=lambda item: item.created_at)
        state.current_recording_id = latest.id
        return {"recording_id": latest.id, "title": latest.title}, [latest.id]

    recording_id = arguments.get("recording_id", "")
    recording = recording_by_id(recordings, recording_id)
    if recording is None:
        return {"error": "Recording not found."}, []
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
        }, [recording.id]
    if name == "summarize_recording":
        if not recording.summary:
            return {
                "error": "No cached summary is available. Ask the user to generate analysis first."
            }, [recording.id]
        return {
            "recording_id": recording.id,
            "summary": recording.summary,
            "key_points": recording.key_points,
            "action_items": [item.model_dump() for item in recording.action_items],
        }, [recording.id]
    if name == "get_transcript":
        if not recording.transcript:
            return {"error": "No cached transcript is available."}, [recording.id]
        return {
            "recording_id": recording.id,
            "transcript": recording.transcript,
        }, [recording.id]
    if name == "extract_action_items":
        return {
            "recording_id": recording.id,
            "action_items": [item.model_dump() for item in recording.action_items],
        }, [recording.id]
    if name == "suggest_title":
        return {"suggested_title": suggest_title(client, recording)}, [recording.id]
    if name == "suggest_tags":
        return {"tags": suggest_tags(client, recording)}, [recording.id]
    return {"error": f"Unsupported tool: {name}"}, []


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
            f'建议将“{action.recording_title}”重命名为“{action.new_title}”。'
            "是否确认？"
        )
    if action.type == "tag_recording":
        return (
            f'建议为“{action.recording_title}”设置标签：'
            f'{", ".join(action.tags or [])}。是否确认？'
        )
    return f'即将删除“{action.recording_title}”。此操作不可撤销，是否确认？'


def run_agent(request: AgentAskRequest) -> AgentReply:
    client = get_client()
    with _sessions_lock:
        state = _sessions.get(request.session_id, SessionState()).model_copy(deep=True)
    if request.current_recording_id:
        state.current_recording_id = request.current_recording_id

    state_context = {
        "current_recording_id": state.current_recording_id,
        "last_search_results_in_order": state.last_search_results,
        "recording_count": len(request.recordings),
    }
    input_items: list[Any] = [{"role": "user", "content": request.query}]
    last_recording_ids: list[str] = []

    instructions = (
        "You are the AI agent inside a voice memo app. Use tools for every claim "
        "about the user's recordings. Never invent recordings or transcript content. "
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
            with _sessions_lock:
                _sessions[request.session_id] = state
            return AgentReply(message=message, recording_ids=last_recording_ids)

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
                with _sessions_lock:
                    _sessions[request.session_id] = state
                return AgentReply(
                    message=confirmation_message(action),
                    recording_ids=[action.recording_id],
                    requires_confirmation=True,
                    pending_action=action,
                )

            output, recording_ids = execute_read_tool(
                client,
                call.name,
                arguments,
                request.recordings,
                state,
            )
            if recording_ids:
                last_recording_ids = recording_ids
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
    return {
        "status": "ok",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "models": {
            "agent": AGENT_MODEL,
            "analysis": ANALYSIS_MODEL,
            "transcription": TRANSCRIPTION_MODEL,
            "embedding": EMBEDDING_MODEL,
        },
    }


@app.post("/recordings/{recording_id}/analyze", response_model=RecordingAnalysis)
async def analyze_recording(
    recording_id: str,
    audio: UploadFile = File(...),
    duration: float = Form(...),
    force_regenerate: bool = Form(False),
) -> RecordingAnalysis:
    del recording_id, force_regenerate
    suffix = Path(audio.filename or "recording.m4a").suffix or ".m4a"
    temp_path = Path(tempfile.gettempdir()) / f"voice-memo-{uuid.uuid4()}{suffix}"
    try:
        with temp_path.open("wb") as destination:
            shutil.copyfileobj(audio.file, destination)
        return analyze_audio(temp_path, duration)
    finally:
        temp_path.unlink(missing_ok=True)
        await audio.close()


@app.post("/search", response_model=SearchReply)
def search(request: SearchRequest) -> SearchReply:
    return semantic_search(request)


@app.post("/agent/ask", response_model=AgentReply)
def ask_agent(request: AgentAskRequest) -> AgentReply:
    return run_agent(request)


@app.post("/agent/confirm", response_model=ConfirmReply)
def confirm_agent_action(request: ConfirmRequest) -> ConfirmReply:
    with _sessions_lock:
        state = _sessions.get(request.session_id)
        if state is None or state.pending_action is None:
            return ConfirmReply(success=False, message="没有等待确认的操作。")
        action = state.pending_action
        state.pending_action = None
        _sessions[request.session_id] = state

    if not request.confirm:
        return ConfirmReply(success=True, message="已取消操作。")
    return ConfirmReply(success=True, message="已确认，正在应用操作。", action=action)
