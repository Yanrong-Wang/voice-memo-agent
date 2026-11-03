"""Download one English and one Mandarin FLEURS sample without full archives."""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

from datasets import Audio, load_dataset


DATA_DIRECTORY = Path(__file__).parents[1] / "data" / "eval" / "fleurs"
CONFIGS = (
    ("en_us", "English (United States)"),
    ("cmn_hans_cn", "Mandarin Chinese (Simplified)"),
)


def wav_duration(data: bytes) -> float:
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("FLEURS audio is not a RIFF/WAVE file.")
    offset = 12
    byte_rate = None
    data_size = None
    while offset + 8 <= len(data):
        chunk_id = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        payload_offset = offset + 8
        if chunk_id == b"fmt " and chunk_size >= 12:
            byte_rate = struct.unpack_from("<I", data, payload_offset + 8)[0]
        elif chunk_id == b"data":
            data_size = chunk_size
            break
        offset = payload_offset + chunk_size + (chunk_size % 2)
    if not byte_rate or data_size is None:
        raise ValueError("FLEURS WAV metadata is incomplete.")
    return data_size / byte_rate


def main() -> None:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    samples = []
    for config, language in CONFIGS:
        print(f"Streaming one {language} FLEURS sample...", flush=True)
        dataset = load_dataset(
            "google/fleurs",
            config,
            split="test",
            streaming=True,
            token=False,
        ).cast_column("audio", Audio(decode=False))
        row = next(iter(dataset))
        audio = row["audio"]
        audio_bytes = audio.get("bytes")
        if not audio_bytes:
            raise RuntimeError(f"FLEURS returned no audio bytes for {config}.")
        audio_path = DATA_DIRECTORY / f"{config}.wav"
        audio_path.write_bytes(audio_bytes)
        samples.append(
            {
                "id": config,
                "language": language,
                "audio_path": str(audio_path.resolve()),
                "duration": round(wav_duration(audio_bytes), 3),
                "reference_transcript": row["transcription"],
            }
        )

    manifest = {
        "dataset": "Google FLEURS",
        "license": "CC BY 4.0",
        "source": "https://huggingface.co/datasets/google/fleurs",
        "samples": samples,
    }
    manifest_path = DATA_DIRECTORY / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Prepared {len(samples)} samples: {manifest_path}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # PyArrow 21 can deadlock while joining its global thread pool on macOS.
    os._exit(0)
