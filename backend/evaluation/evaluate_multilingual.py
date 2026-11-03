"""Measure English WER and Mandarin character error rate on FLEURS samples."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


BACKEND_DIRECTORY = Path(__file__).parents[1]
DATA_DIRECTORY = BACKEND_DIRECTORY / "data" / "eval" / "fleurs"
sys.path.insert(0, str(BACKEND_DIRECTORY))

from app import get_client, transcribe_audio  # noqa: E402


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_unit in reference:
        current = [previous[0] + 1]
        for index, hypothesis_unit in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[index] + 1,
                    previous[index - 1] + (reference_unit != hypothesis_unit),
                )
            )
        previous = current
    return previous[-1]


def english_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.casefold())


def mandarin_characters(text: str) -> list[str]:
    return re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff0-9a-z]", text.casefold())


def main() -> None:
    manifest_path = DATA_DIRECTORY / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("Run evaluation/download_fleurs_sample.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    client = get_client()
    results = []
    for sample in manifest["samples"]:
        transcript, chunks = transcribe_audio(
            client,
            Path(sample["audio_path"]),
            float(sample["duration"]),
        )
        is_mandarin = sample["id"].startswith("cmn_")
        reference_units = (
            mandarin_characters(sample["reference_transcript"])
            if is_mandarin
            else english_words(sample["reference_transcript"])
        )
        hypothesis_units = (
            mandarin_characters(transcript) if is_mandarin else english_words(transcript)
        )
        errors = edit_distance(reference_units, hypothesis_units)
        results.append(
            {
                **sample,
                "transcript": transcript,
                "metric": "cer" if is_mandarin else "wer",
                "errors": errors,
                "reference_units": len(reference_units),
                "error_rate": round(errors / max(1, len(reference_units)), 4),
                "timing_sources": sorted({chunk.timing_source for chunk in chunks}),
            }
        )

    output = {"dataset": manifest["dataset"], "samples": results}
    output_path = DATA_DIRECTORY / "results.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for result in results:
        print(
            f"{result['language']}: {result['metric'].upper()} "
            f"{result['error_rate']:.1%}, timing={result['timing_sources']}"
        )
    print(f"Full results: {output_path}")


if __name__ == "__main__":
    main()
