"""Run transcription, analysis, and retrieval evaluation on AMI meeting clips."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


BACKEND_DIRECTORY = Path(__file__).parents[1]
DATA_DIRECTORY = BACKEND_DIRECTORY / "data" / "eval" / "ami"
sys.path.insert(0, str(BACKEND_DIRECTORY))

# Keep evaluation vectors separate from the app index and avoid domain hints that
# would leak unrelated technical vocabulary into this general meeting benchmark.
os.environ["RAG_COLLECTION_NAME"] = "voice_memo_eval_chunks"
os.environ["OPENAI_TRANSCRIPTION_PROMPT"] = ""

from app import (  # noqa: E402
    AgentRecording,
    SearchRequest,
    analyze_audio,
    semantic_search,
)


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_word in reference:
        current = [previous[0] + 1]
        for index, hypothesis_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[index] + 1,
                    previous[index - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> tuple[int, int, float]:
    reference_words = normalized_words(reference)
    errors = edit_distance(reference_words, normalized_words(hypothesis))
    return errors, len(reference_words), errors / max(1, len(reference_words))


def fact_coverage(text: str, expected_facts: list[str]) -> dict[str, bool]:
    normalized = " ".join(normalized_words(text))
    return {
        fact: " ".join(normalized_words(fact)) in normalized
        for fact in expected_facts
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DATA_DIRECTORY / "manifest.json",
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    manifest_path = arguments.manifest
    if not manifest_path.exists():
        raise SystemExit(
            "Run evaluation/download_ami_sample.py before starting the evaluation."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    recordings: list[AgentRecording] = []
    sample_results = []
    total_errors = 0
    total_words = 0
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for index, sample in enumerate(manifest["samples"]):
        print(f"Analyzing {sample['title']}...", flush=True)
        analysis = analyze_audio(Path(sample["audio_path"]), float(sample["duration"]))
        errors, words, wer = word_error_rate(
            sample["reference_transcript"],
            analysis.transcript,
        )
        total_errors += errors
        total_words += words
        coverage = fact_coverage(analysis.detailed_summary, sample["expected_facts"])
        sample_results.append(
            {
                "id": sample["id"],
                "title": sample["title"],
                "reference_transcript": sample["reference_transcript"],
                "transcript": analysis.transcript,
                "word_errors": errors,
                "reference_words": words,
                "wer": round(wer, 4),
                "summary": analysis.summary,
                "detailed_summary": analysis.detailed_summary,
                "expected_fact_coverage": coverage,
                "transcript_chunks": [
                    chunk.model_dump() for chunk in analysis.transcript_chunks
                ],
            }
        )
        recordings.append(
            AgentRecording(
                id=sample["id"],
                title=sample["title"],
                created_at=base_time + timedelta(minutes=index),
                duration=float(sample["duration"]),
                transcript=analysis.transcript,
                transcript_chunks=analysis.transcript_chunks,
                summary=analysis.summary,
                key_points=analysis.key_points,
                action_items=analysis.action_items,
                suggested_title=analysis.suggested_title,
                suggested_tags=analysis.suggested_tags,
                category=analysis.category,
            )
        )

    retrieval_results = []
    reciprocal_rank_sum = 0.0
    top_one_correct = 0
    passage_facts_found = 0
    passage_facts_total = 0
    for sample in manifest["samples"]:
        reply = semantic_search(
            SearchRequest(query=sample["query"], recordings=recordings, top_k=3)
        )
        ranked_ids = [result.recording_id for result in reply.results]
        rank = ranked_ids.index(sample["id"]) + 1 if sample["id"] in ranked_ids else None
        if rank == 1:
            top_one_correct += 1
        if rank:
            reciprocal_rank_sum += 1 / rank
        top_result_text = reply.results[0].matched_text if reply.results else ""
        passage_coverage = fact_coverage(
            top_result_text,
            sample["expected_retrieval_facts"],
        )
        passage_facts_found += sum(passage_coverage.values())
        passage_facts_total += len(passage_coverage)
        retrieval_results.append(
            {
                "query": sample["query"],
                "expected_recording_id": sample["id"],
                "rank": rank,
                "top_passage_fact_coverage": passage_coverage,
                "results": [result.model_dump() for result in reply.results],
            }
        )

    result = {
        "dataset": manifest["dataset"],
        "meeting": manifest["meeting"],
        "samples": sample_results,
        "aggregate": {
            "word_errors": total_errors,
            "reference_words": total_words,
            "wer": round(total_errors / max(1, total_words), 4),
            "retrieval_top_1_accuracy": round(
                top_one_correct / len(manifest["samples"]), 4
            ),
            "retrieval_mrr": round(
                reciprocal_rank_sum / len(manifest["samples"]), 4
            ),
            "top_passage_fact_coverage": round(
                passage_facts_found / max(1, passage_facts_total), 4
            ),
        },
        "retrieval": retrieval_results,
    }
    output_path = arguments.output or manifest_path.with_name(
        f"results-{manifest_path.stem.removeprefix('manifest').strip('-') or 'clean'}.json"
    )
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nEvaluation summary")
    print(json.dumps(result["aggregate"], indent=2))
    for sample in sample_results:
        covered = sum(sample["expected_fact_coverage"].values())
        total = len(sample["expected_fact_coverage"])
        print(
            f"- {sample['title']}: WER {sample['wer']:.1%}, "
            f"literal summary facts {covered}/{total}"
        )
    for retrieval in retrieval_results:
        scores = [
            f"{item['recording_id']}={item['score']:.3f}"
            for item in retrieval["results"]
        ]
        print(f"- query rank {retrieval['rank']}: {', '.join(scores)}")
    print(f"Full results: {output_path}")


if __name__ == "__main__":
    main()
