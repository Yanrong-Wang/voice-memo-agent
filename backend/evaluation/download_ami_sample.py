"""Download and prepare a small, licensed AMI meeting evaluation set."""

from __future__ import annotations

import json
import urllib.request
import wave
import zipfile
from pathlib import Path
from xml.etree import ElementTree


DATA_DIRECTORY = Path(__file__).parents[1] / "data" / "eval" / "ami"
AUDIO_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/"
    "ES2002a/audio/ES2002a.Mix-Headset.wav"
)
ANNOTATIONS_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/"
    "ami_public_manual_1.6.2.zip"
)

WINDOWS = [
    {
        "id": "ami-kickoff",
        "title": "Remote Control Project Kickoff",
        "start": 55.0,
        "end": 100.0,
        "query": "Who is the project manager and what is the team designing?",
        "expected_retrieval_facts": ["project manager", "remote control"],
        "expected_facts": [
            "Laura",
            "project manager",
            "David",
            "Andrew",
            "Craig",
            "remote control",
        ],
    },
    {
        "id": "ami-design-brief",
        "title": "Remote Control Design Brief",
        "start": 108.0,
        "end": 153.0,
        "query": "What qualities should the new remote control have?",
        "expected_retrieval_facts": ["original", "trendy", "user friendly"],
        "expected_facts": ["original", "trendy", "user friendly"],
    },
    {
        "id": "ami-feature-ideas",
        "title": "Remote Control Feature Ideas",
        "start": 800.0,
        "end": 845.0,
        "query": "What device was used as an example of combining many functions?",
        "expected_retrieval_facts": ["Palm Pilots"],
        "expected_facts": ["Palm Pilots", "cameras", "telephones"],
    },
]


def download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "VoiceMemoAgentEval/1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)


def annotation_words(archive_path: Path) -> list[dict[str, object]]:
    words: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for speaker in "ABCD":
            path = f"words/ES2002a.{speaker}.words.xml"
            root = ElementTree.fromstring(archive.read(path))
            for element in root:
                if not element.tag.endswith("w") or not element.text:
                    continue
                try:
                    start = float(element.attrib["starttime"])
                    end = float(element.attrib["endtime"])
                except (KeyError, ValueError):
                    continue
                words.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker": speaker,
                        "text": element.text,
                        "punctuation": element.attrib.get("punc") == "true",
                    }
                )
    return sorted(words, key=lambda word: (word["start"], word["speaker"]))


def transcript_for_window(
    words: list[dict[str, object]],
    start: float,
    end: float,
) -> str:
    tokens: list[str] = []
    for word in words:
        if float(word["start"]) < start or float(word["start"]) >= end:
            continue
        token = str(word["text"])
        if word["punctuation"] and tokens:
            tokens[-1] += token
        else:
            tokens.append(token)
    return " ".join(tokens)


def crop_wave(source: Path, destination: Path, start: float, end: float) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    with wave.open(str(source), "rb") as input_wave:
        frame_rate = input_wave.getframerate()
        input_wave.setpos(int(start * frame_rate))
        frames = input_wave.readframes(int((end - start) * frame_rate))
        parameters = input_wave.getparams()
    with wave.open(str(destination), "wb") as output_wave:
        output_wave.setparams(parameters)
        output_wave.writeframes(frames)


def main() -> None:
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    source_audio = DATA_DIRECTORY / "ES2002a.Mix-Headset.wav"
    annotations = DATA_DIRECTORY / "ami_public_manual_1.6.2.zip"
    download(AUDIO_URL, source_audio)
    download(ANNOTATIONS_URL, annotations)

    words = annotation_words(annotations)
    samples = []
    for window in WINDOWS:
        clip_path = DATA_DIRECTORY / f"{window['id']}.wav"
        crop_wave(source_audio, clip_path, float(window["start"]), float(window["end"]))
        samples.append(
            {
                **window,
                "duration": float(window["end"]) - float(window["start"]),
                "audio_path": str(clip_path),
                "reference_transcript": transcript_for_window(
                    words,
                    float(window["start"]),
                    float(window["end"]),
                ),
            }
        )

    manifest = {
        "dataset": "AMI Meeting Corpus",
        "meeting": "ES2002a",
        "license": "CC BY 4.0",
        "source": "https://groups.inf.ed.ac.uk/ami/corpus/",
        "audio_url": AUDIO_URL,
        "annotations_url": ANNOTATIONS_URL,
        "samples": samples,
    }
    manifest_path = DATA_DIRECTORY / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(samples)} clips in {DATA_DIRECTORY}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
