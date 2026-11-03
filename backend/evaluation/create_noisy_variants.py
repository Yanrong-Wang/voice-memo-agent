"""Create deterministic noisy AMI clips for robustness regression tests."""

from __future__ import annotations

import array
import json
import math
import random
import wave
from pathlib import Path


DATA_DIRECTORY = Path(__file__).parents[1] / "data" / "eval" / "ami"
TARGET_SNR_DB = 10.0


def add_white_noise(source: Path, destination: Path, snr_db: float) -> None:
    randomizer = random.Random(42)
    with wave.open(str(source), "rb") as input_wave:
        parameters = input_wave.getparams()
        if parameters.sampwidth != 2:
            raise ValueError("Noise generation currently expects 16-bit PCM WAV input.")
        samples = array.array("h", input_wave.readframes(parameters.nframes))

    signal_rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    noise_rms = signal_rms / (10 ** (snr_db / 20))
    mixed = array.array(
        "h",
        (
            max(-32_768, min(32_767, round(sample + randomizer.gauss(0, noise_rms))))
            for sample in samples
        ),
    )
    with wave.open(str(destination), "wb") as output_wave:
        output_wave.setparams(parameters)
        output_wave.writeframes(mixed.tobytes())


def main() -> None:
    source_manifest_path = DATA_DIRECTORY / "manifest.json"
    if not source_manifest_path.exists():
        raise SystemExit("Run download_ami_sample.py first.")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    samples = []
    for sample in source_manifest["samples"]:
        source = Path(sample["audio_path"])
        destination = source.with_name(f"{source.stem}-noise-10db.wav")
        add_white_noise(source, destination, TARGET_SNR_DB)
        samples.append(
            {
                **sample,
                "id": f"{sample['id']}-noise-10db",
                "title": f"{sample['title']} (10 dB Noise)",
                "audio_path": str(destination),
            }
        )

    manifest = {
        **source_manifest,
        "dataset": f"{source_manifest['dataset']} with 10 dB synthetic noise",
        "samples": samples,
    }
    output_path = DATA_DIRECTORY / "manifest-noise-10db.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(samples)} noisy clips: {output_path}")


if __name__ == "__main__":
    main()
