# Real-audio evaluation

This benchmark uses three 45-second excerpts from the AMI Meeting Corpus
meeting `ES2002a`. AMI contains real multi-speaker meetings with human word-level
annotations and is distributed under CC BY 4.0. Audio, annotations, generated
clips, transcripts, and results are stored under `backend/data/eval`, which is
excluded from Git.

From the `backend` directory:

```bash
.venv/bin/python evaluation/download_ami_sample.py
.venv/bin/python evaluation/evaluate_ami.py
```

Create a deterministic 10 dB white-noise set and run the same metrics with:

```bash
.venv/bin/python evaluation/create_noisy_variants.py
.venv/bin/python evaluation/evaluate_ami.py \
  --manifest data/eval/ami/manifest-noise-10db.json
```

Additional languages can use the same manifest schema. Each sample needs an
audio path, reference transcript, retrieval query, and expected facts.

## Multilingual transcription

The multilingual smoke benchmark streams one real English and one real Mandarin
sample from Google FLEURS without downloading full language archives. FLEURS is
licensed CC BY 4.0. Install the evaluation-only dependency, download the two
samples, and calculate English WER and Mandarin character error rate (CER):

```bash
.venv/bin/python -m pip install -r evaluation/requirements.txt
env -u HF_TOKEN -u HUGGING_FACE_HUB_TOKEN \
  .venv/bin/python evaluation/download_fleurs_sample.py
.venv/bin/python evaluation/evaluate_multilingual.py
```

The token variables are unset because a stale local Hugging Face token can make
an otherwise public dataset return `401`. Audio and results remain under the
Git-ignored `backend/data/eval/fleurs` directory.

The evaluation calls the configured OpenAI transcription, Responses, and
embedding APIs. It reports word error rate, literal expected-fact coverage in
the detailed summaries, retrieval top-1 accuracy, mean reciprocal rank, scores,
timestamped sources, and whether the highest-ranked passage itself contains the
expected evidence. Recording-level accuracy can be perfect while passage-level
coverage is lower, which matters for direct timestamp jumps. Literal fact
coverage is a conservative smoke check, not a substitute for human review or a
model-based factuality evaluation.

Source: <https://groups.inf.ed.ac.uk/ami/corpus/>

FLEURS source: <https://huggingface.co/datasets/google/fleurs>

## Latest local run

Using the configured OpenAI models on 2026-11-03:

| Condition | Transcription | Retrieval Top-1 | MRR | Top-passage facts |
| --- | ---: | ---: | ---: | ---: |
| AMI clean | WER 25.25% | 100% | 100% | 83.33% |
| AMI + 10 dB white noise | WER 36.25% | 100% | 100% | 83.33% |

The FLEURS smoke samples produced English WER 5.3% and Mandarin CER 0.0%.
Both languages used model-supplied timestamps after alignment; CJK text is
aligned by character because it does not reliably contain word separators.

Model outputs are nondeterministic, and this three-clip set is a regression
smoke test rather than a statistically representative production benchmark.
