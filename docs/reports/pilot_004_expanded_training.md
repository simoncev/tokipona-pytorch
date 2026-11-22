# Instruction-tuning learning report 04: broader examples and missing evidence

Date: 22 November 2026. Run: `runs/smollm2-expanded-001`.

## Design fixed before generation

We expanded the supervised dataset and froze a source-based review rubric before producing this experiment's model answers. The rubric evaluates support, coverage of the main ideas, unsupported claims and acknowledgment of missing evidence. It is recorded in `docs/evaluation_rubric_v1.md`, with its checksum in the experiment protocol and each dataset row.

| Split | Examples | Answerable | Unanswerable | Contributors | Distinct excerpts |
|---|---:|---:|---:|---:|---:|
| Training | 50 | 40 | 10 | 10 | 10 |
| Validation | 12 | 8 | 4 | 4 | 4 |

Each training excerpt has four supported questions and one question that cannot be answered from that excerpt. Each validation excerpt has two supported questions and one unsupported question. Questions sharing a passage are correlated; 62 examples do not represent 62 independent pieces of source evidence. The validation set also has a higher proportion of unanswerable questions, so answerable and unanswerable performance are reported separately.

Training contributors: Britt Barak, Ellen Shapiro, Antonio Leiva, Sarah Olson, Paco Estévez García, Corey Leigh Latislaw, Zarah Dominguez, Juhani Lehtimäki, Hadi Hariri and John Sundell. Validation contributors: Lara Martin, Erik Hellman, Eduardo Castelló Ferrer and Huyen Tue Dao. The original source-group assignments are preserved. No final-test interview was selected for examples, training, generation or scoring.

The supplied passage must support the answer. For a missing-evidence example, the target explicitly acknowledges the gap instead of inventing a recommendation. These are supervised target answers, not rewards.

## Training and comparison

The experiment starts from the original local SmolLM2-360M-Instruct base, not an earlier adapter. It runs 100 optimizer updates with batch size 1: two complete passes over the 50 items. It uses CPU float32 with four threads, rank-4/alpha-8 q/v LoRA in the final four layers, learning rate 0.0002 and seed 42. All examples fit within 512 tokens, with no answer truncation and supervised end markers retained.

Validation loss is measured before training and at steps 25, 50, 75 and 100. The final checkpoint is evaluated; no parameter search or checkpoint selection based on these results is performed within this experiment. We generate all 12 validation answers and two sampled training answers both before and after, with identical prompts and greedy decoding capped at 128 new tokens. The cap and end-token status are logged for every generation.

The model sees passages and questions as context, while only final assistant answers and their endings are directly supervised. Only training examples update weights. Validation guides development; it is not a fresh independent final test, especially after repeated examination in this project.

## Review interpretation

For answerable items, a pass requires supported material claims and meaningful coverage of the fixed key ideas, without a material unsupported addition. Full coverage is distinguished from partial coverage. An unanswerable item passes only if the response acknowledges insufficient evidence and does not fabricate an answer. Empty responses fail. Source passages and intended key ideas remain available alongside the scored answers for audit.

The assistant prepared the examples and performs the review. This is not independent human certification, a calibrated model judge or a general accuracy benchmark. A higher pass count on this small development set would not establish broad generalization.

## Files and reproduction

- Frozen rubric: `docs/evaluation_rubric_v1.md`.
- Fixed settings, source-spec hash and dataset hashes: `book_sources/expanded_pilot_protocol.json`.
- Assistant-reviewed specifications: `data/instructions/living_by_the_code/expanded_specs_v1.json`.
- Prepared examples and token audit: `data/instructions/living_by_the_code/expanded_context_v1/`.
- Preparation script: `scripts/prepare_expanded_pilot.py`.
- Training results, generations and adapter: `runs/smollm2-expanded-001/`.
- Item-level assistant judgments: `review_before.json` and `review_after.json` in the run directory; aggregated counts and readable comparisons are saved as `review_summary.json` and `review_details.csv`.

Preparation can be rerun with `python -m scripts.prepare_expanded_pilot`; it refuses to replace changed dataset files. Training settings are frozen in the protocol rather than inferred from defaults. For a repeat run, choose a new output directory and record a new protocol.

No reinforcement learning or final-test evaluation is part of this experiment.
