# Source-grounded review rubric v1

Frozen on 22 November 2026, before generating the expanded experiment's model answers. This is assistant-led qualitative evaluation, not a calibrated independent benchmark.

Each answer is assessed against its supplied excerpt, not broad plausibility or exact wording. Reference answers and key ideas are fixed before training. All 12 validation prompts use identical wording and greedy decoding before and after tuning. No final-test sources are used.

| Criterion | Values |
|---|---|
| Source support | 2: all material claims supported or conservative paraphrases; 1: partly supported or too vague to establish grounding; 0: contradicts evidence or supplies unsupported material claims |
| Key idea coverage | 2: covers the fixed key ideas; 1: meaningful but incomplete coverage; 0: misses the requested idea |
| Unsupported claim | Yes/no, with a short explanation when present |
| Missing evidence handling | For unanswerable items: explicitly acknowledges insufficient evidence and does not invent an answer |

A positive item passes if support=2, coverage is at least 1, and no material unsupported claim is present. A negative item passes only if missing evidence is acknowledged and no fabricated answer is supplied. Do not award a negative-item pass merely for a terse or empty response. Empty or unusable answers fail.

Record completeness separately: coverage=1 can pass for supported partial content, but coverage=2 is preferable. Compare pass-to-fail regressions as well as fail-to-pass improvements. Record answerable and unanswerable counts separately, because they have different requirements. Explicit abstention on an answerable item is a failure to answer.

Scores remain judgments by the same assistant that prepared the data. Small sample size, repeated validation use and lack of independent reviewers limit the conclusions. Validation informs development; a final test is reserved for later, after the process is fixed.

No score in this rubric is used to update model weights or as an RL reward.
