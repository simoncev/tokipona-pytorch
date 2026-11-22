# Instruction-tuning learning report 03: training with source passages

Date: 22 November 2026. Run: `runs/smollm2-context-003`.

## Result

The corrected 50-update CPU experiment completed and saved its adapter. Validation cross-entropy decreased by approximately 0.84%. Three of five generated answers changed, but source review found no clear substantive improvement in the three validation answers. A numerical improvement and a wording change are not sufficient evidence of better answers.

| Measurement | Observed result |
|---|---:|
| Validation loss before training | 2.64823465 |
| After update 10 | 2.64707295 |
| After update 25 | 2.64290605 |
| After update 50 | 2.62604826 |
| Absolute decrease | 0.02218639 |
| Relative decrease | 0.83778% |
| Total elapsed time, including loading/generation/evaluation | Approximately 137 seconds |
| Maximum logged process RAM during updates | 1,873.2 MiB, approximately 1.83 GiB; not a measured whole-run peak |
| Trainable parameters | 51,200 |
| Changed generated answers | 2 of 2 sampled training answers; 1 of 3 validation answers |

No examples were dropped, all answers retained their end markers, and both dataset hashes match the completed run manifest. There were exactly 50 logged optimizer updates, all five comparisons contain before and after answers, and adapter weights were saved. The initial validation loss matches the previous interrupted starts because validation inputs and base weights were unchanged.

## Answer review

| Example | Before → after | Source-based assessment |
|---|---|---|
| 001, training: balance vision and execution | General big-picture/detail alignment → adds being hands-on enough to examine details | Slightly closer to the passage; still omits planning, listening and creating room for others |
| 002, training: gradual move into DevRel | Mentions beginning as a developer doing DevRel activities → only says to transition gradually | Loses useful detail; the shorter answer is less informative about the requested process |
| 007, validation: move from QA into development | Identical: community connections, meetups, conferences and volunteering | Supported central advice; no tuning-related improvement observed |
| 008, validation: communication practice | Identical: communication is a significant part of the job and takes effort to master | Relevant and supported reason, but omits practice methods and learning through teaching |
| 009, validation: good leadership | Replaces “someone” with “one”; otherwise identical | Preserves the source's growth theme; wording change adds no substantive value. “Supportive environment” is a broad inference, not a separately stated point in this excerpt |

These judgments distinguish factual support from completeness. A concise answer can be supported while omitting additional ideas in the reference. No calibrated correctness threshold was defined, so these counts are not a pass rate.

## What we learned

The adapter now measurably changes some generated text, unlike the original ten-update question-only pilot. However, changes can remove information as well as add it. We should not select an experiment merely because its answers differ or its loss is lower.

The model already answers these passage-grounded validation prompts reasonably before tuning. A small change in reference-token probabilities did not improve their central content. With only seven training examples and three development questions, this run cannot establish general improvement or robust instruction following.

The next useful dataset expansion is more diverse reviewed training passages, including explicit examples of insufficient evidence. Freeze an evaluation rubric for source support, central idea coverage and abstention before the next run. Keep final-test interviews untouched until the training and selection procedure is fixed. Increasing repetitions on these seven examples alone would not test broader learning.

## Experiment

We changed each example from question → answer to source passage + question → answer. The seven training answers and three validation answers are unchanged. Relevant excerpts come from the cited pages, with whitespace normalized and passage hashes recorded. The contributor-level train/validation division is preserved; final-test interviews are not used.

The longest complete example is 601 tokens. The 640-token sequence limit keeps every answer and its end marker. Questions and passages remain visible as context, but their labels are `-100`: only the final answer and ending contribute directly to the next-token loss. The passage still influences predictions through attention.

| Setting | Value |
|---|---|
| Starting model | Original local SmolLM2-360M-Instruct, not the first pilot's adapter |
| Device | CPU float32, four threads |
| Training / validation | 7 / 3 examples |
| Updates / batch size | 50 / 1 |
| Learning rate | 0.0002 |
| LoRA | Rank 4, alpha 8, q/v projections in the last four layers |
| Sequence / generation limits | 640 / 192 tokens |
| Decoding / seed | Greedy / 42 |

The prompt requests concise answers, without a fixed sentence count, matching the existing reviewed targets. It asks the model not to invent author-specific details and to acknowledge insufficient evidence. No missing-evidence examples were included in training, so abstention is not explicitly demonstrated by this dataset.

Fifty updates with batch size one mean seven complete passes over seven training examples, then one more example. That repetition can help us observe fitting, but seven distinct examples provide little diversity. Validation is also very small: all three questions come from one contributor. Repeatedly inspecting them makes them development data, not a fresh independent test.

## Power-loss recovery

The first context run, `smollm2-context-001`, saved logs through update 15 before the laptop lost power. It had no saved adapter or optimizer checkpoint. Its files were preserved. During restart `smollm2-context-002`, source review found that one answer listed advocacy activities excluded by its selected excerpt. That restart was stopped at update 24. Both partial runs are preserved and excluded from conclusions. The excerpt for example 002 was expanded to include the supporting material; the corrected dataset is versioned as `context_reviewed_v2`. The completed experiment starts fresh from the base in `smollm2-context-003`, with a 640-token limit to retain the expanded passage. It does not resume either interrupted run's weights.

## How to judge the result

Compare this run's validation loss and generated answers before versus after its own training. Cross-entropy values from the original question-only dataset and this passage-based dataset are not directly comparable measures of training success: the model receives different evidence.

The comparison with the first pilot also changes update count, sequence length, output allowance and prompt wording. It cannot isolate the effect of source context by itself. The separate context-control experiment in report 02 tested inference-time evidence use.

Source review uses the central ideas in each passage:

- **007, PDF p. 56:** community connections, meetups/conferences, peers and giving back.
- **008, PDF pp. 56–57:** development involves communication as well as coding; training, reading and teaching can help; teaching deepens understanding.
- **009, PDF p. 58:** good leaders help peers grow; leadership can be learned.

Exact reference wording is not required. Source support, central idea coverage, omissions and unsupported additions are reviewed separately. These three development questions cannot establish general accuracy or final-test performance.

## Reproduction and evidence

Preparation: `python -m scripts.prepare_context_pilot`.

```powershell
.\.venv\Scripts\python.exe -m tokipona_torch.pilot --model models/SmolLM2-360M-Instruct --train data/instructions/living_by_the_code/context_reviewed_v2/train.jsonl --valid data/instructions/living_by_the_code/context_reviewed_v2/valid.jsonl --output runs/smollm2-context-003 --steps 50 --max-length 640 --max-new-tokens 192
```

Use a new output directory for another experiment; the runner protects existing runs. The context dataset is in `data/instructions/living_by_the_code/context_reviewed_v2/`, including `preparation.json` with token counts and source checksum.

The run saves `run.json`, `loss.csv`, `token_inspection.json`, `before_after.json` and `adapter/`. All three validation answers, plus two training examples, are included in before/after comparisons. No reinforcement learning or final-test evaluation is involved.
