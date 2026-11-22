# Instruction-tuning learning report 02: supplying the source

Run reviewed: `runs/smollm2-pilot-001` • Evaluation: 22 November 2026

## Question and method

Can the model answer these questions better when the relevant passage is included in the prompt? This is an inference experiment. It does not train the model, update the adapter or use final-test data.

The same three Lara Martin validation questions were evaluated with the original base model and the saved 10-step adapter. Each model received three prompt conditions, giving 18 generated answers in total.

| Condition | What changes | What it tests |
|---|---|---|
| Question only | Original question, with the allowance increased from 80 to 192 output tokens | Whether the earlier cutoff explains the weak answer |
| Instruction only | Same question, a two-sentence instruction and an explicit statement that no passage is available | Whether a clearer instruction alone improves restraint and relevance |
| Passage and instruction | Exactly the same instruction, with a relevant source excerpt replacing the missing-passage marker | Whether adding evidence improves source fidelity |

Both models use CPU float32 and greedy decoding, with the same 192-token output allowance. The source excerpts were selected from the cited validation pages, without inserting the reviewed reference answers into the prompts. Their text and checksums are saved with the results.

The instruction asks for two concise sentences, no invented author-specific details, and an acknowledgment when evidence is missing. An acknowledgment in the instruction-only condition is appropriate behavior, not an incorrect answer.

The original 80-token results are preserved. Comparing question-only outputs at 80 and 192 tokens tests output length. Comparing instruction-only and passage-plus-instruction outputs at the same 192-token allowance isolates supplying source evidence within this prompt design. Comparing question-only with passage-plus-instruction changes both wording and evidence, so it cannot isolate the effect of context alone.

## How to interpret this experiment

The source text is temporary input, not new model knowledge stored in the weights. A model can answer correctly by reading a passage even if it never learned that interview during training. This is context-grounded generation. The passages here were selected manually; an automated retrieval system would add a separate search step.

Any difference between the base model and the saved adapter concerns the effect of our small tuning run. Any difference between prompt conditions concerns what information and instructions we provide at inference time. Keep those two comparisons separate.

This is an assistant-led qualitative review of three development questions. It does not establish general accuracy, statistical significance or final-test performance. The base model may already know some related material; we cannot establish its original training exposure.

## Results: reading the actual answers

All 18 generations completed. Across all nine question/condition pairs, the base model and saved adapter produced identical text. The improvements described below therefore come from providing different inputs, not a demonstrated benefit of the ten-step adapter.

| Question | Question only, up to 192 tokens | Concise instruction, no passage | Same instruction with passage | Source review |
|---|---|---|---|---|
| 007: moving from QA into development | Long list of courses, online communities and coding challenges; still reaches the limit | Suggests a development foundation and learning technologies, rather than acknowledging missing evidence | Identifies community connections, local meetups and conferences | Supplying the passage recovers the central source recommendation; giving back to the community is omitted |
| 008: why communication practice matters | Completes a generic list about clarity, trust, misunderstandings, collaboration and productivity | Gives a general explanation about conveying ideas and collaboration | Says communication improves skills and effectiveness | Still vague: it does not explain that development is more than coding or mention learning through teaching |
| 009: describing a good leader | Completes generic advice about inspiration, fairness, accountability and decisions | Repeats generic transparency, accountability and listening claims | Identifies helping others grow and become better | Recovers the source's central definition; the separate point that leadership can be learned is omitted |

For question 009, the source-grounded answer was: “Lara Martin describes a good leader as someone who helps others grow and become better.” This is a direct conceptual match to the passage, unlike the generic traits supplied without it.

For question 008, the source-grounded answer was: “Lara Martin recommends communication practice for programmers because it helps them improve their skills and make their work more effective.” This sounds plausible but is too general to demonstrate that the specific evidence was understood well.

These are qualitative judgments about central ideas, not a calibrated pass/fail score. In this small set, the passage clearly improved source-specific content on questions 007 and 009. It did not produce a sufficiently specific explanation on question 008.

## What the controls revealed

**A longer allowance fixes some incomplete endings, but not unsupported attribution.** With question-only prompts, question 007 still reached 192 tokens. Questions 008 and 009 ended naturally at 135 and 88 tokens. Their completed answers still contained details not established by the tested passages. The earlier cutoff was a real limitation, but increasing it did not by itself solve source fidelity.

**A concise instruction controls length better than it controls evidence use.** Without a passage, the model still answered all three questions as though it knew the interviewee's advice. It did not acknowledge the missing evidence despite the explicit instruction. This is a remaining failure of the prompt/model combination.

**Supplying a passage makes some source-specific answers possible without further training.** The three passage-grounded answers ended naturally at 37, 23 and 18 tokens. All were one sentence, despite the request for two; they were concise, but did not obey the exact sentence-count instruction. Do not confuse improved grounding with perfect instruction following.

## Next learning step

Prepare a separate context-grounded training dataset: each example contains a relevant passage, a question and a concise supported answer. Preserve the current interview-level splits and leave this completed pilot unchanged. Inspect the new token lengths before setting the sequence limit, because adding passages increases the prompt size.

Then compare the base and newly trained adapter under identical context-grounded prompts. Evaluate factual support, missing key ideas, refusal when evidence is unavailable, and instruction compliance as separate checks. The weak result for question 008 and the missing-evidence behavior are useful development signals. Keep final-test interviews out of these decisions.

## Evidence and reproduction

- Results and all 18 generated answers: `runs/smollm2-pilot-001/evaluation/context_control.json`.
- Evaluation script: `scripts/evaluate_context.py`.
- Source excerpts: PDF pages 56, 56–57 and 58, retained in the prepared source sections.

The script protects an existing evaluation file from overwrite. The previous run, original comparison file and adapter are preserved.
