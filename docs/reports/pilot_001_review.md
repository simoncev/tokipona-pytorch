# Instruction-tuning learning report 01

Run: `runs/smollm2-pilot-001` • Reviewed: 22 November 2026

## Finding

The supervised training pipeline completed, but the four originally saved generated answers did not change. A follow-up evaluation generated all three validation answers from the base and saved adapter: all three pairs were also identical, including the third validation question that was absent from the original comparison file. Validation loss decreased slightly. These observations are compatible: training can change token probabilities without changing which token greedy decoding selects.

This is an assistant-led source review, not a human certification or an independent automated benchmark. Judgments below concern fidelity to the cited book passages, rather than whether general advice sounds plausible.

## What the experiment did

| Item | Observed result |
|---|---|
| Base model | SmolLM2-360M-Instruct, loaded from the local model directory |
| Hardware execution | CPU, float32, four CPU threads |
| Training examples | 7, from Britt Barak, Ellen Shapiro and Antonio Leiva |
| Validation examples | 3, all from Lara Martin |
| Optimizer updates | 10, batch size 1 |
| Trainable parameters | 51,200 of 361,872,320, approximately 0.0141% |
| Adapter placement | q/v projections in the last four layers |
| Total recorded duration | Approximately 54 seconds, including loading, generation and evaluation |
| Logged process RAM | Approximately 1.78–1.81 GiB during update logging; not a measured whole-run peak |
| Final test | Not used |

With seven examples and ten updates, the model saw each training example once, then three examples a second time. This is a mechanics pilot, not training on all 520 pages of the book.

The current training and validation dataset hashes match those recorded by the run. The saved adapter contains eight nonzero LoRA B matrices. In this implementation B starts at zero, so their nonzero values are evidence that adapter updates occurred. Unchanged text therefore does not mean that training was skipped.

## Reading the loss results

| Evaluation point | Validation cross-entropy |
|---|---:|
| Before training | 3.48878853 |
| After step 5 | 3.48845860 |
| After step 10 | 3.48806808 |

The reduction was 0.00072044, approximately 0.02065%. This is a very small numerical improvement, with no repeated-run uncertainty estimate. It is not evidence of a useful improvement in generated answers.

During loss evaluation, the model is supplied the question and the correct preceding answer tokens and predicts the next answer token. This is called teacher forcing. During generation it must continue from its own previous predictions. A slightly lower reference-answer loss need not produce a better free-running answer.

The step-by-step training losses use different examples. For example, step 1 and step 2 cannot be compared as if they measured the same question. Three repeated training examples did show small loss reductions, which is consistent with a small amount of fitting.

## Source review of the saved answers

The same findings apply to both the before and after text, because each pair is identical. The original output allowance was 80 new tokens. The follow-up evaluation confirmed that all six validation generations reached this limit without emitting their end token. Judgments apply only to these captured prefixes; we cannot assume what an unrestricted continuation would say.

| Example | Split and source | What the source supports | What the generated answer does | Review |
|---|---|---|---|---|
| 001: balancing vision and execution | Training; Britt Barak, PDF p. 23 | Translate vision into plans, enable others, listen, move between strategy and detail | Supplies a SMART-goals framework and repeats a vision heading, rather than explaining the central balance | On topic, but important details are unsupported by this passage and the response is incomplete |
| 002: moving into Developer Relations | Training; Britt Barak, PDF pp. 24–25 | Build engineering experience, try advocacy alongside development, gradually shift the balance | Gives generic introductory programming study advice | Misses the requested career transition and is incomplete |
| 007: moving from QA into development | Validation; Lara Martin, PDF p. 56 | Meetups, conferences, peers making similar transitions, and community contribution | Recommends online-course providers and speaks as Lara Martin | Misses the passage's central recommendation; named providers are unsupported by the cited passage |
| 008: communication practice | Validation; Lara Martin, PDF pp. 56–57 | Communication is part of development; practice through training, reading and teaching; teaching strengthens understanding | Lists clarity, trust and fewer misunderstandings | Broadly relevant, but these specifics are not established by this passage and the teaching point is absent |
| 009: describing a good leader | Validation; Lara Martin, PDF p. 58 | A leader helps peers grow, and leadership can be learned | Describes inspiration, motivation, fairness, transparency and accountability | Generic leadership advice with some thematic overlap; the passage's growth and learnability points are not clearly conveyed, and several attributed details are unsupported |

“Unsupported” here does not mean the advice is universally false. It means the model presents details as the contributor's advice without evidence in the passage we are testing. The references are paraphrases; exact wording is not required.

All three validation items were inspected. None showed a before/after improvement under these generation settings. We have not defined a calibrated correctness threshold, so this is not a pass-rate or accuracy measurement. The two original validation comparisons reproduced exactly during the follow-up evaluation.

## What this teaches us

1. **An instruction model already follows familiar answer patterns.** Before our tuning, it could produce fluent lists of career advice. Fluency does not demonstrate knowledge of this book.
2. **Loss and generated quality measure different things.** Small improvements in reference-token probabilities can leave greedy answers unchanged.
3. **The validation task includes unseen factual attribution.** Lara Martin's interview was excluded from training. Without a source passage in the prompt, the model is asked to recall her specific advice despite not being taught that interview in this experiment. That mixes a knowledge test with an instruction-following test.
4. **Few examples and small updates limit the conclusion.** We cannot infer that LoRA is ineffective, that the model cannot learn, or that a longer run will necessarily solve the problem.
5. **Generation length is another variable.** All three validation questions hit the token limit both before and after tuning. A longer-output control should be evaluated separately before interpreting missing content as purely a knowledge or reasoning failure.

## Next controlled learning exercise

Keep this run unchanged. Use a separate evaluation to supply a short relevant book passage with each question, and ask for a concise answer grounded only in that passage. Compare the base and current adapter using the same prompts and decoding settings. This tests whether the model can use supplied evidence and separates that ability from remembering an unseen interview. This control requires no further training.

For a later training run, a context-grounded instruction dataset can teach that behavior explicitly. Keep each passage and its question variants in its existing source split. Do not add validation answers to training to make the current score look better. Reserve final-test examples until settings are fixed.

## Evidence files

- `runs/smollm2-pilot-001/run.json`: settings, counts, hashes and validation losses.
- `runs/smollm2-pilot-001/loss.csv`: individual update logs.
- `runs/smollm2-pilot-001/before_after.json`: the four original generated comparisons.
- `runs/smollm2-pilot-001/evaluation/validation_comparison.json`: all three validation pairs with token counts and stopping reasons; evaluation only, with no weight updates.
- `data/prepared/living_by_the_code/sections.jsonl`: cited source passages.
- `data/instructions/living_by_the_code/reviewed/`: training and validation references.

No reinforcement learning, reward function or final-test evaluation was involved.
