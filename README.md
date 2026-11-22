# Tokipona on PyTorch

An interactive experiment for understanding instruction tuning in Python/PyTorch on an **Intel i7 CPU with 32 GB RAM**. The proposed starting model is `HuggingFaceTB/SmolLM2-360M-Instruct`.

The immediate goal is to turn an English PDF into traceable instruction/answer examples, train a small LoRA adapter, and inspect exactly what changes before and after training. GPU training is outside the current experiment.

## Current implementation

This project began as a standalone port of the Slavic Prism training path from the Tokipona source archive at commit `29b46d647b3b7c62caf5e9437408358b1e86ac89`.

**The current code still targets Gemma 2. SmolLM2 support and the notebook workflow below are planned, not implemented.**

| Component | Implemented behavior | Required change |
|---|---|---|
| `tokipona_torch/data.py` | JSONL ingestion; chat, prompt/completion and text formats; response-only labels; truncation; ordered 90/10 split | Model-appropriate chat handling, visible token/mask inspection, and separate train/validation/test preparation |
| `tokipona_torch/model.py` | Gemma 2 loading, LoRA adapters, adapter loading | Accept SmolLM2, verify compatible LoRA targets, and use CPU float32 |
| `tokipona_torch/__main__.py` | `split`, `train`, `generate`, `merge`; loss logging, optional validation and checkpoint resume | CPU-first defaults and observable evaluation workflow |
| PDF preparation | Not implemented | Extract text, preserve provenance, and create reviewed instruction examples |
| Interactive notebook | Not implemented | Inspect data, tokens, masks, loss and before/after generation |
| Reinforcement learning | Not implemented | Separate future stage with an explicit reward function |

Response-only training currently masks prompt tokens with `-100`; padding is also excluded from the loss. Only the final assistant answer is supervised. Examples with no usable target after truncation are skipped and counted. Raw-text response detection and system-message folding currently contain Gemma-specific assumptions.

## Local environment baseline

Inspection on 22 November 2026 found:

| Dependency | Project `.venv` |
|---|---|
| Python | 3.14.2 |
| PyTorch | 2.15.0+cpu |
| Transformers | 4.57.6 |
| PEFT | 0.18.0 |
| Accelerate | 1.12.0 |
| SentencePiece | 0.2.3 |
| PDF extraction / notebook support | `pypdf`, `ipykernel` and `notebook` were absent |

PyTorch, Transformers, PEFT and Accelerate imported successfully. A CPU tensor operation succeeded and `pip check` reported no broken requirements. This verifies basic environment operation, not a completed model-training run.

The default shell `python` resolves to `C:\Python314\python.exe`, where PEFT was not available. Use the project interpreter explicitly:

```powershell
.\.venv\Scripts\python.exe -c "import sys, torch, peft; print(sys.executable); print(torch.__version__); print(peft.__version__)"
.\.venv\Scripts\python.exe -m pip check
```

In VS Code, select **Python: Select Interpreter** and choose:

```text
C:\Development\Genrep\tokipona-pytorch\.venv\Scripts\python.exe
```

Select that same environment as the notebook kernel once kernel support is installed. The active VS Code interpreter was not independently confirmed during inspection.

`requirements.txt` contains the existing Transformers/PEFT dependencies but does not declare PyTorch, PDF extraction or notebook packages. CPU setup documentation and dependencies need to be completed together. `requirements-cuda.txt` belongs to the earlier GPU workflow and is not used for this experiment.

## Dataset status and policy

No `train.jsonl`, other dataset, English PDF, notebook or model weights were found in the project folder during the initial inspection. The original `train.jsonl` may exist elsewhere; its location still needs to be supplied or identified. Missing data must not be treated as recovered.

The existing `.gitignore` excludes `.venv/`, `data/`, `runs/`, `models/`, Python caches and GGUF files. Keep source PDFs and generated datasets under `data/` so they remain local by default.

Planned layout:

```text
data/
  source/          English PDF
  extracted/       Page text and extraction metadata
  instructions/    Reviewed examples with provenance
  splits/
    train.jsonl
    valid.jsonl
    test.jsonl
notebooks/         Interactive inspection notebook
runs/              Adapters, loss logs and comparison outputs
models/            Local base or merged checkpoints
```

Each generated example should carry the source document, page or passage identifier, and task type. Answers must be supported by the source; extraction alone does not create reliable question/answer pairs. Review the extracted text and generated examples before training. Scanned PDFs may require a separate OCR step.

Use distinct datasets with distinct purposes:

| Split | Purpose |
|---|---|
| Training | Updates adapter weights |
| Validation | Guides experiment settings and checkpoint selection |
| Final test | Reserved for final evaluation after settings are fixed |

Split by source passage or section **before** generating related questions. Keep overlapping passages, duplicates and paraphrases in the same split. Record the split seed, source membership and counts. The current ordered 90/10 splitter does not provide these protections or produce a final test set.

## Summary plan

1. **Prepare the CPU environment.** Resolve VS Code/Pylance to the project interpreter, add PDF and notebook dependencies, and explicitly use CPU execution. Avoid automatic GPU selection and CUDA quantization. Begin with float32, batch size 1 and short sequences; measure runtime and RAM before scaling.
2. **Adapt model loading and chat formatting.** Support `HuggingFaceTB/SmolLM2-360M-Instruct`, use its tokenizer chat template, preserve supported system messages, and verify the LoRA target modules. Do not reuse literal Gemma turn markers for SmolLM2.
3. **Prepare the English PDF dataset.** Extract page text, inspect extraction quality, group source passages into train/validation/test splits, then create and review instruction examples with provenance.
4. **Make tokenization and masking visible.** Show the original example, rendered chat, token IDs, decoded token pieces, attention mask and labels. Identify prompt, answer and padding tokens; explain `-100` and the shifted next-token prediction used by causal language models. Check whether truncation removes answer tokens or the end-of-answer marker.
5. **Run an observable supervised fine-tuning experiment.** Record baseline generation on fixed development prompts, train a small LoRA adapter with response-only loss, and log training loss, validation loss, supervised token counts, trainable parameter counts, settings and elapsed time. Start with a short smoke run before a longer experiment.
6. **Compare and evaluate.** Generate again with the same prompts and decoding settings; display before/after answers side by side. Use validation results while developing. Evaluate the untouched final test set only after the configuration is fixed. Lower training loss alone is not evidence of improved instruction following.
7. **Consider reinforcement learning separately.** Start only after the supervised baseline is understood. Define a reward function, its inputs, scoring rules and limitations, and separate reward-training data from evaluation data. Supervised response loss is not a reward function; the existing training command does not perform reinforcement learning.

The notebook should expose these steps as separate cells so data preparation, encoding, baseline generation, training and evaluation can be inspected independently. Save the seed, model revision, dependency versions, dataset identifiers and generation settings with each run.

## Existing command-line interface

The following commands describe the current Gemma-based implementation. They do not imply SmolLM2 support is complete.

```powershell
.\.venv\Scripts\python.exe -m tokipona_torch --help
.\.venv\Scripts\python.exe -m tokipona_torch train --help
.\.venv\Scripts\python.exe -m tokipona_torch generate --help
```

- `split`: ordered 90/10 JSONL split; retained for the legacy workflow.
- `train --loss all`: full-sequence continuation training.
- `train --loss response`: supervise the final answer with the prompt retained as context.
- `generate`: generate from a base model or a PEFT adapter; greedy decoding by default.
- `merge`: merge an adapter into a full Hugging Face checkpoint.

For current training or generation commands, specify `--device cpu` and omit `--four-bit`. The current default model remains `google/gemma-2-2b-it`; passing the proposed SmolLM2 model will fail until model loading is adapted. Raw continuation training is available but is not a prerequisite for the planned instruction-tuning experiment starting from an instruction model.

## Verification scope

The inherited README reported a synthetic CPU smoke test using a tiny randomly initialized Gemma 2 checkpoint, covering splitting, labels, training, validation, resume, adapter generation and merged generation. That historical test was not rerun during the current inspection, and no test suite is present in this folder.

The new experiment still needs checks for SmolLM2 chat boundaries, response labels, padding and truncation, split isolation, CPU training, and reproducible before/after generation. No English PDF has been processed and no SmolLM2 adapter has been trained yet. GGUF/Ollama export is deferred until the supervised experiment is working and understood.
