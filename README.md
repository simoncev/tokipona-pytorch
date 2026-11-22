# Tokipona on PyTorch

An interactive experiment for understanding instruction tuning in Python/PyTorch on an **Intel i7 CPU with 32 GB RAM**. The starting model is `HuggingFaceTB/SmolLM2-360M-Instruct`.

The goal is to turn an English PDF into traceable instruction/answer examples, train a small LoRA adapter, and inspect exactly what changes before and after training. GPU training is outside the current experiment.

## Current implementation

This project began as a standalone port of the Slavic Prism training path from the Tokipona source archive at commit `29b46d647b3b7c62caf5e9437408358b1e86ac89`.

**SmolLM2/Llama support and a CPU pilot notebook are implemented. The local SmolLM2 model is now available and the first real CPU pilot completed. Offline mechanics tests additionally use a tiny random Llama and synthetic tokenizer. See the learning reports for observed results and limitations.**

| Component | Current behavior |
|---|---|
| `tokipona_torch/data.py` | Model chat templates, response-only labels, token/mask inspection, draft rejection |
| `tokipona_torch/model.py` | SmolLM2/Llama and Gemma 2 loading, CPU float32, checked q/v LoRA targets |
| `tokipona_torch/pilot.py` | CPU-only supervised pilot, loss/RAM/time logs, validation, saved adapter and before/after generations |
| `notebooks/01_instruction_pilot.ipynb` | Interactive data, chat, token, masking, baseline, training and comparison cells |
| PDF preparation | Extracted pages, source sections, grouped splits, draft and reviewed pilot examples |
| Reinforcement learning | Separate future stage; not implemented |

Response-only labels mask prompt tokens and padding with `-100`. Only the final assistant answer is supervised, including its end marker. The CPU pilot rejects truncation or a missing supervised end marker. The general CLI retains legacy full-sequence loss and raw Gemma response detection; use structured `messages` for SmolLM2.

## Start the instruction pilot

Use the existing CPU PyTorch environment. Install notebook support when network access is available:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

Open `notebooks/01_instruction_pilot.ipynb` in VS Code and select the project's `.venv` interpreter as its kernel. The notebook is unexecuted and does not claim results on the book. The first tokenizer/model load downloads `HuggingFaceTB/SmolLM2-360M-Instruct` unless it is cached. For an offline copy, set `config.model` to a complete local Hugging Face checkpoint directory.

The same experiment can run without a notebook kernel:

```powershell
.\.venv\Scripts\python.exe -m tokipona_torch.pilot
```

Defaults: CPU float32, 4 CPU threads, batch size 1, 256-token sequences, 10 optimizer steps, rank-4 LoRA on q/v projections in the last 4 layers. Only the reviewed 7 training examples update weights; the 3 validation examples guide development. Final test data is never opened. This is a mechanics experiment, not enough data to demonstrate useful improvement.

Outputs in `runs/smollm2-pilot-001/`:

- `token_inspection.json`: rendered chats, token pieces, IDs, attention masks, labels, supervised target counts and EOS/truncation checks.
- `loss.csv`: step, example, training loss, periodic validation loss, elapsed seconds and process RAM.
- `before_after.json`: fixed greedy generations before and after tuning, with references and development split labels.
- `run.json`: settings, model revision when available, dependency versions, dataset hashes, parameter counts and validation results.
- `adapter/`: PEFT adapter weights and tokenizer.

Choose a new output path for another run (`--output runs/smollm2-pilot-002`). The runner refuses to overwrite a nonempty run directory. Loss is shifted next-token cross-entropy on answer tokens; validation NLL is weighted by the number of supervised tokens. It is not a reward function.

See [pilot source review](docs/pilot_review.md) for the review scope and dataset limitations.

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

`requirements.txt` declares Transformers/PEFT and process-memory inspection. PyTorch is already installed in this project environment; install a CPU build separately for a fresh environment. `requirements-notebook.txt` adds kernel support and notebook validation. PDF extraction remains a separate dependency and is not needed to train on the existing JSONL. `requirements-cuda.txt` belongs to the earlier GPU workflow and is not used for this experiment.

## Dataset status and policy

No original `train.jsonl`, dataset, PDF, notebook or model weights were found during the initial inspection. The original `train.jsonl` may exist elsewhere; its location still needs to be supplied or identified. Missing data must not be treated as recovered. The subsequent book preparation below creates a new dataset; it does not recover the original one.

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

The main CLI defaults to SmolLM2 and CPU, while retaining legacy Gemma support. For the observable book experiment, prefer `python -m tokipona_torch.pilot`.

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

Training and generation now default to `--device cpu`; omit `--four-bit` for this CPU experiment. The default model is `HuggingFaceTB/SmolLM2-360M-Instruct`. Raw continuation training is available but is not a prerequisite for the planned instruction-tuning experiment starting from an instruction model.

## Verification scope

The inherited README reported a synthetic CPU smoke test using a tiny randomly initialized Gemma 2 checkpoint, covering splitting, labels, training, validation, resume, adapter generation and merged generation. That historical test was not rerun during the current inspection, and no test suite is present in this folder.

Six offline checks exercise system messages, final-answer masking, real EOS versus padding, truncation and draft rejection, shifted cross-entropy, LoRA updates with frozen base weights, PDF discretionary-hyphen cleanup, and a complete tiny CPU pilot including adapter reload. They use a synthetic tokenizer with the published SmolLM2 chat-template structure and a randomly initialized tiny Llama, not the real SmolLM2 vocabulary or weights. The real tokenizer and locally downloaded model have also been used for CPU training. Reports below distinguish mechanics checks from observed model behavior. GGUF/Ollama export is deferred until the supervised experiment is working and understood.

## Living by the Code: first preparation pass

The uploaded 520-page book was extracted through the uploaded-document text service. Text was returned for 450 pages; 70 pages returned no text and need visual review before assuming they are blank. The original PDF could not be downloaded locally, and installing `pypdf` failed because network name resolution was unavailable. This pass does not use local PDF parsing or OCR.

- `data/extracted/living_by_the_code/pages.jsonl`: all 520 page records, preserving original extraction text and explicit no-text status.
- `book_sources/living_by_the_code.json`: table-of-contents section boundaries. Printed Arabic page numbers map to PDF pages by adding 18.
- `scripts/prepare_book.py`: standard-library preparation script; cleans discretionary hyphens and common running headers without changing the raw records.
- `data/prepared/living_by_the_code/sections.jsonl`: 41 interviews and 10 Further Reading sections, with contributor and page metadata. Further Reading contributors remain unset pending review.
- `data/prepared/living_by_the_code/manifest.json`: source checksum, extraction limitations, seed and split membership. Each interview and its following Further Reading stay together: 33 interview groups for training, 4 validation, 4 final test.
- `data/instructions/living_by_the_code/pilot.draft.jsonl`: 10 assistant-authored paraphrased examples grounded in source pages. These remain **unapproved drafts**. Separate assistant-reviewed copies are available in `data/instructions/living_by_the_code/reviewed/train.jsonl` (7) and `valid.jsonl` (3).
- `data/prepared/living_by_the_code/drafts/`: separate draft files assigned using the source manifest. An empty test draft file is intentional; final test source groups are reserved for later evaluation preparation.

Rebuild sections and assign draft examples with:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_book.py
```

The script checks complete page-record coverage, unique draft IDs, contributor identity and cited-page membership. Human review must still check answers against the cited pages, extraction quality, and near-duplicate content across groups. Source-level grouping alone cannot prevent repeated ideas across interviews. Advice is attributed to its speaker; historical technology predictions should not become unqualified present-day recommendations.

Raw book text and generated examples remain in ignored `data/`; only preparation code, metadata and documentation are tracked. The assistant reviewed the pilot against its cited source pages; see `docs/pilot_review.md`. Keep the draft unchanged and edit reviewed copies deliberately. `scripts/prepare_book.py` rebuilds source sections and draft assignments, not the reviewed examples. A larger dataset still needs further review. The training implementation now accepts SmolLM2; no book-model run has completed.

Run the offline verification suite with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Architecture and chat-template references: [model config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/main/config.json) and [tokenizer config](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/main/tokenizer_config.json). The runtime loads the actual model tokenizer; the offline fixture does not substitute for it.

## Learning reports

- [01: First supervised pilot](docs/reports/pilot_001_review.md): training completed; generated validation answers were unchanged.
- [02: Source-context controls](docs/reports/pilot_002_context_control.md): compare longer output, explicit instructions and supplied passages without training.
- [03: Training with source passages](docs/reports/pilot_003_context_training.md): completed run `smollm2-context-003`; validation loss 2.64823 to 2.62605, with no clear substantive validation-generation improvement.

The corrected context dataset is `data/instructions/living_by_the_code/context_reviewed_v2/`. Prepare it with `python -m scripts.prepare_context_pilot`. Full examples reach 601 tokens, so use the recommended 640-token sequence limit. The desired answers and interview-level splits are preserved. The preparation script will not overwrite a modified dataset.
