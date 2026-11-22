# Tokipona on PyTorch

This is a standalone port of the Slavic Prism training path in Tokipona, based on the source archive for commit `29b46d647b3b7c62caf5e9437408358b1e86ac89`. It uses PyTorch, Hugging Face Transformers and PEFT for Gemma 2 LoRA. Bring your own JSONL training data and base checkpoint.

| Tokipona baseline | PyTorch project |
|---|---|
| `src/prepare_adapters.py` | `split` preserves the original ordered 90/10 split. |
| `src/run_pipeline.py` | `train --loss all` learns from the complete text sequence, then `merge` produces a v23-style checkpoint. |
| `src/train_gemma_lora.py`, `src/train_skl1.py`, `training/run_sft_v25.py` | `train --loss response` supervises the final answer while retaining the prompt as context. Start from a merged v23-style checkpoint for cumulative tuning. |
| `training/fuse_v25.py`, `training/build_gguf.py` | `merge` writes a standard Hugging Face checkpoint and tokenizer for llama.cpp export. |
| Ollama clients | Use `generate` to test the model directly or export to GGUF for the existing clients. |

The existing MySQL exporters, RAG clients and notebooks can stay in your original project. This package starts at their JSONL output boundary. It contains no catalogue data, training corpus, or weights.

## Setup on your RTX 5060 Ti

From this folder in PowerShell, use Python 3.12 and a driver that supports the CUDA 12.8 wheel.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
.\.venv\Scripts\hf.exe auth login
```

Accept the terms on the [Gemma 2 2B IT model page](https://huggingface.co/google/gemma-2-2b-it) before downloading it, or point `--model` at a local Hugging Face Gemma 2 checkpoint. The examples below use `python` from the activated environment, or you can substitute `.\.venv\Scripts\python.exe`.

On macOS, install `torch==2.8.0` from PyPI and use `requirements.txt`. Omit `--four-bit` and use `--device mps`. This path uses float32, which increases memory use. Start with a short sequence. CPU also uses float32 and is suitable for tiny local checkpoints and smoke tests.

## Input formats

| JSONL row | Training behavior |
|---|---|
| `{"text":"[КНИГА: ... \| АВТОР: ...] Текст..." }` | Use `--loss all` for raw continuation. |
| `{"messages":[{"role":"user","content":"Што е животот?"},{"role":"assistant","content":"..."}]}` | Uses the Gemma tokenizer's chat template. A leading system message is folded into the first user turn. With `--loss response`, only the final assistant turn is supervised. |
| `{"prompt":"<start_of_turn>user\nПрашање...<end_of_turn>\n<start_of_turn>model\n","completion":"Одговор...<end_of_turn>"}` | Concatenates the two fields and masks the prompt when using `--loss response`. |
| `{"text":"<start_of_turn>user\nПрашање...<end_of_turn>\n<start_of_turn>model\nОдговор...<end_of_turn>"}` | Preserves the literal turn markers. With `--loss response`, only content after the final model header is supervised. |

Examples are right-truncated at `--max-length`. Training reports rows skipped because they have no usable target tokens after truncation. Padding is excluded from the loss. If the source uses overlapping book windows and you want held-out loss, supply train and validation exports from disjoint source passages or books. The original ordered split does not remove overlaps.

## Stage 1. Raw continuation

If you already have `train.jsonl` and `valid.jsonl`, use them directly. Otherwise split your existing export.

```powershell
python -m tokipona_torch split data/merged_train.jsonl data/raw
python -m tokipona_torch train --train data/raw/train.jsonl --valid data/raw/valid.jsonl --output runs/v23 --loss all --device cuda --four-bit --max-length 512 --steps 10
python -m tokipona_torch merge --adapter runs/v23 --output models/v23
```

The example uses 10 optimizer updates to check the pipeline before a longer run. Use `--accumulation 4 --max-length 1024 --learning-rate 2e-5` to carry over the corresponding raw-training settings. `--steps` counts optimizer updates, so compare the effective number of batches before reusing an MLX iteration count.

## Stage 2. Instruction tuning

Use the merged output of stage 1 or a compatible full Hugging Face Gemma 2 checkpoint. If you only have an MLX-quantized or GGUF copy of v23, start stage 1 from the original Gemma base to produce a compatible v23-style checkpoint.

```powershell
python -m tokipona_torch train --model models/v23 --train data/train_final1.jsonl --output runs/v25 --loss response --device cuda --four-bit --steps 10 --max-length 512
python -m tokipona_torch generate --adapter runs/v25 --prompt "Што е животот?" --system "Ти си словенски мислител." --device cuda --four-bit
python -m tokipona_torch merge --adapter runs/v25 --output models/v25
```

The default LoRA configuration uses rank 8 on the query and value projections of the last eight layers. PEFT alpha 160 at rank 8 gives a multiplier of 20. CUDA `--four-bit` uses NF4, so the numeric results can differ from the original MLX quantization. Omit it for regular LoRA. Use `generate --raw` to inspect plain continuation or to pass literal Gemma turns. Generation is greedy by default.

Add a separate `--valid` file for held-out loss. To continue an interrupted run, pass `--resume runs/v25/checkpoint-<step>` with the same base, adapter settings, training inputs and output directory. `--steps` remains the total update count. Keep the base checkpoint at its original path, or give `generate` or `merge` an explicit `--model` override.

## GGUF and Ollama

With [llama.cpp](https://github.com/ggml-org/llama.cpp) and its converter dependencies installed, convert the merged checkpoint. Substitute your local llama.cpp path and quantizer executable.

```powershell
python path/to/llama.cpp/convert_hf_to_gguf.py models/v25 --outfile models/v25-f16.gguf --outtype f16
llama-quantize models/v25-f16.gguf models/v25-q4_k_m.gguf Q4_K_M
```

Create `models/Modelfile` with `FROM ./v25-q4_k_m.gguf` and run `ollama create slavic-prism-v25-torch -f models/Modelfile`. For literal turn-marker experiments use the matching input format consistently in both training and inference.

## Verification

The CPU smoke test uses a small randomly initialized Gemma 2 checkpoint and synthetic data. It covers ordered splitting, full-sequence and response-only labels, both training stages, held-out loss, checkpoint resume, PEFT generation and merged-model generation. Full Gemma 2 2B training, GPU paths, and GGUF/Ollama conversion were outside the smoke test.
