"""Command line for preparing data, tuning Gemma, and using adapters."""

import argparse
from pathlib import Path

from .data import gemma_messages, load_dataset, split_jsonl


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def train(args):
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    from .model import add_lora, device_for, load_base

    if args.max_length < 2 or args.learning_rate <= 0:
        raise ValueError("training requires --max-length >= 2 and a positive learning rate")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("training output already contains files; choose a new directory or --resume")
    if args.resume and not (args.resume / "trainer_state.json").is_file():
        raise ValueError("--resume must point to a Trainer checkpoint directory")
    if args.valid and args.train.resolve() == args.valid.resolve():
        raise ValueError("use a separate file for validation")
    device = device_for(args.device)
    set_seed(args.seed)
    model, tokenizer, dtype = load_base(args.model, device, args.four_bit)
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("--max-length exceeds the model context window")
    model = add_lora(model, args)
    model.config.use_cache = False
    model.print_trainable_parameters()
    tokenizer.padding_side = "right"
    train_data = load_dataset(args.train, tokenizer, args.max_length, args.loss)
    valid_data = load_dataset(args.valid, tokenizer, args.max_length, args.loss) if args.valid else None
    training = TrainingArguments(
        output_dir=str(args.output),
        max_steps=args.steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cpu=device == "cpu",
        bf16=device == "cuda" and dtype == torch.bfloat16,
        fp16=device == "cuda" and dtype == torch.float16,
        logging_steps=args.log_every,
        eval_strategy="steps" if valid_data else "no",
        eval_steps=args.eval_every,
        save_steps=args.save_every,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training, train_dataset=train_data, eval_dataset=valid_data,
        processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer),
    )
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    trainer.save_model()
    if valid_data:
        print(trainer.evaluate())


def generate(args):
    import torch

    from .model import device_for, load_adapter, load_base

    device = device_for(args.device)
    if args.adapter:
        model, tokenizer = load_adapter(args.adapter, args.model, device, args.four_bit)
    else:
        model, tokenizer, _ = load_base(args.model or DEFAULT_MODEL, device, args.four_bit)
    if args.raw:
        has_bos = tokenizer.bos_token and args.prompt.startswith(tokenizer.bos_token)
        inputs = tokenizer(args.prompt, add_special_tokens=not has_bos, return_tensors="pt")
    else:
        turns = [{"role": "user", "content": args.prompt}]
        if args.system:
            turns.insert(0, {"role": "system", "content": args.system})
        inputs = tokenizer.apply_chat_template(
            gemma_messages(turns), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: inputs[key].to(input_device) for key in ("input_ids", "attention_mask")}
    start = inputs["input_ids"].shape[1]
    if start + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt plus requested output exceeds the model context window")
    if args.temperature < 0:
        raise ValueError("--temperature must be non-negative")
    sampling = {"do_sample": args.temperature > 0}
    if args.temperature > 0:
        sampling.update(temperature=args.temperature, top_p=0.9)
    model.eval()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, **sampling)
    print(tokenizer.decode(output[0, start:], skip_special_tokens=True))


def merge(args):
    from .model import load_adapter

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("merge output already contains files; choose a new directory")
    model, tokenizer = load_adapter(args.adapter, args.model, "cpu", merge=True)
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Merged Hugging Face checkpoint: {args.output}")


DEFAULT_MODEL = "google/gemma-2-2b-it"


def main():
    parser = argparse.ArgumentParser(prog="python -m tokipona_torch")
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split", help="split an existing JSONL in the original 90/10 order")
    split.add_argument("source", type=Path)
    split.add_argument("destination", type=Path)

    training = commands.add_parser("train", help="train a Gemma 2 LoRA adapter")
    training.add_argument("--model", default=DEFAULT_MODEL)
    training.add_argument("--train", type=Path, required=True)
    training.add_argument("--valid", type=Path)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--loss", choices=("all", "response"), required=True)
    training.add_argument("--max-length", type=positive, default=1024)
    training.add_argument("--steps", type=positive, default=100)
    training.add_argument("--batch-size", type=positive, default=1)
    training.add_argument("--accumulation", type=positive, default=1)
    training.add_argument("--learning-rate", type=float, default=1e-5)
    training.add_argument("--layers", type=positive, default=8)
    training.add_argument("--rank", type=positive, default=8)
    training.add_argument("--alpha", type=positive, default=160)
    training.add_argument("--log-every", type=positive, default=10)
    training.add_argument("--eval-every", type=positive, default=50)
    training.add_argument("--save-every", type=positive, default=100)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--resume", type=Path)

    inference = commands.add_parser("generate", help="generate from a merged model or PEFT adapter")
    inference.add_argument("--model", help="Hugging Face model or adapter base override")
    inference.add_argument("--adapter", type=Path)
    inference.add_argument("--prompt", required=True)
    inference.add_argument("--max-new-tokens", type=positive, default=128)
    inference.add_argument("--temperature", type=float, default=0)
    prompt_mode = inference.add_mutually_exclusive_group()
    prompt_mode.add_argument("--raw", action="store_true")
    prompt_mode.add_argument("--system")

    for command in (training, inference):
        command.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
        command.add_argument("--four-bit", action="store_true", help="CUDA NF4 loading")

    fusion = commands.add_parser("merge", help="merge a PEFT adapter into a full Hugging Face checkpoint")
    fusion.add_argument("--adapter", type=Path, required=True)
    fusion.add_argument("--output", type=Path, required=True)
    fusion.add_argument("--model", help="Hugging Face base override")

    args = parser.parse_args()
    try:
        if args.command == "split":
            split_jsonl(args.source, args.destination)
        else:
            {"train": train, "generate": generate, "merge": merge}[args.command](args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
