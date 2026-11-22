"""Loading SmolLM2/Llama and legacy Gemma checkpoints with PEFT adapters."""

import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def device_for(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch installation")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this PyTorch installation")
    return requested


def load_base(source, device, four_bit=False, tokenizer_source=None, merge=False):
    config = AutoConfig.from_pretrained(source)
    if config.model_type not in {"gemma2", "llama"}:
        raise ValueError("supported checkpoint architectures: llama (SmolLM2) and gemma2")
    if getattr(config, "quantization_config", None):
        raise ValueError("supply an unquantized checkpoint, or use --four-bit to quantize at load time")
    if four_bit and device != "cuda":
        raise ValueError("--four-bit requires CUDA and requirements-cuda.txt")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source or source, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer is required for response boundary inspection")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a padding or end-of-sequence token")
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32
    if merge:
        dtype = "auto"
    elif device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    options = {"config": config, "dtype": dtype, "attn_implementation": "eager"}
    if four_bit:
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype,
        )
        options["device_map"] = {"": torch.cuda.current_device()}
    model = AutoModelForCausalLM.from_pretrained(source, **options)
    if not four_bit and device != "cpu":
        model.to(device)
    return model, tokenizer, dtype


def add_lora(model, args):
    count = model.config.num_hidden_layers
    if args.layers > count:
        raise ValueError(f"--layers cannot exceed the model's {count} layers")
    targets = {name.rsplit('.', 1)[-1] for name, _ in model.named_modules()}
    if not {"q_proj", "v_proj"}.issubset(targets):
        raise ValueError("model does not expose the expected q_proj and v_proj LoRA targets")
    if args.four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    config = LoraConfig(
        task_type="CAUSAL_LM", r=args.rank, lora_alpha=args.alpha,
        lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        layers_pattern="layers", layers_to_transform=list(range(count - args.layers, count)),
    )
    return get_peft_model(model, config)


def load_adapter(adapter, base, device, four_bit=False, merge=False):
    config = PeftConfig.from_pretrained(adapter)
    model, tokenizer, _ = load_base(
        base or config.base_model_name_or_path, device, four_bit,
        tokenizer_source=adapter, merge=merge,
    )
    return PeftModel.from_pretrained(model, adapter), tokenizer
