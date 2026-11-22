"""JSONL ingestion and causal language model labels."""

import json
from pathlib import Path
import re

IGNORE = -100
GEMMA_MODEL_TURN = re.compile(r"<start_of_turn>model\r?\n")


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            yield number, row


def text_field(row, name):
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name!r} must be a non-empty string")
    return value


def chat_messages(messages, tokenizer=None):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    turns = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        turns.append({"role": message.get("role"), "content": text_field(message, "content")})
    if turns[0]["role"] == "system" and tokenizer is not None and "<start_of_turn>" in str(tokenizer.chat_template):
        system = turns.pop(0)["content"]
        if not turns or turns[0]["role"] != "user":
            raise ValueError("the system message must be followed by a user turn")
        turns[0]["content"] = system + "\n\n" + turns[0]["content"]
    start = 1 if turns[0]['role'] == 'system' else 0
    if start and len(turns) == 1:
        raise ValueError("the system message must be followed by a user turn")
    for index, turn in enumerate(turns[start:]):
        expected = "user" if index % 2 == 0 else "assistant"
        if turn["role"] != expected:
            raise ValueError(f"expected {expected!r} at turn {index + 1}")
    return turns


def render_example(row, tokenizer, loss_mode):
    boundary = 0
    add_special_tokens = True
    if "messages" in row:
        turns = chat_messages(row["messages"], tokenizer)
        if turns[-1]["role"] != "assistant":
            raise ValueError("training messages must end with an assistant turn")
        full = tokenizer.apply_chat_template(turns, tokenize=False)
        if loss_mode == "response":
            prompt = tokenizer.apply_chat_template(
                turns[:-1], tokenize=False, add_generation_prompt=True
            )
            if not full.startswith(prompt):
                raise ValueError("chat template does not preserve the generation prefix")
            boundary = len(prompt)
        add_special_tokens = False
    elif "prompt" in row or "completion" in row:
        prompt = text_field(row, "prompt")
        full = prompt + text_field(row, "completion")
        if loss_mode == "response":
            boundary = len(prompt)
    else:
        full = text_field(row, "text")
        if loss_mode == "response":
            turns = list(GEMMA_MODEL_TURN.finditer(full))
            if not turns or not full[turns[-1].end():].removesuffix("<end_of_turn>").strip():
                raise ValueError("response loss requires messages, prompt/completion, or a Gemma model turn")
            boundary = turns[-1].end()

    if tokenizer.bos_token and full.startswith(tokenizer.bos_token):
        add_special_tokens = False
    return full, boundary, add_special_tokens


def encode(row, tokenizer, max_length, loss_mode):
    full, boundary, add_special_tokens = render_example(row, tokenizer, loss_mode)
    tokens = tokenizer(
        full, add_special_tokens=add_special_tokens, truncation=True, max_length=max_length,
        return_offsets_mapping=bool(boundary),
    )
    ids = tokens["input_ids"]
    labels = ids.copy()
    if boundary:
        for index, (start, end) in enumerate(tokens["offset_mapping"]):
            if start < boundary or start == end:
                labels[index] = IGNORE
    if not any(label != IGNORE for label in labels[1:]):
        return None
    return {"input_ids": ids, "attention_mask": tokens["attention_mask"], "labels": labels}


def inspect_example(row, tokenizer, max_length=256):
    """Expose the exact training labels, including end markers and truncation."""
    full, boundary, add_special_tokens = render_example(row, tokenizer, 'response')
    untruncated = tokenizer(full, add_special_tokens=add_special_tokens)['input_ids']
    encoded = encode(row, tokenizer, max_length, 'response')
    tokens = []
    if encoded:
        for i, (token_id, label) in enumerate(zip(encoded['input_ids'], encoded['labels'])):
            tokens.append(dict(position=i, token_id=token_id, piece=tokenizer.convert_ids_to_tokens(token_id),
                decoded=tokenizer.decode([token_id]), attention_mask=encoded['attention_mask'][i],
                label=label, contributes_to_loss=i > 0 and label != IGNORE))
    return dict(rendered_chat=full, response_character_boundary=boundary,
        original_tokens=len(untruncated), retained_tokens=len(tokens), truncated=len(untruncated)>max_length,
        supervised_tokens=sum(t['contributes_to_loss'] for t in tokens), tokens=tokens,
        target_text=tokenizer.decode([t['token_id'] for t in tokens if t['contributes_to_loss']]),
        eos_supervised=any(t['token_id']==tokenizer.eos_token_id and t['contributes_to_loss'] for t in tokens))


def load_dataset(path, tokenizer, max_length, loss_mode):
    examples = []
    dropped = 0
    for number, row in read_rows(path):
        if row.get('approved_for_training') is False:
            raise ValueError(f"{path}:{number}: draft example is not approved for training; use reviewed data")
        try:
            example = encode(row, tokenizer, max_length, loss_mode)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if example is None:
            dropped += 1
        else:
            examples.append(example)
    if not examples:
        raise ValueError(f"{path}: no examples with usable target tokens")
    print(f"{path}: {len(examples)} examples, {dropped} skipped with no usable targets")
    return examples


def split_jsonl(source, destination):
    rows = [row for _, row in read_rows(source)]
    if len(rows) < 2:
        raise ValueError("splitting requires at least two JSONL rows")
    boundary = max(1, int(len(rows) * 0.9))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    train, valid = destination / "train.jsonl", destination / "valid.jsonl"
    if train.exists() or valid.exists():
        raise ValueError("split output already exists; choose a new directory")
    for path, subset in ((train, rows[:boundary]), (valid, rows[boundary:])):
        with path.open("x", encoding="utf-8") as output:
            for row in subset:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{path}: {len(subset)} examples")
