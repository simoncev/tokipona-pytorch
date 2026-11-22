"""Observable CPU-only supervised pilot: python -m tokipona_torch.pilot."""
import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

from .data import chat_messages, inspect_example, load_dataset, read_rows


@dataclass
class PilotConfig:
    model: str = 'HuggingFaceTB/SmolLM2-360M-Instruct'
    train: str = 'data/instructions/living_by_the_code/reviewed/train.jsonl'
    valid: str = 'data/instructions/living_by_the_code/reviewed/valid.jsonl'
    output: str = 'runs/smollm2-pilot-001'
    steps: int = 10
    max_length: int = 256
    max_new_tokens: int = 80
    learning_rate: float = 0.0002
    layers: int = 4
    rank: int = 4
    alpha: int = 8
    threads: int = 4
    seed: int = 42
    eval_every: int = 5


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def batch_for(example):
    import torch
    return {key:torch.tensor([value], dtype=torch.long, device='cpu') for key,value in example.items()}


def validation_loss(model, dataset):
    """Mean NLL over supervised next-token targets, weighted by target count."""
    import torch
    model.eval()
    total, count = 0., 0
    with torch.inference_mode():
        for example in dataset:
            n = sum(label != -100 for label in example['labels'][1:])
            total += model(**batch_for(example)).loss.item()*n
            count += n
    return total/count


def generate_answer(model, tokenizer, messages, max_new_tokens, return_metadata=False):
    import torch
    inputs = tokenizer.apply_chat_template(chat_messages(messages, tokenizer), tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors='pt')
    inputs = {key:inputs[key] for key in ('input_ids','attention_mask')}
    if inputs['input_ids'].shape[1]+max_new_tokens > model.config.max_position_embeddings:
        raise ValueError('Generation exceeds model context window')
    model.eval()
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, use_cache=True)
    tokens=result[0, inputs['input_ids'].shape[1]:]
    text=tokenizer.decode(tokens,skip_special_tokens=True)
    if return_metadata:
        eos=bool(len(tokens) and tokens[-1].item()==tokenizer.eos_token_id)
        return dict(text=text,generated_tokens=len(tokens),ended_with_eos=eos,
            hit_token_limit=len(tokens)>=max_new_tokens and not eos)
    return text


def run_pilot(config, model_and_tokenizer=None):
    import psutil
    import torch
    from transformers import set_seed
    from .model import add_lora, load_base

    if min(config.steps,config.max_length,config.threads,config.max_new_tokens,config.layers,config.rank,config.alpha,config.eval_every) < 1 or config.learning_rate <= 0:
        raise ValueError('Pilot settings must be positive')
    if config.max_length < 2:
        raise ValueError('max_length must be at least 2')
    out = Path(config.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty output directory to preserve previous runs')
    train_rows = [row for _,row in read_rows(config.train)]
    valid_rows = [row for _,row in read_rows(config.valid)]
    if not train_rows or not valid_rows:
        raise ValueError('Pilot needs nonempty training and validation datasets')
    train_groups = {row['source']['group_id'] for row in train_rows}
    valid_groups = {row['source']['group_id'] for row in valid_rows}
    if train_groups & valid_groups:
        raise ValueError('Training and validation source groups overlap')
    for split, rows in [('train',train_rows),('validation',valid_rows)]:
        if any(row.get('approved_for_training') is not True or row.get('split')!=split for row in rows):
            raise ValueError('Use reviewed examples with explicit source split membership')
    torch.set_num_threads(config.threads)
    set_seed(config.seed)
    started = time.perf_counter()
    if model_and_tokenizer is None:
        model, tokenizer, _ = load_base(config.model, 'cpu')
    else:
        model, tokenizer = model_and_tokenizer
        if any(p.device.type != 'cpu' or (p.is_floating_point() and p.dtype != torch.float32) for p in model.parameters()):
            raise ValueError('Pilot requires CPU float32 model parameters')
    if config.max_length > model.config.max_position_embeddings:
        raise ValueError('max_length exceeds model context window')
    inspections = [{ 'id': row['id'], **inspect_example(row,tokenizer,config.max_length)} for row in train_rows+valid_rows]
    if any(row['truncated'] or not row['supervised_tokens'] or not row['eos_supervised'] for row in inspections):
        raise ValueError('Inspect examples: truncated answer, missing target tokens, or missing supervised EOS. Increase max_length if needed.')
    train_data = load_dataset(config.train,tokenizer,config.max_length,'response')
    valid_data = load_dataset(config.valid,tokenizer,config.max_length,'response')
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'token_inspection.json',inspections)
    # Fix development prompts before any weight updates; never load final test data.
    dev = [('train',row) for row in train_rows[:2]]+[('validation',row) for row in valid_rows]
    comparisons=[]
    for split,row in dev:
        record=generate_answer(model,tokenizer,row['messages'][:-1],config.max_new_tokens,return_metadata=True)
        comparisons.append(dict(id=row['id'],split=split,prompt=row['messages'][:-1],
            reference=row['messages'][-1]['content'],before=record['text'],before_generation=record))
        dump(out/'before_after.json',comparisons)
        print('Baseline generation:',row['id'],flush=True)
    initial_valid = validation_loss(model,valid_data)
    model = add_lora(model,SimpleNamespace(layers=config.layers,rank=config.rank,alpha=config.alpha,four_bit=False))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=config.learning_rate)
    manifest = dict(config=asdict(config), device='cpu',dtype='float32',batch_size=1,
        base_revision=getattr(model.config,'_commit_hash',None),
        dependencies={p:importlib.metadata.version(p) for p in ('torch','transformers','peft','accelerate')},
        dataset_sha256={k:hashlib.sha256(Path(p).read_bytes()).hexdigest() for k,p in [('train',config.train),('validation',config.valid)]},
        train_examples=len(train_data),validation_examples=len(valid_data),
        trainable_parameters=sum(p.numel() for p in parameters),total_parameters=sum(p.numel() for p in model.parameters()),
        initial_validation_loss=initial_valid,final_test_used=False)
    dump(out/'run.json',manifest)
    print(json.dumps(manifest,indent=2),flush=True)
    order, rng = list(range(len(train_data))), random.Random(config.seed)
    with (out/'loss.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['step','example_id','training_loss','validation_loss','supervised_tokens','elapsed_seconds','rss_mb'])
        writer.writeheader()
        for step in range(config.steps):
            if step%len(order)==0:
                rng.shuffle(order)
            index=order[step%len(order)]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss=model(**batch_for(train_data[index])).loss
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss; stopping pilot')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters,1.)
            optimizer.step()
            evaluation=validation_loss(model,valid_data) if (step+1)%config.eval_every==0 or step+1==config.steps else None
            row=dict(step=step+1,example_id=train_rows[index]['id'],training_loss=loss.item(),validation_loss=evaluation,
                supervised_tokens=sum(t!=-100 for t in train_data[index]['labels'][1:]),
                elapsed_seconds=round(time.perf_counter()-started,2),rss_mb=round(psutil.Process().memory_info().rss/2**20,1))
            writer.writerow(row);f.flush()
            print(json.dumps(row),flush=True)
    model.eval()
    model.config.use_cache=True
    model.save_pretrained(out/'adapter')
    tokenizer.save_pretrained(out/'adapter')
    for row in comparisons:
        record=generate_answer(model,tokenizer,row['prompt'],config.max_new_tokens,return_metadata=True)
        row['after']=record['text']
        row['after_generation']=record
        dump(out/'before_after.json',comparisons)
        print('Tuned generation:',row['id'],flush=True)
    manifest.update(final_validation_loss=evaluation,elapsed_seconds=time.perf_counter()-started)
    dump(out/'run.json',manifest)
    return manifest, comparisons


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name,field in PilotConfig.__dataclass_fields__.items():
        parser.add_argument('--'+name.replace('_','-'),type=field.type,default=field.default)
    args=parser.parse_args()
    run_pilot(PilotConfig(**vars(args)))


if __name__=='__main__':
    main()
