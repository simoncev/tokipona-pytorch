"""Generate every validation answer before/after an existing adapter; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    import torch
    from peft import PeftModel
    from transformers import set_seed
    from tokipona_torch.data import chat_messages, read_rows
    from tokipona_torch.model import load_base

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/smollm2-pilot-001'))
    args = parser.parse_args()
    run=json.loads((args.run/'run.json').read_text(encoding='utf-8'))
    config=run['config']
    destination=args.run/'evaluation'/'validation_comparison.json'
    if destination.exists():
        raise ValueError('Evaluation exists; preserve it or choose another run')
    torch.set_num_threads(config['threads'])
    set_seed(config['seed'])
    started=time.perf_counter()
    rows=[row for _,row in read_rows(config['valid'])]
    valid_hash=hashlib.sha256(Path(config['valid']).read_bytes()).hexdigest()
    if valid_hash != run['dataset_sha256']['validation']:
        raise ValueError('Validation dataset differs from the original run')
    if any(row.get('split')!='validation' for row in rows):
        raise ValueError('Expected validation data only')
    model,tokenizer,_=load_base(config['model'],'cpu')

    def generate(messages):
        inputs=tokenizer.apply_chat_template(chat_messages(messages,tokenizer),tokenize=True,
            add_generation_prompt=True,return_dict=True,return_tensors='pt')
        inputs={k:inputs[k] for k in ('input_ids','attention_mask')}
        model.eval()
        with torch.inference_mode():
            ids=model.generate(**inputs,do_sample=False,max_new_tokens=config['max_new_tokens'],
                pad_token_id=tokenizer.pad_token_id,use_cache=True)[0,inputs['input_ids'].shape[1]:]
        stopped=bool(len(ids) and ids[-1].item()==tokenizer.eos_token_id)
        return dict(text=tokenizer.decode(ids,skip_special_tokens=True),generated_tokens=len(ids),
            ended_with_eos=stopped,hit_token_limit=len(ids)>=config['max_new_tokens'] and not stopped)

    comparisons=[]
    for row in rows:
        comparisons.append(dict(id=row['id'],source=row['source'],prompt=row['messages'][:-1],
            reference=row['messages'][-1]['content'],before=generate(row['messages'][:-1])))
    model=PeftModel.from_pretrained(model,args.run/'adapter')
    for comparison in comparisons:
        comparison['after']=generate(comparison['prompt'])
        comparison['identical_text']=comparison['before']['text']==comparison['after']['text']
    destination.parent.mkdir(parents=True,exist_ok=True)
    result=dict(run=str(args.run),method='CPU float32; fixed prompts; greedy decoding; original run token limit',
        validation_sha256=valid_hash,
        max_new_tokens=config['max_new_tokens'],weights_updated=False,final_test_used=False,
        elapsed_seconds=time.perf_counter()-started,comparisons=comparisons)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2),flush=True)


if __name__=='__main__':
    main()
