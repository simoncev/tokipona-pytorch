"""Read-only CPU controls: output length, instruction wording, and source context."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    import torch
    from transformers import set_seed
    from peft import PeftModel
    from tokipona_torch.model import load_base
    from tokipona_torch.data import read_rows, chat_messages

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/smollm2-pilot-001'))
    args=parser.parse_args()
    manifest=json.loads((args.run/'run.json').read_text(encoding='utf-8'))
    config=manifest['config']
    output=args.run/'evaluation/context_control.json'
    if output.exists():
        raise ValueError('Preserve the existing context evaluation before another run')
    source_path=Path('data/prepared/living_by_the_code/sections.jsonl')
    sections={s['section_id']:s for s in map(json.loads,source_path.read_text(encoding='utf-8').splitlines())}
    if hashlib.sha256(Path(config['valid']).read_bytes()).hexdigest()!=manifest['dataset_sha256']['validation']:
        raise ValueError('Validation examples differ from original run')
    rows=[row for _,row in read_rows(config['valid'])]
    # Select complete, relevant Q/A passages by explicit textual boundaries.
    boundaries={
        'lbtc-pilot-007':('One thing that helped me grow as a professional','What advice would you give to people that are already'),
        'lbtc-pilot-008':('Get better at communication.','Let’s talk about other ways to learn.'),
        'lbtc-pilot-009':('What is your leadership philosophy?','Are there any trends in software development'),
    }
    cases=[]
    instruction='Answer the question in two concise sentences. Do not invent author-specific details. If the source passage is insufficient or unavailable, say so.'
    for row in rows:
        if row['split']!='validation':
            raise ValueError('Validation examples only')
        section=sections[row['source']['section_id']]
        text='\n'.join(p['text'] for p in section['pages'] if p['pdf_page'] in row['source']['pdf_pages'])
        start,end=boundaries[row['id']]
        passage=text[text.index(start):text.index(end)].strip()
        question=row['messages'][0]['content']
        prompts={
            'question_only':question,
            'instruction_only':instruction+'\n\nSource passage: [not provided]\n\nQuestion: '+question,
            'passage_and_instruction':instruction+'\n\nSource passage:\n'+passage+'\n\nQuestion: '+question,
        }
        cases.append(dict(id=row['id'],source=row['source'],passage=passage,
            passage_sha256=hashlib.sha256(passage.encode()).hexdigest(),reference=row['messages'][-1]['content'],
            conditions={name:dict(prompt=prompt) for name,prompt in prompts.items()}))
    result=dict(run=str(args.run),weights_updated=False,final_test_used=False,max_new_tokens=192,
        device='cpu',dtype='float32',decoding='greedy',
        source_sections_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        cases=cases,status='in_progress')
    output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    torch.set_num_threads(config['threads']);set_seed(config['seed'])
    started=time.perf_counter()
    model,tokenizer,_=load_base(config['model'],'cpu')
    for phase in ('base','adapter'):
        if phase=='adapter':
            model=PeftModel.from_pretrained(model,args.run/'adapter')
        model.eval()
        for case in cases:
            for name,condition in case['conditions'].items():
                inputs=tokenizer.apply_chat_template(chat_messages([{'role':'user','content':condition['prompt']}],tokenizer),
                    tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True)
                inputs={k:inputs[k] for k in ('input_ids','attention_mask')}
                if inputs['input_ids'].shape[1]+192>model.config.max_position_embeddings:
                    raise ValueError('Context window exceeded')
                with torch.inference_mode():
                    tokens=model.generate(**inputs,do_sample=False,max_new_tokens=192,
                        pad_token_id=tokenizer.pad_token_id,use_cache=True)[0,inputs['input_ids'].shape[1]:]
                eos=bool(len(tokens) and tokens[-1].item()==tokenizer.eos_token_id)
                condition[phase]=dict(text=tokenizer.decode(tokens,skip_special_tokens=True),
                    generated_tokens=len(tokens),ended_with_eos=eos,hit_token_limit=len(tokens)>=192 and not eos)
                save()
                print(case['id'],name,phase,'tokens',len(tokens),'EOS',eos,flush=True)
    result.update(status='complete',elapsed_seconds=time.perf_counter()-started)
    save()
    print('Saved:',output,flush=True)


if __name__=='__main__':
    main()
