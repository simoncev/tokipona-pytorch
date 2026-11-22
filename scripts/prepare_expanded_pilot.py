"""Build the frozen 50/12 source-grounded dataset; do not use final-test passages."""
import hashlib
import json
import math
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize(text):
    return ' '.join(text.replace('\u2019', "'").split())


def main():
    from transformers import AutoTokenizer
    from tokipona_torch.data import inspect_example

    root=Path('data/instructions/living_by_the_code')
    spec_path=root/'expanded_specs_v1.json'
    spec=json.loads(spec_path.read_text(encoding='utf-8'))
    sections_path=Path('data/prepared/living_by_the_code/sections.jsonl')
    sections={s['section_id']:s for s in map(json.loads,sections_path.read_text(encoding='utf-8').splitlines())}
    rows=[]
    for group in spec['groups']:
        section=sections[group['section_id']]
        if section['kind']!='interview' or section['split'] not in ('train','validation'):
            raise ValueError('Training and validation interview sources only')
        page=next(p for p in section['pages'] if p['pdf_page']==group['pdf_page'])
        normalized=normalize(page['text'])
        start=normalized.index(normalize(group['start']))
        end=normalized.index(normalize(group['end']),start) if group['end'] else len(normalized)
        passage=normalized[start:end].strip()
        for i,item in enumerate(group['positive']+[group['negative']]):
            kind='unanswerable' if i==len(group['positive']) else 'answerable'
            prompt=(spec['instruction']+'\n\nSource: interview with '+section['contributor']+
                '\nSource passage:\n'+passage+'\n\nQuestion: '+item['question'])
            rows.append(dict(id=f"expanded-{group['section_id']}-{i+1:02d}",split=section['split'],
                task_type=kind,passage=passage,question=item['question'],key_ideas=item['key_ideas'],
                messages=[dict(role='user',content=prompt),dict(role='assistant',content=item['answer'])],
                source=dict(document_id=section['document_id'],section_id=section['section_id'],
                    contributor=section['contributor'],group_id=section['group_id'],pdf_pages=[group['pdf_page']],
                    passage_sha256=digest(passage.encode())),approved_for_training=True,
                review_status='assistant_source_reviewed',
                review=dict(scope='Assistant checked target against selected excerpt; negative question is not answered by excerpt.',
                    rubric=spec['rubric'],rubric_sha256=digest(Path(spec['rubric']).read_bytes()))))
    assert len({r['id'] for r in rows})==62
    subsets={split:[r for r in rows if r['split']==split] for split in ('train','validation')}
    assert len(subsets['train'])==50 and len(subsets['validation'])==12
    assert not ({r['source']['group_id'] for r in subsets['train']} & {r['source']['group_id'] for r in subsets['validation']})
    tokenizer=AutoTokenizer.from_pretrained('models/SmolLM2-360M-Instruct',local_files_only=True)
    audit=[]
    for row in rows:
        view=inspect_example(row,tokenizer,8192)
        assert view['eos_supervised'] and view['supervised_tokens'] and not view['truncated']
        audit.append(dict(id=row['id'],tokens=view['original_tokens'],targets=view['supervised_tokens']))
    max_length=max(512,64*math.ceil(max(r['tokens'] for r in audit)/64))
    destination=root/'expanded_context_v1'
    destination.mkdir(exist_ok=True)
    for split,name in [('train','train.jsonl'),('validation','valid.jsonl')]:
        contents=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in subsets[split])
        path=destination/name
        if path.exists() and path.read_text(encoding='utf-8')!=contents:
            raise ValueError('Refusing to replace a modified expanded dataset')
        path.write_text(contents,encoding='utf-8')
    preparation=dict(spec_sha256=digest(spec_path.read_bytes()),source_sections_sha256=digest(sections_path.read_bytes()),
        rubric_sha256=digest(Path(spec['rubric']).read_bytes()),final_test_used=False,
        recommended_max_length=max_length,examples=audit,
        counts={split:dict(examples=len(subset),contributors=len({r['source']['group_id'] for r in subset}),
            passages=len({r['source']['passage_sha256'] for r in subset}),
            answerable=sum(r['task_type']=='answerable' for r in subset),
            unanswerable=sum(r['task_type']=='unanswerable' for r in subset)) for split,subset in subsets.items()})
    (destination/'preparation.json').write_text(json.dumps(preparation,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in preparation.items() if k!='examples'},indent=2))


if __name__=='__main__':
    main()
