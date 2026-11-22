"""Add cited source passages to the reviewed pilot; preserve targets and splits."""
import hashlib
import json
import math
from pathlib import Path

INSTRUCTION = ('Answer the question concisely using only the source passage. '
    'Do not invent author-specific details. If the source passage is insufficient or unavailable, say so.')
BOUNDARIES = {
    '001': ('What is your definition of a good leader?', None),
    '002': ('Developer Relations, or Developer Experience', 'Describe your work-life balance.'),
    '003': ('Many times', 'What are the three tools'),
    '004': ('Part of my secret', "You\u2019re the founder"),
    '005': ('One of the things that I', 'What is your daily motivation'),
    '006': ('When I first started developing apps', '\u201cThe best way to learn'),
    '007': ('One thing that helped me grow as a professional', 'What advice would you give to people that are already'),
    '008': ('Get better at communication.', 'Let\u2019s talk about other ways to learn.'),
    '009': ('What is your leadership philosophy?', 'Are there any trends in software development'),
    '010': ('A few years ago, you started working on your book', 'You recently quit your work'),
}


def normalize(text):
    return ' '.join(text.split())


def build_rows(rows, sections):
    output=[]
    for original in rows:
        row=json.loads(json.dumps(original))
        if row.get('approved_for_training') is not True or row['split'] not in ('train','validation'):
            raise ValueError('Only reviewed training and validation examples are allowed')
        section=sections[row['source']['section_id']]
        if section['split']!=row['split'] or section['group_id']!=row['source']['group_id']:
            raise ValueError('Source split or group mismatch')
        text=normalize('\n'.join(p['text'] for p in section['pages'] if p['pdf_page'] in row['source']['pdf_pages']))
        start,end=BOUNDARIES[row['id'].rsplit('-',1)[-1]]
        begin=text.index(normalize(start))
        finish=text.index(normalize(end),begin) if end else len(text)
        passage=text[begin:finish].strip()
        question=row['messages'][0]['content']
        row['parent_id']=row['id']
        row['id']=row['id'].replace('pilot','context')
        row['passage']=passage
        row['question']=question
        row['source']['passage_sha256']=hashlib.sha256(passage.encode()).hexdigest()
        row['messages'][0]['content']=(INSTRUCTION+'\n\nSource: interview with '+row['source']['contributor']+
            '\nSource passage:\n'+passage+'\n\nQuestion: '+question)
        row['review']['context_review']='Relevant Q/A excerpt selected from cited pages; whitespace normalized. Previously reviewed answer retained unchanged.'
        output.append(row)
    return output


def main():
    from transformers import AutoTokenizer
    from tokipona_torch.data import inspect_example
    source=Path('data/prepared/living_by_the_code/sections.jsonl')
    sections={row['section_id']:row for row in map(json.loads,source.read_text(encoding='utf-8').splitlines())}
    destination=Path('data/instructions/living_by_the_code/context_reviewed_v2')
    prepared={}
    for split in ('train','valid'):
        path=Path('data/instructions/living_by_the_code/reviewed')/(split+'.jsonl')
        prepared[split]=build_rows([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()],sections)
    assert not ({r['source']['group_id'] for r in prepared['train']} & {r['source']['group_id'] for r in prepared['valid']})
    tokenizer=AutoTokenizer.from_pretrained('models/SmolLM2-360M-Instruct',local_files_only=True)
    audit=[]
    for rows in prepared.values():
        for row in rows:
            view=inspect_example(row,tokenizer,8192)
            audit.append(dict(id=row['id'],tokens=view['original_tokens'],targets=view['supervised_tokens'],eos_supervised=view['eos_supervised']))
            assert view['eos_supervised'] and view['supervised_tokens'] and not view['truncated']
    max_length=max(512,64*math.ceil(max(r['tokens'] for r in audit)/64))
    destination.mkdir(parents=True,exist_ok=True)
    for split,rows in prepared.items():
        path=destination/(split+'.jsonl')
        contents=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows)
        if path.exists() and path.read_text(encoding='utf-8')!=contents:
            raise ValueError('Refusing to replace changed context dataset')
        path.write_text(contents,encoding='utf-8')
    audit_report=dict(source_sections_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        train_examples=len(prepared['train']),validation_examples=len(prepared['valid']),final_test_used=False,
        recommended_max_length=max_length,examples=audit)
    (destination/'preparation.json').write_text(json.dumps(audit_report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit_report,indent=2))


if __name__=='__main__':
    main()
