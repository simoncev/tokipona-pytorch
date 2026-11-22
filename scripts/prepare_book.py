"""Build traceable sections and source-group splits from extracted page records.

No model inference or training. Run from the repository root.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def clean(text):
    # PDF discretionary hyphens occur as U+FFFE in this extraction.
    text = re.sub(r'(?<=\w)[\ufffe\uffff\u00ad]\s*\n\s*(?=\w)', '', text)
    text = text.replace('\ufffe', '').replace('\uffff', '').replace('\u00ad', '')
    lines = [line for line in text.splitlines() if not re.fullmatch(
        r'\s*(?:\d+ Living by the Code|An Interview with .+ \d+|\d+)\s*', line)]
    return '\n'.join(lines).strip()


def prepare(config_path, pages_path, output):
    config = json.loads(config_path.read_text(encoding='utf-8'))
    raw = pages_path.read_bytes()
    pages = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    assert [p['pdf_page'] for p in pages] == list(range(1, config['pdf_pages'] + 1)), 'Missing or duplicate page records'
    entries = config['entries']
    groups = [f'interview-{i:02d}' for i, entry in enumerate(entries) if entry['kind'] == 'interview']
    shuffled = groups.copy()
    random.Random(config['seed']).shuffle(shuffled)
    n_valid = max(1, round(len(groups) * .1))
    membership = {group: ('validation' if i < n_valid else 'test' if i < 2*n_valid else 'train')
                  for i, group in enumerate(shuffled)}
    sections = []
    group = None
    for i, entry in enumerate(entries):
        start = entry['printed_page'] + config['printed_page_offset']
        end = entries[i+1]['printed_page'] + config['printed_page_offset'] - 1 if i+1 < len(entries) else config['content_end_pdf_page']
        if entry['kind'] == 'interview':
            group = f'interview-{i:02d}'
        section_pages = pages[start-1:end]
        sections.append(dict(document_id=config['document_id'], section_id=f'section-{i:02d}',
            title=entry['title'], kind=entry['kind'], contributor=entry['title'] if entry['kind']=='interview' else None,
            group_id=group, split=membership[group], pdf_page_start=start, pdf_page_end=end,
            printed_page_start=entry['printed_page'], pages=[dict(pdf_page=p['pdf_page'], text=clean(p['text'])) for p in section_pages]))
    write_jsonl(output/'sections.jsonl', sections)
    manifest = dict(document_id=config['document_id'], seed=config['seed'],
        extraction_method='Uploaded-document text service; original PDF unavailable locally',
        pages_sha256=hashlib.sha256(raw).hexdigest(), page_count=len(pages),
        text_page_count=sum(bool(p['text'].strip()) for p in pages),
        no_text_pages=[p['pdf_page'] for p in pages if not p['text'].strip()],
        section_count=len(sections), interview_count=len(groups),
        groups=membership, policy='Interview plus following Further Reading kept together; split before example generation.',
        limitations=['No-text pages may be blank or image-only; not OCR-verified.',
                    'Section ends follow table-of-contents starts and may include dividers.',
                    'Further Reading author attribution requires review; never inherit interview contributor.',
                    'Draft examples require human review; no training-ready split is produced.'])
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k in ('page_count','text_page_count','section_count','interview_count')}))
    return sections, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('book_sources/living_by_the_code.json'))
    parser.add_argument('--pages', type=Path, default=Path('data/extracted/living_by_the_code/pages.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('data/prepared/living_by_the_code'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sections, manifest = prepare(args.config, args.pages, args.output)
    pilot_path = Path('data/instructions/living_by_the_code/pilot.draft.jsonl')
    if pilot_path.exists():
        by_id = {s['section_id']: s for s in sections}
        drafts = [json.loads(line) for line in pilot_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        assert len({row['id'] for row in drafts}) == len(drafts), 'Duplicate draft IDs'
        for row in drafts:
            section = by_id[row['source']['section_id']]
            assert row['source']['contributor'] == section['contributor']
            assert row['review_status'] == 'draft' and not row['approved_for_training']
            for page in row['source']['pdf_pages']:
                assert section['pdf_page_start'] <= page <= section['pdf_page_end']
                assert any(p['pdf_page'] == page and p['text'] for p in section['pages'])
            row['source']['group_id'] = section['group_id']
            row['split'] = section['split']
        for split in ('train', 'validation', 'test'):
            write_jsonl(args.output/'drafts'/f'{split}.draft.jsonl', [r for r in drafts if r['split'] == split])
        print(json.dumps({'draft_examples':len(drafts), 'draft_split_counts':{
            split:sum(r['split']==split for r in drafts) for split in ('train','validation','test')}}))
