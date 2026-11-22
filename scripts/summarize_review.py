"""Aggregate explicit assistant review judgments; this script does not judge text."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/smollm2-expanded-001'))
    args=parser.parse_args()
    run=json.loads((args.run/'run.json').read_text(encoding='utf-8'))
    data={r['id']:r for r in map(json.loads,Path(run['config']['valid']).read_text(encoding='utf-8').splitlines())}
    comparisons={r['id']:r for r in json.loads((args.run/'before_after.json').read_text(encoding='utf-8')) if r['split']=='validation'}
    rubric=Path('docs/evaluation_rubric_v1.md')
    if any(r['review']['rubric_sha256']!=hashlib.sha256(rubric.read_bytes()).hexdigest() for r in data.values()):
        raise ValueError('Rubric changed after dataset preparation')
    phases={}
    for phase in ('before','after'):
        review=json.loads((args.run/f'review_{phase}.json').read_text(encoding='utf-8'))
        grades={r['id']:r for r in review['grades']}
        if len(grades)!=len(review['grades']) or set(grades)!=set(data) or set(comparisons)!=set(data):
            raise ValueError('Review must cover every validation ID exactly once')
        for example_id,grade in grades.items():
            assert grade['support'] in (0,1,2) and grade['coverage'] in (0,1,2)
            if not comparisons[example_id][phase].strip():
                grade['passed']=False
            elif data[example_id]['task_type']=='unanswerable':
                grade['passed']=grade['proper_abstention'] and not grade['material_unsupported_claim']
            else:
                grade['passed']=grade['support']==2 and grade['coverage']>=1 and not grade['material_unsupported_claim']
        phases[phase]=grades
    summary={phase:{kind:dict(passed=sum(grades[i]['passed'] for i,r in data.items() if r['task_type']==kind),
        total=sum(r['task_type']==kind for r in data.values())) for kind in ('answerable','unanswerable')}
        for phase,grades in phases.items()}
    summary.update(improved=[i for i in data if not phases['before'][i]['passed'] and phases['after'][i]['passed']],
        regressed=[i for i in data if phases['before'][i]['passed'] and not phases['after'][i]['passed']],
        changed_text=[i for i in data if comparisons[i]['before']!=comparisons[i]['after']],
        rubric_sha256=hashlib.sha256(rubric.read_bytes()).hexdigest(),reviewer='assistant',final_test_used=False)
    (args.run/'review_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (args.run/'review_details.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['id','contributor','task_type','question','reference','before','after','before_pass','after_pass','before_reason','after_reason'])
        for i,row in data.items():
            writer.writerow([i,row['source']['contributor'],row['task_type'],row['question'],row['messages'][-1]['content'],
                comparisons[i]['before'],comparisons[i]['after'],phases['before'][i]['passed'],phases['after'][i]['passed'],
                phases['before'][i]['rationale'],phases['after'][i]['rationale']])
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
