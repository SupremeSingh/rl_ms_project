import json
import sys
from pathlib import Path
import pytest
from math_rl.reference_audit import reference_decision
from math_rl.provenance import sha256
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_math_references import audit


def test_multi_answer_and_missing_reference():
    q='Find all values of b. Enter all possible values separated by commas.'
    assert not reference_decision(q,r'First \boxed{-10879}, then \boxed{10879}.','10879')['eligible']
    assert not reference_decision(q,r'\boxed{10879}','10879')['eligible']
    assert not reference_decision('Q',r'\boxed{-10879,10879}','10879')['eligible']
    assert not reference_decision('Q',r'\boxed{\frac{1}{2}}','.5')['eligible']
    assert not reference_decision('Q',r'\boxed{3','3')['eligible']
    assert not reference_decision('Q',r'\boxed{4}','3')['eligible']
    assert reference_decision('How many values of k exist?',r'\boxed{4}','4')['eligible']
    assert reference_decision('Q',r'\boxed{1,000}','1000')['eligible']


def test_filtered_evaluation_is_paired_and_preserves_source(tmp_path):
    source=tmp_path/'source';(source/'data').mkdir(parents=True)
    questions=[dict(id='q0',question='Q',split='test',level=4,ground_truth='1',reference_solution=r'\boxed{1}'),
        dict(id='q1',question='Find all values of b',split='test',level=5,ground_truth='2',reference_solution=r'\boxed{-2}, \boxed{2}'),
        dict(id='q2',question='Q2',split='train',level=4,ground_truth='3',reference_solution=r'\boxed{4}')]
    snapshot=source/'data/questions.json';snapshot.write_text(json.dumps(dict(questions=questions)))
    manifest=dict(conditions={'grpo':{},'lstd':{}},seeds=[101,102],data=dict(questions_sha256=sha256(snapshot)))
    (source/'manifest.json').write_text(json.dumps(manifest))
    for method in manifest['conditions']:
        for seed in manifest['seeds']:
            folder=source/f'{method}-seed{seed}';folder.mkdir()
            for name in ['base-test.jsonl','final-test.jsonl']:
                scores=[int(method=='lstd' and name=='final-test.jsonl'),0]
                rows=[dict(id=q['id'],ground_truth=q['ground_truth'],score=s,verifier_status='correct' if s else 'incorrect') for q,s in zip(questions,scores)]
                (folder/name).write_text('\n'.join(map(json.dumps,rows)))
    original={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
    out=tmp_path/'audit';audit(source,out)
    result=json.loads((out/'summary.json').read_text())
    assert result['retained_test_questions']==1
    assert result['excluded_by_split']=={'test':1,'train':1}
    assert result['paired']['lstd_minus_grpo']['mean_pp']==100
    assert result['methods']['grpo']['base_accuracy']==0
    assert result['manual_reference_review_complete'] is False
    assert original=={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError): audit(source,out)
