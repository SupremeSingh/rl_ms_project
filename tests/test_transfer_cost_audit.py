"""Accounting tests: allocations, deduplication, tokens and conditional head budgets."""
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import pytest
ROOT=Path(__file__).resolve().parents[1]
def module():
    spec=importlib.util.spec_from_file_location('slice_cost',ROOT/'tools/transfer/cost_audit.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_slurm_does_not_double_count_typed_gpus_or_steps():
    m=module();text='101|100_0|COMPLETED|3600|8|cpu=8,gres/gpu=1,gres/gpu:a5000=1|start|end|\n101.batch|100_0.batch|COMPLETED|3600|8|gres/gpu=1|s|e|\n'
    rows=m.parse_sacct(text);assert len(rows)==1
    assert rows[0]['allocated_gpu_hours']==1 and rows[0]['allocated_cpu_hours']==8
    with pytest.raises(ValueError,match='Duplicate'):m.parse_sacct(text+text)


def test_reused_study_allocation_and_missing_accounting(tmp_path):
    m=module();study=tmp_path/'study';side=tmp_path/'study-submission';side.mkdir()
    (side/'jobs.tsv').write_text('generate\t100\nfit\t200\n')
    export=tmp_path/'sacct.txt';export.write_text('101|100_0|COMPLETED|1800|2|cpu=2,gres/gpu=1|s|e|\n102|100_1|COMPLETED|1800|2|cpu=2,gres/gpu=1|s|e|\n')
    ledger=m.job_ledger([study],export)
    assert ledger['total_allocated_gpu_hours']==1 and ledger['missing_job_ids']==['200']
    other=tmp_path/'other-submission';other.mkdir();(other/'jobs.tsv').write_text('generate\t100\n')
    with pytest.raises(ValueError,match='double count'):m.job_ledger([study,tmp_path/'other'],export)


def test_trajectory_tokens_retention_and_tamper(tmp_path):
    m=module();questions=[dict(id='q',domain='gsm8k',split='train',family='q',prompt_token_ids=[1,2])]
    (tmp_path/'questions.json').write_text(json.dumps(questions))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(questions_sha256=m.digest(tmp_path/'questions.json'),config=dict(prefixes_per_answer=8,responses_per_question=2))))
    payload=dict(question_id='q',responses=[dict(token_ids=[1,2,3],verifier_status='correct',capped=False),dict(token_ids=[1,2],verifier_status='parse_failure',capped=True)])
    record=dict(payload=payload,sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest())
    folder=tmp_path/'shards/0/trajectories';folder.mkdir(parents=True);p=folder/'q.json';p.write_text(json.dumps(record))
    result=m.trajectory_usage(tmp_path);r=result['groups']['gsm8k/train']
    assert r['response_tokens_including_eos']==5 and r['prompt_tokens_times_answers']==4
    assert r['answers']==2 and r['retained_answers']==1 and r['sampled_prefixes']==3
    subset=m.subset_usage(result['by_question'],'gsm8k/train',1);assert subset['response_tokens_including_eos']==5
    record['payload']['responses'][0]['token_ids'].append(4);p.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='Changed trajectory'):m.trajectory_usage(tmp_path)


def test_nested_subsets_and_no_unverifiable_question(tmp_path):
    m=module();records={q:dict(group='math/train',family=q,retained_question=int(q!='a'),answers=4,response_tokens_including_eos=n) for q,n in [('a',100),('b',200),('c',300)]}
    assert m.subset_usage(records,'math/train',1)['response_tokens_including_eos']==200
    assert m.subset_usage(records,'math/train',2)['response_tokens_including_eos']==500
    with pytest.raises(ValueError):m.subset_usage(records,'math/train',3)


def test_unavailable_scheduler_is_explicit(tmp_path):
    m=module();side=tmp_path/'study-submission';side.mkdir();(side/'jobs.tsv').write_text('generate\t100\n')
    with patch.object(m.subprocess,'run',side_effect=FileNotFoundError('sacct')):
        ledger=m.job_ledger([tmp_path/'study'])
    assert ledger['accounting']['kind']=='unavailable' and ledger['missing_job_ids']==['100']


def test_cancelled_unallocated_array_range_is_not_missing():
    m=module();rows=m.parse_sacct('200|200_[0-4]|CANCELLED|0|0||s|e|\n')
    assert len(rows)==1 and rows[0]['allocated_gpu_hours']==0


def test_full_audit_shares_collection_and_never_needs_model_loading(tmp_path):
    m=module();source=tmp_path/'source';study=tmp_path/'clarify'
    identity=dict(model_hashes={'toy':'same'},layers=[18],verifier={'toy':'same'})
    def collection(folder,entries,extra):
        folder.mkdir()
        qs=[dict(id=q,domain=d,split='train',family=q,prompt_token_ids=[1]) for q,d in entries]
        (folder/'questions.json').write_text(json.dumps(qs))
        manifest=dict(identity,questions_sha256=m.digest(folder/'questions.json'),config=dict(prefixes_per_answer=8,responses_per_question=2),**extra)
        (folder/'manifest.json').write_text(json.dumps(manifest))
        f=folder/'shards/0/trajectories';f.mkdir(parents=True)
        for q in qs:
            payload=dict(question_id=q['id'],responses=[dict(token_ids=[1,2],verifier_status='correct',capped=False),dict(token_ids=[1,2],verifier_status='incorrect',capped=False)])
            record=dict(payload=payload,sha256=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest())
            (f/(q['id']+'.json')).write_text(json.dumps(record))
    collection(source,[('gsm-a','gsm8k'),('gsm-b','gsm8k'),('num-a','numina')],{})
    collection(study,[('math-a','math'),('math-b','math')],dict(protocol='slice-transfer-clarify-v1',source=str(source),source_manifest_sha256=m.digest(source/'manifest.json'),seeds=[42]))
    (study/'summary.json').write_text(json.dumps(dict(selected_question_brier={f'{d}/{n}/{a}':.2 for d in ('gsm8k','math') for n in (1,2) for a in ('rms_two_thirds','transferred_slice','target_slice')})))
    (study/'packed').mkdir();(study/'packed/complete.json').write_text('{}')
    fit=study/'fits/42';fit.mkdir(parents=True);(fit/'models.pt').write_bytes(b'identity check only; no torch load')
    (fit/'locked.json').write_text(json.dumps(dict(manifest=m.digest(study/'manifest.json'),packed=m.digest(study/'packed/complete.json'),models=m.digest(fit/'models.pt'),source_training_questions=1)))
    record=m.build(study)
    assert len(record['studies'])==2
    assert record['original_collection']['gsm8k/train']['answers']==4
    assert record['new_math_collection']['math/train']['answers']==4
    assert len(record['head_adaptation_curves'])==12
    curves=[r for r in record['head_adaptation_curves'] if r['target']=='gsm8k' and r['head_questions']==2]
    assert all(r['response_tokens_including_eos']==8 for r in curves)
    out=tmp_path/'audit';m.write_report(record,out)
    assert 'INCOMPLETE ALLOCATION ACCOUNTING' in (out/'report.txt').read_text()
    assert .25<record['theoretical_head_resources']['matrix_storage_ratio']<.251
