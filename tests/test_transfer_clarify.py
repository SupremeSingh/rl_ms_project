"""Protocol/linear algebra acceptance; GPU/data ingestion covered by explicit smoke."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/transfer'))


def module(name='clarify'):
    spec=importlib.util.spec_from_file_location('clarify_'+name,ROOT/'tools/transfer'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_wide_grid_and_spectral_solution_matches_solve():
    m=module();torch.set_num_threads(2);torch.manual_seed(42)
    x=torch.randn(80,6);y=(x[:,0]>0).float()
    heads,best,scores=m.ridge_grid(x,y,x,y,[str(i) for i in range(len(x))])
    assert set((.01,1,10,100,1000,10000))<=set(heads)
    assert best!=max(heads)
    for eps,h in heads.items():
        z=((x-h['mean'])/h['scale']).double();t=torch.cat([z,torch.ones(len(z),1,dtype=torch.float64)],1)
        w=torch.linalg.solve(t.T@t+eps*torch.eye(t.shape[1]),t.T@y.double())
        assert torch.allclose(h['w'],w,atol=1e-7,rtol=1e-7)
    assert best==min(scores,key=scores.get)


def test_normalization_uses_only_train_features():
    m=module();torch.manual_seed(7)
    x=torch.randn(30,4);y=(x[:,0]>0).float()
    a,_,_=m.ridge_grid(x,y,x,y,[str(i) for i in range(30)])
    b,_,_=m.ridge_grid(x,y,x+100,y,[str(i) for i in range(30)])
    assert torch.equal(a[.01]['mean'],x.mean(0))
    assert torch.equal(a[.01]['mean'],b[.01]['mean'])


def test_nested_question_subsets_keep_all_prefixes():
    m=module();d=dict(x=torch.randn(8,3),y=torch.tensor([0.,1.]*4),question=['b','b','a','a','d','d','c','c'],position=list(range(8)))
    order=m.ordering(d);assert order==['a','b','c','d']
    a=m.subset(d,set(order[:1]));b=m.subset(d,set(order[:2]))
    assert a['question']==['a','a'] and len(b['question'])==4
    assert set(a['question'])<set(b['question'])


def test_math_numeric_eligibility_and_no_fake_500_denominator():
    m=module();rows,ex=m.math_rows([dict(problem='Add',solution=r'\boxed{7}',answer='7'),dict(problem='Point',solution=r'\boxed{(2,3)}',answer='(2,3)')])
    assert len(rows)==1 and len(ex)==1 and rows[0]['answer']=='7'


def test_no_test_reads_while_fitting(tmp_path):
    m=module();reads=[]
    def data(out,domain,split):
        assert split!='test';reads.append((domain,split))
        return dict(x=torch.zeros(2,9216),y=torch.tensor([0.,1.]),question=['a','b'],position=[0,1])
    with patch.object(m,'verify',return_value={'budgets':[100],'val_questions':20}),patch.object(m,'data',data):
        with pytest.raises(ValueError,match='Too few source'):m.fit(tmp_path,0,'cpu')
    assert reads==[('numina','train'),('numina','val')]


def test_submit_full_stops_at_validation_and_preserves_path(tmp_path):
    m=module('submit_clarify');smoke=tmp_path/'smoke';(smoke/'status').mkdir(parents=True)
    import json
    (smoke/'manifest.json').write_text(json.dumps(dict(smoke=True,source=str(tmp_path/'alias-source'),code={})))
    (smoke/'status/evaluate-0.json').write_text(json.dumps(dict(state='complete')))
    calls=[]
    class Result:
        returncode=0;stderr=''
        def __init__(self,n):self.stdout=str(n)+';cluster'
    def run(cmd,**kw):calls.append(cmd);return Result(100+len(calls))
    with patch.object(m.subprocess,'run',run):m.submit(tmp_path/'full',tmp_path/'alias-source','full',smoke,4,4)
    assert len(calls)==5 and 'evaluate' not in calls[-1]
    assert calls[-1][-2:] == ['validate',str(tmp_path/'full')]
    assert '--array=0-4%4' in calls[3]
    assert '--dependency=afterok:104' in calls[-1]
    assert calls[0][calls[0].index('--source')+1]==str(tmp_path/'alias-source')


def test_ambiguous_submission_is_not_retried(tmp_path):
    m=module('submit_clarify')
    with patch.object(m.subprocess,'run',side_effect=m.subprocess.TimeoutExpired('sbatch',90)) as run:
        with pytest.raises(m.subprocess.TimeoutExpired):m.submit(tmp_path/'smoke',tmp_path/'source','smoke')
        assert run.call_count==1
    with pytest.raises(FileExistsError):m.submit(tmp_path/'smoke',tmp_path/'source','smoke')


def test_score_reports_clipped_and_raw_question_metrics():
    m=module();pred=torch.tensor([-1.,.2,2.]).double();y=torch.tensor([0.,0.,1.]);r=m.score(pred,y,['a','a','b'])
    assert r['question_clipped_brier']<=r['question_brier']
    assert 0<=r['ece_10_bins']<=1


def test_regularization_ceiling_blocks_test_selection():
    m=module();x=torch.zeros(4,2);y=torch.ones(4);vy=torch.zeros(4)
    with patch.object(m,'GRID',(.01,1.)):
        with pytest.raises(ValueError,match='ceiling 1e10'):m.ridge_grid(x,y,x,vy,['a','b','c','d'])


def test_evaluation_writes_selected_and_fixed_metrics(tmp_path):
    m=module();torch.set_num_threads(2)
    from slice_critic.storage import write_json,read_json
    d=dict(x=torch.zeros(8,9216),y=torch.tensor([0.,1.]*4),question=['a','a','b','b','c','c','d','d'],position=list(range(8)))
    selection={};heads={}
    for domain in ('gsm8k','math'):
        for arm in m.ARMS:
            key=f'{domain}/100/{arm}';width=1536 if arm=='rms_two_thirds' else 3
            h=dict(w=torch.cat([torch.zeros(width),torch.tensor([.5])]).double(),mean=torch.zeros(width),scale=torch.ones(width),condition=1.)
            heads[key]={.01:h,1.:h};selection[key]=dict(epsilon=.01)
    for seed in m.SEEDS:write_json(tmp_path/'fits'/str(seed)/'locked.json',dict(selection=selection))
    write_json(tmp_path/'packed/complete.json',dict(coverage={}))
    state=dict(representations={'source':{},'gsm8k':{},'math':{}},heads=heads)
    with patch.object(m,'verify',return_value={'budgets':[100],'scope':'toy'}),patch.object(m,'validate'),patch.object(m,'data',return_value=d),patch.object(m,'load_tensor',return_value=state),patch.object(m.base,'transform',side_effect=lambda r,x:torch.zeros(len(x),3)):
        m.evaluate(tmp_path)
    result=read_json(tmp_path/'summary.json')
    assert len(result['fixed_epsilon_metrics'])==2*3*5*2
    assert len(result['paired'])==4
    assert all(abs(v-.25)<1e-8 for v in result['selected_question_brier'].values())
    assert result['constant_baselines']['math/100']['question_brier']==.25
