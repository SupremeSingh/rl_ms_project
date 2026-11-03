"""CPU protocol tests; real Qwen/Math-Verify are exercised by the cluster smoke."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
def module(name):
    path=ROOT/'tools/transfer'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('transfer_'+name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_references_and_overlap():
    m=module('study')
    rows,ex=m.normalize_rows([dict(question='What is 2+3?',answer='work #### 5'),
                             dict(question='bad',answer='5')],'gsm8k')
    assert len(rows)==1 and ex['unsupported_reference']==1
    assert m.exclude(rows,{m.family_key('What is 8+9?')})==[]
    rows,ex=m.normalize_rows([dict(problem='q',solution=r'\boxed{5}',source='gsm8k'),
                             dict(problem='q',solution=r'\boxed{6}',source='math'),
                             dict(problem='q',solution=r'\boxed{7}',source='olympiads')],'numina')
    assert len(rows)==1 and rows[0]['answer']=='7' and ex['held_out_source_tag']==2


def test_question_weighting():
    m=module('study')
    pred=torch.tensor([0.,0.,1.]);y=torch.ones(3)
    assert np.allclose(m.question_brier(pred,y,['a','a','b']),[1,0])
    assert m.question_brier(pred,y,['a','a','b']).mean()==.5


def test_pool_matches_question_budget():
    m=module('study')
    def data(prefix):return dict(x=torch.randn(12,4),y=torch.tensor([0.,1.]*6),question=[prefix+str(i//2) for i in range(12)],position=list(range(12)))
    a,b=data('a'),data('b');p=m.pool(a,b,6,42)
    assert len(set(p['question']))==6
    assert sum(q.startswith('a') for q in set(p['question']))==3
    assert m.pool(a,b,6,42)['question']==p['question']


def test_ridge_fit_and_frozen_transform():
    m=module('study');torch.set_num_threads(2);torch.manual_seed(42)
    x=torch.randn(40,8);y=(x[:,0]>0).float()
    data=dict(x=x,y=y)
    record=m.fit_features(data,data,5,42,2,'cpu')
    z=m.transform(record,x)
    assert z.shape==(40,5) and not z.requires_grad
    head=m.ridge_fit(z,y,z,y,[str(i) for i in range(40)])
    assert head['epsilon'] in m.EPSILONS
    assert torch.isfinite(m.ridge_predict(head,z)).all()
    assert torch.equal(head['mean'],z.mean(0))


def test_combiner_output_and_no_input_gradients():
    m=module('study');torch.set_num_threads(2)
    x=torch.randn(8,1536);data=dict(x=x,y=torch.tensor([0.,1.]*4))
    record=m.fit_features(data,data,768,42,1,'cpu',True)
    z=m.transform(record,x)
    assert z.shape==(8,768) and not z.requires_grad and x.grad is None


def test_submission_dependencies_and_timeout_stop(tmp_path):
    m=module('submit');calls=[]
    class Result:
        returncode=0;stderr=''
        def __init__(self,n):self.stdout=str(n)+';cluster\n'
    def run(cmd,**kwargs):calls.append(cmd);return Result(100+len(calls))
    with patch.object(m.subprocess,'run',run):m.submit(tmp_path/'smoke','smoke',4,2)
    assert len(calls)==5
    assert '--dependency=afterok:101' in calls[1]
    assert '--array=0-3%2' in calls[1] and '--gres=gpu:a5000:1' in calls[1]
    assert '--array=0-2%3' in calls[3]
    assert '--dependency=afterok:104' in calls[4]
    with patch.object(m.subprocess,'run',side_effect=m.subprocess.TimeoutExpired('sbatch',90)) as r:
        with pytest.raises(m.subprocess.TimeoutExpired):m.submit(tmp_path/'ambiguous','smoke')
        assert r.call_count==1
    with pytest.raises(FileExistsError):m.submit(tmp_path/'ambiguous','smoke')


def test_fit_never_loads_test(tmp_path):
    m=module('study');reads=[]
    # Stop immediately after loading train/validation inputs, before expensive fitting.
    def read(out,domain,split):
        reads.append((domain,split));assert split!='test'
        return dict(x=torch.zeros(2,9216),y=torch.tensor([0.,1.]),question=['a','b'],position=[0,0])
    with patch.object(m,'verify',return_value={}),patch.object(m,'data_split',read),patch.object(m,'pool',side_effect=RuntimeError('stop')):
        with pytest.raises((RuntimeError,KeyError)):m.fit(tmp_path,0,'cpu')
    assert set(reads)=={('numina','train'),('numina','val'),('gsm8k','train'),('gsm8k','val')}


def test_saved_shard_pipeline_with_toy_backend(tmp_path):
    m=module('study')
    from slice_critic.config import TrainingConfig
    from slice_critic.storage import write_json,read_json
    class Backend:
        hidden_size=1536;model_files={'toy':'identity'}
        def __init__(self,*args):pass
        def generate(self,prompt,count,max_tokens,seed):
            return [dict(token_ids=[1,2,3],response=str(i%2),capped=False) for i in range(count)]
        def trajectory_features(self,prompt,answer,positions,layers):
            return torch.ones(len(positions),len(layers)*1536)*prompt[0]
    class Grader:
        provenance={'kind':'toy'}
        def __init__(self,*args):pass
        def check(self):pass
        def __call__(self,responses,reference):
            return [dict(status='correct' if int(r) else 'incorrect',diagnostic_reward=int(r)) for r in responses]
    questions=[]
    for domain in ('numina','gsm8k'):
        for s in ('train','val','test'):
            questions.append(dict(domain=domain,split=s,id=domain+'-'+s,prompt_token_ids=[1],reference='1'))
    c=TrainingConfig(dataset='toy',responses_per_question=2)
    manifest=dict(shards=1,verifier={'kind':'toy'},config=c.__dict__,model_path='unused',model_hashes={'toy':'identity'},layers=[27,26,24,20,12,18])
    write_json(tmp_path/'manifest.json',manifest);write_json(tmp_path/'questions.json',questions)
    with patch.object(m,'verify',return_value=manifest),patch.object(m,'TransformersBackend',Backend),patch.object(m,'VerificationClient',Grader):
        m.generate(tmp_path,0)
        m.generate(tmp_path,0)  # Same generation/feature caches are resumable.
        m.pack(tmp_path)
    for d in ('numina','gsm8k'):
        for s in ('train','val','test'):
            record=m.data_split(tmp_path,d,s)
            assert record['x'].shape==(6,9216)
            assert set(record['y'].tolist())=={0.,1.}
    done=read_json(tmp_path/'shards/0/complete.json')
    assert done['label_statuses']['gsm8k/test/correct']==1


def test_verifier_preflight_blocks_before_model_or_output(tmp_path):
    from types import SimpleNamespace
    m=module('study')
    args=SimpleNamespace(out=tmp_path/'new',model=None,tokens=256,answers=4,epochs=2,
                         verifier_python='/isolated-verifier/bin/python')
    class BadGrader:
        def __init__(self,python):assert python==args.verifier_python
        def check(self):raise RuntimeError('missing pinned verifier')
    with patch.object(m,'VerificationClient',BadGrader),patch.object(m,'resolve_model') as model:
        with pytest.raises(RuntimeError,match='missing pinned verifier'):m.prepare(args)
        model.assert_not_called()
    assert not args.out.exists()


def test_launcher_uses_isolated_verifier():
    source=(ROOT/'tools/transfer/run.sh').read_text()
    assert '--verifier-python "$PWD/.venv-ppo-verifier/bin/python"' in source
    assert '--verifier-python "$PWD/.venv/bin/python"' not in source
    m=module('study')
    assert {'tools/transfer/run.sh','tools/transfer/submit.py'}<=set(m.code_hashes())


def test_changed_verifier_blocks_before_encoder(tmp_path):
    m=module('study')
    from slice_critic.config import TrainingConfig
    class Grader:
        provenance={'kind':'changed'}
        def __init__(self,*args):pass
        def check(self):pass
    manifest=dict(shards=1,config=TrainingConfig(dataset='toy').__dict__,verifier={'kind':'prepared'})
    with patch.object(m,'verify',return_value=manifest),patch.object(m,'VerificationClient',Grader),patch.object(m,'TransformersBackend') as model:
        with pytest.raises(ValueError,match='Verifier identity changed'):m.generate(tmp_path,0)
        model.assert_not_called()



def test_model_path_alias_is_preserved_in_submission(tmp_path):
    m=module('submit');real=tmp_path/'real-cache';real.mkdir()
    alias=tmp_path/'cache-alias';alias.symlink_to(real,target_is_directory=True)
    model=alias/'snapshot';(real/'snapshot').mkdir()
    calls=[]
    class Result:
        returncode=0;stderr=''
        def __init__(self,n):self.stdout=str(n)
    def run(cmd,**kwargs):calls.append(cmd);return Result(200+len(calls))
    with patch.object(m.subprocess,'run',run):m.submit(tmp_path/'alias-study','smoke',model=model)
    command=calls[0]
    assert command[command.index('--model')+1]==str(model)
    assert str(model)!=str(model.resolve())


def test_full_accepts_only_known_path_only_smoke_migration(tmp_path):
    m=module('submit');smoke=tmp_path/'smoke';(smoke/'status').mkdir(parents=True)
    import hashlib,json
    wrapper='tools/transfer/run.sh'
    manifest=dict(protocol='slice-transfer-v1',model_path=str(tmp_path/'cache-alias'/'snapshot'),
                  code={'tools/transfer/submit.py':m.PATH_FIX_PREDECESSOR,
                        wrapper:hashlib.sha256((ROOT/wrapper).read_bytes()).hexdigest()})
    (smoke/'manifest.json').write_text(json.dumps(manifest))
    (smoke/'status/evaluate-0.json').write_text(json.dumps(dict(state='complete')))
    (smoke/'report.txt').write_text('successful smoke')
    calls=[]
    class Result:
        returncode=0;stderr=''
        def __init__(self,n):self.stdout=str(n)
    def run(cmd,**kwargs):calls.append(cmd);return Result(300+len(calls))
    with patch.object(m.subprocess,'run',run):m.submit_full(tmp_path/'full',smoke,4,4,None)
    assert len(calls)==5
    assert calls[0][calls[0].index('--model')+1]==manifest['model_path']
    audit=json.loads((tmp_path/'full-submission/smoke-reuse.json').read_text())
    assert set(audit['recognized_changes'])=={'tools/transfer/submit.py'}
    manifest['code'][wrapper]='0'*64
    with pytest.raises(ValueError,match='Implementation changed'):m.smoke_changes(manifest)
    manifest['code']={'tools/transfer/submit.py':'1'*64}
    with pytest.raises(ValueError,match='Implementation changed'):m.smoke_changes(manifest)
