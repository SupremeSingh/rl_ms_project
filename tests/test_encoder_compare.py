import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from torch import nn
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/encoder'))
import compare as c
import submit_compare as submit


def test_fixed_prefix_schedule_is_independent_of_suffix_and_reward():
    m=dict(stride=32)
    a=dict(token_ids=list(range(100)),score=0)
    b=dict(token_ids=list(range(200)),score=1)
    assert c.positions(a,m)==[0,32,64,96]
    assert [x for x in c.positions(b,m) if x<100]==c.positions(a,m)


def test_factorial_counts_and_full_official_configs(tmp_path,monkeypatch):
    m=dict(adapt_seeds=[42,43,44],compressor_seeds=[42,43,44,45,46],model='/base',adapt_steps=1000)
    assert len(c.combinations(m))==48
    assert {a for _,a,_ in c.combinations(m)}==set('ABCD')
    monkeypatch.setattr(c,'checked',lambda *args:dict(context=2560))
    a,b=c.configs(tmp_path,m,43)
    assert a['max_seq_length']==b['max_seq_length']==2560
    assert b['per_device_train_batch_size']==8 and b['gradient_accumulation_steps']==1
    assert b['seed']==43 and b['model_name_or_path']==str(tmp_path/'adapt-43/mntp-model')
    assert a['train_file']==str(tmp_path/'adaptation.txt')
    assert not a['do_eval'] and not b['do_eval']


class Block(nn.Module):
    def forward(self,x):return (x+1,)
class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(100,4);self.layers=nn.ModuleList([Block() for _ in range(28)])
        self.norm=nn.Identity();self.config=SimpleNamespace(rms_norm_eps=1e-6,max_position_embeddings=100)
    def forward(self,input_ids,**kw):
        x=self.embedding(input_ids)
        for l in self.layers:x=l(x)[0]
        return x


def test_mean_pooling_then_rms_exact_layers_and_hooks_cleaned():
    torch.manual_seed(1);model=Toy().eval().requires_grad_(False);seq=[1,2,3]
    r=c.pooled(model,seq);base=model.embedding(torch.tensor(seq)).mean(0)
    expected=lambda l:c.rms_coordinates(base+l+1,1e-6)
    torch.testing.assert_close(r['A'],expected(27));torch.testing.assert_close(r['B'],expected(18))
    torch.testing.assert_close(r['multi'],torch.cat([expected(l) for l in c.LAYERS]))
    assert all(not l._forward_hooks for l in model.layers)
    model.train()
    with pytest.raises(ValueError,match='Frozen'):c.pooled(model,seq)


def test_sealed_data_and_rollout_provenance(tmp_path):
    c.seal(tmp_path,'questions.json',[dict(id='q',split='train')])
    assert len(c.items(tmp_path,dict(answers=8)))==8
    with pytest.raises(FileExistsError):c.seal(tmp_path,'questions.json',[])
    (tmp_path/'questions.json').write_text('[]')
    with pytest.raises(ValueError,match='Changed artifact'):c.checked(tmp_path,'questions.json')


def test_question_inventory_uses_actual_package_run_provenance(tmp_path):
    p=tmp_path/'questions.json';c.write_json(p,[])
    c.write_json(tmp_path/'run.json',dict(questions_sha256=c.sha256(p)))
    assert c.provenance(p)['questions_sha256']==c.sha256(p)
    p.write_text('[1]')
    with pytest.raises(ValueError):c.provenance(p)


def test_submission_dependencies_four_gpu_gates_and_no_duplicate_retry(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(submit,'resolve_questions',lambda x:tmp_path/'questions.json')
    monkeypatch.setattr(submit.subprocess,'run',lambda cmd,**kw:(calls.append(cmd) or SimpleNamespace(stdout=str(len(calls)))))
    args=SimpleNamespace(out=tmp_path/'out',source=tmp_path/'src',questions=None,numina=tmp_path/'numina',official=tmp_path/'official',mode='smoke',smoke=None)
    submit.submit(args)
    assert len(calls)==9
    assert '--gres=gpu:a5000:4' in calls[3] and '--gres=gpu:a5000:4' in calls[4]
    assert '--array=0-15%4' in calls[5] and '--array=0-7%4' in calls[6]
    for i,cmd in enumerate(calls[1:],1):assert '--dependency=afterok:'+str(i) in cmd
    with pytest.raises(FileExistsError):submit.submit(args)


def test_pipeline_score_uses_ridge_not_sigmoid_and_questions_equal(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'items',lambda *args:[dict(id='a',split='val'),dict(id='b',split='val')])
    rows={'a':dict(x={'A':torch.tensor([[1.],[1.]])},y=torch.zeros(2)),
          'b':dict(x={'A':torch.tensor([[0.]])},y=torch.ones(1))}
    monkeypatch.setattr(c,'feature',lambda out,b,row:rows[row['id']])
    value,qs=c.score(tmp_path,{},'causal','A',dict(weights=torch.tensor([2.,0.],dtype=torch.float64)),'val')
    assert value==2.5 and qs=={'a':4.,'b':1.}


def test_primary_intervals_pair_seeds_and_cluster_questions():
    m=dict(adapt_seeds=[42,43],compressor_seeds=[42,43]);rows={}
    for b in ['causal','adapted-42','adapted-43']:
        for a in c.ARMS:
            for seed in ([42,43] if a in ('C','D') else [0]):
                v=0 if b=='causal' else 1
                rows[f'{b}-{a}-{seed}']=dict(brier={'q1':v,'q2':v},mc={'q1':[v,v],'q2':[v,v]})
    r=c.primary_intervals(m,rows,draws=20)
    assert len(r)==8
    assert all(v['mean']==1 and v['family_adjusted_interval']==[1,1] for v in r.values())


def legacy_corpus(tmp_path):
    rows=[]
    for split,n in [('calibration',512),('screen',256),('train',10000),('val',1000),('test',2000)]:
        for i in range(n):
            code=''.join(chr(65+(i//(26**k))%26) for k in range(4))
            rows.append(dict(id=f'{split}-{i}',question=f'{split} {code}',ground_truth='7',split=split,
                             reference_solution='Must not enter adapted text',prompt_token_ids=[9]))
    path=tmp_path/'questions.json';c.write_json(path,rows)
    c.write_json(tmp_path/'manifest.json',dict(protocol='historical',questions_sha256=c.sha256(path)))
    return path


def test_historical_manifest_schema_budget_and_tamper_checks(tmp_path):
    import question_source as source
    path=legacy_corpus(tmp_path)
    assert submit.resolve_questions(path)==path.resolve()
    rows=source.load_questions(path)
    assert source.main_budget(rows)==dict(train=10000,val=1000,test=2000)
    assert rows[0]['reference']=='7' and 'reference_solution' not in rows[0]
    assert source.provenance(path)['metadata_path']==str(tmp_path/'manifest.json')
    path.write_text('[]')
    with pytest.raises(ValueError,match='hash differs'):submit.resolve_questions(path)


def test_legacy_source_smoke_keeps_assignments_and_seals_metadata(tmp_path,monkeypatch):
    path=legacy_corpus(tmp_path)
    src=tmp_path/'actor-source';src.mkdir()
    c.write_json(src/'manifest.json',dict(protocol='fixed-policy-critics-v1',actor_path='/actor',actor_identity={},base_identity={}))
    c.write_json(src/'verification.json',dict(passed=True,files={}))
    monkeypatch.setattr(c,'model_identity',lambda *args:{})
    monkeypatch.setattr(c,'identity',lambda:{'test':'runtime'})
    monkeypatch.setattr(c.subprocess,'check_output',lambda cmd,**kw:c.study.OFFICIAL if 'rev-parse' in cmd else '')
    args=SimpleNamespace(source=src,questions=path,numina=tmp_path/'absent',official=tmp_path/'official',mode='smoke')
    out=tmp_path/'new';c.prepare(out,args)
    selected=c.checked(out,'questions.json')
    assert len(selected)==16
    assert all(q['id'].startswith(q['split']+'-') for q in selected)
    assert {q['split'] for q in selected}=={'train','val','test'}
    m=c.checked(out,'manifest.json')
    assert m['main_question_budget']==dict(train=10000,val=1000,test=2000)
    assert m['question_metadata_hash']==c.sha256(tmp_path/'manifest.json')
    assert all(q['ground_truth']=='7' for q in selected)
