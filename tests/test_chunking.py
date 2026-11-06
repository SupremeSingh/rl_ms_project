"""Segmentation, causal state selection, chunk algebra and actual trainer hook tests."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from slice_critic.chunks import ChunkPlan, character_boundaries, token_chunks
from slice_critic.cumulative import CumulativeCritic, raw_design
from math_rl.chunked import chunk_credit, compress_rewards, plans_for_batch

ROOT = Path(__file__).resolve().parents[1]


class Characters:
    is_fast = True
    eos_token_id = 0
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)
    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(c) for c in text], offset_mapping=[(i,i+1) for i in range(len(text))])


@pytest.mark.parametrize('mode,text', [
    ('math', 'First expand $x=2$. Next substitute.\nThen simplify the expression.'),
    ('math', r'Compute $\frac{1}{2}=0.5$ and continue.'),
    ('math', '\n\n'), ('math', '答えは二です。'),
    ('gsm8k', '\n\nFirst calculate.\nSecond calculate.\n#### 4'),
])
def test_character_preservation_and_token_partition(mode, text):
    p = character_boundaries(text, mode)
    assert ''.join(text[a:b] for a,b in zip(p,p[1:])) == text
    plan = token_chunks(Characters(), [ord(c) for c in text]+[0], mode)
    assert plan.ends[-1] == len(text)+1
    assert sum(b-a for a,b in zip(plan.starts,plan.ends)) == len(text)+1
    assert plan.expand(torch.arange(len(plan.starts)))[-1] == len(plan.starts)-1


def test_eos_and_alignment_fail_closed():
    assert token_chunks(Characters(), [0]).starts == (0,)
    class Bad(Characters):
        def __call__(self, *args, **kwargs): return dict(input_ids=[99],offset_mapping=[(0,1)])
    with pytest.raises(ValueError,match='changed token IDs'): token_chunks(Bad(), [65,0])
    with pytest.raises(ValueError,match='fast tokenizer'): token_chunks(None,[65])
    with pytest.raises(ValueError): ChunkPlan((0,2),(1,3),3)


def test_boundary_crossing_token_uses_start_character_and_eos_last_chunk():
    class Cross(Characters):
        def decode(self,*a,**k): return 'First line.\nSecond line.'
        def __call__(self,*a,**k):
            return dict(input_ids=[1,2,3],offset_mapping=[(0,10),(10,14),(14,23)])
    # Newline is at char 11: token starting at 10 belongs to the first chunk.
    p=token_chunks(Cross(),[1,2,3,0], 'gsm8k')
    assert p.starts == (0,2) and p.ends == (2,4)


def test_chunk_td_terminal_padding_and_constant_broadcast():
    plans=[ChunkPlan((0,2),(2,5),5),ChunkPlan((0,),(3,),3)]
    mask=torch.tensor([[1.,1.,1.,1.,1.],[1.,1.,1.,0.,0.]])
    values=torch.tensor([[.2,99,.6,99,99],[.4,99,99,0,0.]])
    rewards=torch.zeros_like(mask);rewards[0,4]=1
    a,r,b,v=chunk_credit(values,rewards,mask,plans,gae_lambda=0.)
    torch.testing.assert_close(a,torch.tensor([[.4,.4,.4,.4,.4],[-.4,-.4,-.4,0,0.]]))
    torch.testing.assert_close(r,torch.tensor([[.6,.6,1.,1.,1.],[0.,0.,0.,0.,0.]]))
    assert b.sum()==3 and not a.requires_grad
    cr,cm=compress_rewards(rewards,plans)
    assert cr.tolist()==[[0.,1.],[0.,0.]] and cm.tolist()==[[1.,1.],[1.,0.]]


@pytest.mark.parametrize('method', ['ridge','lstd'])
def test_chunk_matrix_statistics_once_per_state_and_lambda_one_identity(method):
    features=[torch.tensor([[1.,2.],[3.,4.]])]
    mask=torch.ones(1,2);reward=torch.tensor([[0.,1.]])
    c=CumulativeCritic(['q'],method=method,trace_lambda=1.)
    for step in (1,2):
        values,heads,m=c.update(features,reward,mask,['q'],step)
        phi=raw_design(features[0])
        torch.testing.assert_close(c.stats[0]['a'],step*phi.T@phi)
        torch.testing.assert_close(c.stats[0]['b'],step*phi.sum(0))
        assert len(heads)==1 and not m['cross_fitted']
    assert c.stats[0]['transitions']==4 and c.stats[0]['answers']==2


def test_conventional_boundary_loss_leaves_attention_intact_and_restores_on_error():
    from math_rl.chunked_critic import boundary_loss
    mask=torch.ones(1,4)
    batch=dict(responses=torch.tensor([[65,10,66,0]]),attention_mask=mask)
    calls=[]
    class Critic:
        def _forward_micro_batch(self,batch):
            torch.testing.assert_close(batch['attention_mask'],mask)
            return torch.tensor([[1.,2.,3.,4.]],requires_grad=True)
    critic=Critic()
    def loss(**kw):
        calls.append(kw['response_mask'].clone())
        return (kw['vpreds']*kw['response_mask']).sum()
    core=SimpleNamespace(compute_value_loss=loss)
    with pytest.raises(RuntimeError,match='simulated'):
        with boundary_loss(critic,Characters(),'gsm8k',core):
            pred=critic._forward_micro_batch(batch)
            objective=core.compute_value_loss(vpreds=pred,response_mask=mask)
            objective.backward()
            assert pred.grad.tolist()==[[1.,1.,0.,0.]]
            raise RuntimeError('simulated')
    assert core.compute_value_loss is loss and '_forward_micro_batch' not in critic.__dict__


def test_upstream_math_splitter_is_vendored_with_license():
    # Exact algorithm copy apart from the self-contained dependency helper.
    root=ROOT/'packages/slice-critic/src/slice_critic/_vendor'
    assert 'Copyright (c) 2024 McGill NLP' in (root/'VINEPPO_LICENSE').read_text()
    assert 'MAX_PART_LENGTH = 100' in (root/'vineppo_math.py').read_text()


def test_actual_trainer_chunk_hook_fits_only_boundaries_and_bypasses_token_gae(tmp_path, monkeypatch):
    import sys
    import time
    from datasets import Dataset
    monkeypatch.setattr(Dataset, 'from_parquet', lambda *a, **k: [{'extra_info': {'prompt_id': 'q'}}])
    fake = SimpleNamespace()
    original = lambda *a, **k: pytest.fail('Token GAE must not run in chunk mode')
    fake.compute_advantage = original
    result = {}
    class Parent:
        def fit(self):
            self.global_steps = 1
            data = SimpleNamespace(batch=dict(responses=torch.tensor([[65,10,66,0]]),
                response_mask=torch.ones(1,4), token_level_rewards=torch.tensor([[0.,0.,0.,1.]])),
                non_tensor_batch=dict(pretoken_features=[np.array([[.1,.2],[.4,.8]])],
                    critic_encoder_epoch=[0],extra_info=[dict(prompt_id='q')]))
            value = fake.compute_advantage(data)
            assert value.batch['chunk_start_mask'].tolist() == [[1.,1.,0.,0.]]
            torch.testing.assert_close(value.batch['advantages'][0,1:4],value.batch['advantages'][0,1].expand(3))
            result.update(value.batch)
            self.global_steps = 2
    # The upstream token whitening is imported lazily; this test isolates the real hook.
    whitening=SimpleNamespace(masked_whiten=lambda x,m: (x-x[m.bool()].mean())/(x[m.bool()].std()+1e-8))
    monkeypatch.setitem(sys.modules,'verl.utils.torch_functional',whitening)
    tree=ast.parse((ROOT/'src/math_rl/trainer.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    scope=dict(ray_trainer=SimpleNamespace(RayPPOTrainer=Parent,compute_advantage=original),
               json=json, torch=torch, Path=Path, time=time)
    scope['ray_trainer']=fake
    fake.RayPPOTrainer=Parent
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'trainer','exec'),scope)
    trainer=object.__new__(scope['MathTrainer'])
    class Attr(dict):
        def __getattr__(self,k):return self[k]
    trainer.config=Attr(actor_rollout_ref=Attr(model=Attr(critic_feature_mode='frozen')),
        data=Attr(seed=42,train_files='unused'),algorithm=Attr(lam=.95),experiment=Attr(method='ridge',
            cumulative_epsilon=.01,critic_lambda=.99,critic_folds=1,chunking='gsm8k'))
    trainer.tokenizer=Characters();trainer.linear=True;trainer.method='ridge';trainer.output=tmp_path
    trainer.total_training_steps=1
    trainer.fit()
    assert fake.compute_advantage is original
    stats=torch.load(tmp_path/'linear-critic/statistics.pt',weights_only=True)
    assert len(stats['stats'])==1 and stats['stats'][0]['transitions']==2
    assert trainer.completed_updates==1


@pytest.mark.parametrize('lam', [0., .95, 1.])
def test_chunk_gae_matches_explicit_trace_and_lambda_return(lam):
    plans=[ChunkPlan((0,2),(2,5),5)]
    values=torch.tensor([[.2,99.,.6,99.,99.]])
    rewards=torch.tensor([[0.,0.,0.,0.,1.]])
    mask=torch.ones_like(values)
    adv,targets,_,_=chunk_credit(values,rewards,mask,plans,gae_lambda=lam)
    # Two chunk TD errors are .4 and .4; the first collects the later error.
    expected=torch.tensor([[.4+lam*.4,.4+lam*.4,.4,.4,.4]])
    torch.testing.assert_close(adv,expected)
    torch.testing.assert_close(targets,expected+torch.tensor([[.2,.2,.6,.6,.6]]))
    if lam==1.:
        torch.testing.assert_close(targets,mask) # Monte Carlo return at every chunk.


def test_chunk_gae_decay_counts_chunks_not_their_token_lengths():
    def credit(plan):
        mask=torch.ones(1,plan.tokens)
        v=torch.zeros_like(mask);v[0,list(plan.starts)]=torch.tensor([.2,.6])
        r=torch.zeros_like(mask);r[0,-1]=1.
        return chunk_credit(v,r,mask,[plan])[0][0,list(plan.starts)]
    torch.testing.assert_close(credit(ChunkPlan((0,1),(1,2),2)),
                               credit(ChunkPlan((0,20),(20,50),50)))
    with pytest.raises(ValueError,match='GAE lambda'):
        chunk_credit(torch.zeros(1,2),torch.zeros(1,2),torch.ones(1,2),
                     [ChunkPlan((0,),(2,),2)],gae_lambda=float('nan'))
