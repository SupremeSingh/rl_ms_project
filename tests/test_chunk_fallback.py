"""Upstream malformed-LaTeX reconstruction must never discard sampled answers."""
import pytest
import torch
from slice_critic.chunks import character_boundaries, token_chunks
from slice_critic._vendor import vineppo_math
from math_rl.chunked import chunk_credit, compress_rewards, plans_for_batch


class Characters:
    is_fast=True
    eos_token_id=0
    def decode(self,ids,**kw):return ''.join(chr(i) for i in ids)
    def __call__(self,text,**kw):
        return dict(input_ids=list(map(ord,text)),offset_mapping=[(i,i+1) for i in range(len(text))])


def test_long_repeated_latex_reproduces_upstream_bug_but_preserves_actions():
    # >20 characters reproduces the SAME final-boundary assertion in the GPU log.
    text='$a$'*12
    with pytest.raises(AssertionError,match='Last index'):vineppo_math.split_solution_inplace(text)
    assert character_boundaries(text,'math')==[0,len(text)]
    plan=token_chunks(Characters(),list(map(ord,text))+[0],'math')
    assert plan.fallback_reason=='upstream_assertion'
    assert plan.starts==(0,) and plan.ends==(len(text)+1,)
    mask=torch.ones(1,plan.tokens);rewards=torch.zeros_like(mask);rewards[0,-1]=1
    values=torch.zeros_like(mask);values[0,0]=.25
    adv,returns,_,_=chunk_credit(values,rewards,mask,[plan])
    torch.testing.assert_close(adv,torch.full_like(mask,.75))
    torch.testing.assert_close(returns,mask)


def test_normal_segmentation_matches_upstream_without_fallback():
    text='First compute the result carefully. Then simplify the final expression.'
    assert character_boundaries(text,'math')==sorted(set(vineppo_math.split_solution_inplace(text)))
    tokens=list(map(ord,text))+[0]
    plan=token_chunks(Characters(),tokens,'math')
    assert plan.fallback_reason is None
    assert token_chunks(Characters(),tokens,'math',fallback_tokens=1)==plan


@pytest.mark.parametrize('size', [1, 8, 16, 64, 128])
@pytest.mark.parametrize('eos', [False, True])
def test_configurable_fallback_preserves_actions_reward_and_causal_starts(size,eos):
    text='$a$'*50
    tokens=list(map(ord,text))+([0] if eos else [])
    original=tokens.copy()
    plan=token_chunks(Characters(),tokens,'math',fallback_tokens=size)
    if size==64:
        assert plan==token_chunks(Characters(),tokens,'math')
    assert plan.fallback_reason=='upstream_assertion'
    assert plan.starts==tuple(range(0,len(tokens),size))
    assert all(0 < b-a <= size for a,b in zip(plan.starts,plan.ends))
    assert [token for a,b in zip(plan.starts,plan.ends) for token in tokens[a:b]]==original
    assert tokens==original
    mask=torch.ones(1,len(tokens));rewards=torch.zeros_like(mask);rewards[0,-1]=1
    # Monte Carlo GAE must retain the final return at every fallback state.
    adv,targets,boundaries,_=chunk_credit(torch.zeros_like(mask),rewards,mask,[plan],gae_lambda=1.)
    torch.testing.assert_close(adv,mask)
    torch.testing.assert_close(targets,mask)
    assert boundaries[0].nonzero().flatten().tolist()==list(plan.starts)
    compact,valid=compress_rewards(rewards,[plan])
    assert compact.sum()==1 and compact[0,-1]==1 and valid.sum()==len(plan.starts)


@pytest.mark.parametrize('size', [0, -1, True, 1.5, '64'])
def test_invalid_fallback_size_rejected(size):
    from slice_critic import TrainingConfig
    with pytest.raises(ValueError,match='positive integer'):
        token_chunks(Characters(),[65,0],fallback_tokens=size)
    with pytest.raises(ValueError,match='chunk_fallback_tokens'):
        TrainingConfig(dataset='example',chunk_fallback_tokens=size).validate()


def test_offline_batch_and_transformer_loss_use_same_fallback_boundaries():
    from slice_critic import TrainingConfig
    from slice_critic.training import _state_positions
    from math_rl.chunked_critic import boundary_loss
    from types import SimpleNamespace
    text='$a$'*12;tokens=list(map(ord,text))+[0];size=8
    config=TrainingConfig(dataset='example',chunk_fallback_tokens=size)
    expected=_state_positions(config,SimpleNamespace(tokenizer=Characters()),dict(token_ids=tokens))
    responses=torch.tensor([tokens]);mask=torch.ones_like(responses)
    plans=plans_for_batch(Characters(),responses,mask,'math',size)
    assert list(plans[0].starts)==expected
    class Critic:
        def _forward_micro_batch(self,batch):
            torch.testing.assert_close(batch['attention_mask'],mask)
            return torch.ones_like(mask,dtype=torch.float32,requires_grad=True)
    core=SimpleNamespace(compute_value_loss=lambda **kw:kw['response_mask'])
    critic=Critic()
    with boundary_loss(critic,Characters(),'math',core,fallback_tokens=size):
        critic._forward_micro_batch(dict(responses=responses,attention_mask=mask))
        loss_mask=core.compute_value_loss(response_mask=mask)
    assert loss_mask[0].nonzero().flatten().tolist()==expected


def test_invalid_partition_uses_fixed_token_fallback(monkeypatch):
    monkeypatch.setattr(vineppo_math,'split_solution_inplace',lambda text:[0,len(text)-1])
    plan=token_chunks(Characters(),list(map(ord,'A long response'))+[0],fallback_tokens=4)
    assert plan.fallback_reason=='invalid_partition'
    assert plan.starts==(0,4,8,12)


def test_unexpected_programming_errors_are_not_hidden(monkeypatch):
    def broken(text):raise RuntimeError('unrelated defect')
    monkeypatch.setattr(vineppo_math,'split_solution_inplace',broken)
    with pytest.raises(RuntimeError,match='unrelated defect'):character_boundaries('some response','math')
