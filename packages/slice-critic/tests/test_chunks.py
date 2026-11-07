from importlib.resources import files
import torch
from slice_critic.chunks import character_boundaries, ChunkPlan, token_chunks


def test_installed_splitter_preserves_math_and_includes_license():
    text='Expand the expression $x^2+2x+1$. Then simplify the result.\nFinish with the final answer.'
    points=character_boundaries(text,'math')
    assert ''.join(text[a:b] for a,b in zip(points,points[1:]))==text
    assert len(points)>2
    license=files('slice_critic._vendor').joinpath('VINEPPO_LICENSE').read_text()
    assert 'Copyright (c) 2024 McGill NLP' in license


def test_broadcast_never_drops_terminal_action():
    p=ChunkPlan((0,2),(2,5),5)
    assert p.expand(torch.tensor([.2,.5])).tolist()==[float(torch.tensor(.2))]*2+[.5]*3


def test_installed_fixed_token_fallback_is_configurable_and_preserves_eos():
    from slice_critic import TrainingConfig
    class Characters:
        is_fast=True
        eos_token_id=0
        def decode(self,ids,**kwargs):return ''.join(chr(i) for i in ids)
        def __call__(self,text,**kwargs):
            return dict(input_ids=list(map(ord,text)),offset_mapping=[(i,i+1) for i in range(len(text))])
    config=TrainingConfig(dataset='example',chunk_fallback_tokens=8).validate()
    tokens=list(map(ord,'$a$'*12))+[0]
    plan=token_chunks(Characters(),tokens,fallback_tokens=config.chunk_fallback_tokens)
    assert plan.fallback_reason=='upstream_assertion'
    assert plan.starts==(0,8,16,24,32) and plan.ends==(8,16,24,32,37)
    assert [v for a,b in zip(plan.starts,plan.ends) for v in tokens[a:b]]==tokens
