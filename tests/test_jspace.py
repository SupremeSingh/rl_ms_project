"""Numerical and data-isolation tests; no model download needed."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from math_rl.jspace import vocabulary_directions, reconstruct, rotated_dictionary, representations


def test_dictionary_matches_qwen_rms_readout():
    torch.manual_seed(71)
    U=torch.randn(13,6);gain=torch.rand(6)+.5;J=torch.randn(6,6);h=torch.randn(4,6)
    dictionary=vocabulary_directions(U,gain,J)
    transported=h@J.T
    rms=(transported.square().mean(1,keepdim=True)+1e-6).sqrt()
    logits=((transported/rms)*gain)@U.T
    row_norms=((U*gain)@J).norm(dim=1)
    torch.testing.assert_close((h@dictionary.T)*row_norms/rms,logits,atol=1e-5,rtol=1e-5)


def test_sparse_nonnegative_exact_known_case_and_residual():
    dictionary=torch.eye(4)
    x=torch.tensor([[3.,0.,2.,0.],[-1.,0.,0.,0.],[0.,0.,0.,0.]])
    y,ids,coeff=reconstruct(x,dictionary,2)
    torch.testing.assert_close(y,torch.tensor([[3.,0.,2.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]]))
    assert (coeff>=0).all()
    torch.testing.assert_close(y,(dictionary[ids]*coeff[:,:,None]).sum(1))
    features=representations(x,x,y,y)
    torch.testing.assert_close(features['residual']+y,x)
    assert features['jspace_final'].shape==(3,8)


def test_pursuit_monotonic_and_random_geometry_matched():
    torch.manual_seed(3)
    d=torch.nn.functional.normalize(torch.randn(30,8),dim=1);x=torch.randn(5,8)
    control=rotated_dictionary(d)
    torch.testing.assert_close(d@d.T,control@control.T)
    assert not torch.allclose(d,control)
    prior=x.square().sum(1)
    for k in (1,2,4,8):
        y,_,_=reconstruct(x,d,k);error=(x-y).square().sum(1)
        assert (error<=prior+1e-5).all()
        prior=error


def runner():
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    import compare_jspace_probes
    return compare_jspace_probes


def questions():
    rows=[]
    for split,count in [('train',18),('val',9),('test',9)]:
        for i in range(count):
            rows.append(dict(id=f'{split}/{i}',split=split,question=f'{split} problem {i}',
                reference_solution=r'Answer: \boxed{2}',ground_truth='2',level=4))
    return rows


def test_calibration_is_train_only_and_excluded_and_indices_preserved():
    module=runner();original=questions()
    selected,calibration,audit=module.select_questions(original,dict(calibration_prompts=2,smoke=True))
    assert len(calibration)==2 and all(q['split']=='train' for q in calibration)
    assert not {q['id'] for q in selected}&{q['id'] for q in calibration}
    assert {q['split'] for q in selected}=={'train','val','test'}
    for q in selected+calibration:assert original[q['original_index']]['id']==q['id']
    # Selection is deterministic and independent of answer outcomes.
    assert selected==module.select_questions(original,dict(calibration_prompts=2,smoke=True))[0]


def test_reference_filter_and_split_leak_fail_closed():
    module=runner();rows=questions()
    rows[0]['reference_solution']=r'\boxed{2} or \boxed{3}'
    selected,audit=module.eligible_questions(rows)
    assert rows[0]['id'] not in {q['id'] for q in selected}
    assert not audit[0]['eligible']
    rows[1]['question']=rows[2]['question']
    with pytest.raises(ValueError,match='Duplicate'):module.eligible_questions(rows)


def test_official_library_fit_when_installed():
    # Real upstream estimator, not a mock. Cluster setup installs the pinned package.
    jlens=pytest.importorskip('jlens')
    from torch import nn
    class Tiny:
        n_layers=2;d_model=3
        def __init__(self):
            self.layers=nn.ModuleList([nn.Identity(),nn.Linear(3,3,bias=False)])
            with torch.no_grad():self.layers[1].weight.copy_(torch.diag(torch.tensor([1.,2.,3.])))
            self.layers.requires_grad_(False)
        def encode(self,prompt,max_length):return torch.ones(1,6,dtype=torch.long)
        def forward(self,ids):
            x=torch.ones(*ids.shape,3)
            for layer in self.layers:x=layer(x)
            return x
    model=Tiny()
    lens=jlens.fit(model,['calibration'],source_layers=[0],target_layer=1,dim_batch=2,skip_first=0,max_seq_len=6)
    torch.testing.assert_close(lens.jacobians[0],torch.diag(torch.tensor([1.,2.,3.])))


def test_synthetic_probe_fit_and_paired_report(tmp_path):
    module=runner()
    from math_rl.jspace import REPRESENTATIONS
    from math_rl.provenance import write_json
    cfg=dict(heads=['linear_logit','mlp2'],seeds=[42],learning_rates=[.01],epochs=2,
             min_epochs=1,patience=2,smoke=True,response_limit=None)
    qs=[]
    for i,split in enumerate(('train','val','test')):
        qs.append(dict(id=f'q{i}',split=split,question='Synthetic question',ground_truth='1'))
    rng=torch.Generator().manual_seed(19)
    split_data={}
    for split in ('train','val','test'):
        x=torch.randn(16,4,generator=rng);y=(x[:,0]>0).float()
        split_data[split]=dict(x=x,y=y,question=torch.arange(16)//8,
                              position=torch.arange(16)%8,response=torch.arange(16)//8)
    for name in REPRESENTATIONS:
        folder=tmp_path/name;(folder/'heads').mkdir(parents=True);(folder/'trajectories').mkdir()
        for split,data in split_data.items():module.frozen.save_tensor(folder/f'{split}.pt',data)
        for i,q in enumerate(qs):
            write_json(folder/'trajectories'/f'{i:05d}.json',dict(responses=[dict(
                score=1,verifier_status='correct',finish_reason='stop',token_ids=[1],response='Answer 1')]))
        module.frozen.fit(folder,cfg,qs)
    for i,q in enumerate(qs):
        write_json(tmp_path/f'question-{i:05d}.json',dict(attempted=1,retained=1,statuses={'correct':1},
            seconds=.1,readouts=[dict(response_index=0,prefix_lengths=[0],tokens=[['1']],coefficients=[[1.]])]))
    module.summarize(tmp_path,dict(config=cfg,questions=qs))
    result=json.loads((tmp_path/'summary.json').read_text())
    assert len(result['results'])==24
    # Identical synthetic inputs imply identical fits/differences across arms.
    assert all(abs(row['mean'])<1e-10 for row in result['paired'].values())
    assert (tmp_path/'responses.html').exists()
    assert result['answers']['retained']==3


def test_official_hf_adapter_on_tiny_qwen_without_download():
    jlens=pytest.importorskip('jlens')
    transformers=pytest.importorskip('transformers')
    from types import SimpleNamespace
    from math_rl.critic_layers import sampled_layer_states
    config=transformers.Qwen2Config(vocab_size=32,hidden_size=12,intermediate_size=24,
        num_hidden_layers=4,num_attention_heads=3,num_key_value_heads=1,
        max_position_embeddings=128,attention_dropout=0.)
    config._attn_implementation='eager'
    hf=transformers.Qwen2ForCausalLM(config).eval()
    class Tokens:
        def __call__(self,text,**kwargs):return SimpleNamespace(input_ids=torch.arange(24)[None]%32)
    adapter=jlens.from_hf(hf,Tokens(),force_bos=False)
    lens=jlens.fit(adapter,['sample'],source_layers=[2],dim_batch=2,max_seq_len=128)
    assert torch.isfinite(lens.jacobians[2]).all()
    tokens=torch.arange(24)[None]%32
    captured=sampled_layer_states(hf.model,tokens,4,20,[0,2,19])
    short=sampled_layer_states(hf.model,tokens[:,:7],4,3,[2])
    torch.testing.assert_close(captured['two_thirds'][1],short['two_thirds'][0])
    h=captured['two_thirds'].float()
    dictionary=vocabulary_directions(hf.lm_head.weight,hf.model.norm.weight,lens.jacobians[2])
    raw=(hf.lm_head.weight*hf.model.norm.weight)@lens.jacobians[2]
    transported=lens.transport(h,2)
    divisor=(transported.square().mean(-1,keepdim=True)+hf.model.norm.variance_epsilon).sqrt()
    torch.testing.assert_close((h@dictionary.T)*raw.norm(dim=1)/divisor,adapter.unembed(transported),atol=1e-5,rtol=1e-5)
