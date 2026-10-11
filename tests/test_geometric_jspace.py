"""Geometric layer ordering, official multi-source lenses, causal extraction and fits."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from math_rl.jspace import geometric_layers, vocabulary_directions
from math_rl.critic_layers import sampled_block_states


def module():
    scripts=Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('geometric',scripts/'geometric_jspace_probes.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_exact_counting_rule():
    assert geometric_layers(28)==[27,26,24,20,12]
    assert geometric_layers(28,3)==[27,25,19,1]
    assert geometric_layers(32)==[31,30,28,24,16,0]
    assert geometric_layers(1)==[0]
    with pytest.raises(ValueError):geometric_layers(28,1)


def test_official_lenses_and_final_identity_with_causal_blocks():
    jlens=pytest.importorskip('jlens')
    transformers=pytest.importorskip('transformers')
    torch.manual_seed(8)
    config=transformers.Qwen2Config(vocab_size=16,hidden_size=8,intermediate_size=16,
        num_hidden_layers=5,num_attention_heads=2,num_key_value_heads=1,
        max_position_embeddings=64,attention_dropout=0.)
    config._attn_implementation='eager'
    model=transformers.Qwen2ForCausalLM(config).eval()
    class Tokens:
        def __call__(self,*args,**kwargs):return SimpleNamespace(input_ids=torch.arange(24)[None]%16)
    adapter=jlens.from_hf(model,Tokens(),force_bos=False)
    layers=geometric_layers(5)
    lens=jlens.fit(adapter,['example'],source_layers=layers[1:],target_layer=4,dim_batch=2,max_seq_len=32)
    assert set(lens.jacobians)==set(layers[1:])
    tokens=torch.arange(24)[None]%16
    captured=sampled_block_states(model.model,tokens,4,20,[0,3],layers)
    for position,index in ((0,0),(3,1)):
        changed=tokens.clone();changed[:,4+position:]=(changed[:,4+position:]+1)%16
        other=sampled_block_states(model.model,changed,4,20,[position],layers)
        for layer in layers:torch.testing.assert_close(captured[layer][index],other[layer][0])
    before_norm=[]
    hook=model.model.norm.register_forward_pre_hook(lambda mod,args:before_norm.append(args[0].detach()))
    model.model(input_ids=tokens,use_cache=False)
    hook.remove()
    torch.testing.assert_close(captured[4],before_norm[0][0,[3,6]])
    # An identity transport yields precisely the final-block logit directions.
    D=vocabulary_directions(model.lm_head.weight,model.model.norm.weight,torch.eye(8))
    expected=torch.nn.functional.normalize(model.lm_head.weight*model.model.norm.weight,dim=1)
    torch.testing.assert_close(D,expected)


def test_ridge_aggregation_selects_on_validation_before_test(tmp_path,monkeypatch):
    m=module();layers=[4,3,1];dimension=4
    manifest=dict(layers=layers,dimension=dimension,config={'smoke':True},questions=[],
                  trajectory_source=str(tmp_path/'source'))
    (tmp_path/'source/trajectories').mkdir(parents=True)
    torch.manual_seed(19)
    for split_index,split in enumerate(('train','val','test')):
        metadata=dict(y=torch.arange(24).float()%2,question=torch.arange(split_index*4,split_index*4+4).repeat_interleave(6),
                      position=torch.tensor([0,3]*12),response=torch.tensor([0,0,1,1,2,2]*4))
        blocks={kind:[torch.randn(24,dimension) for _ in layers] for kind in ('jspace','hidden','random')}
        for name in m.names(layers):
            (tmp_path/name).mkdir(exist_ok=True)
            if name=='hidden_two_thirds':x=blocks['hidden'][1]
            elif name.endswith('_concat'):x=torch.cat(blocks[name.split('_')[0]],1)
            else:
                kind,block=name.split('-block');x=blocks[kind][layers.index(int(block)-1)]
            torch.save(dict(metadata,x=x),tmp_path/name/f'{split}.pt')
        for qid in metadata['question'].unique().tolist():
            manifest['questions'].append(dict(id=f'q{qid}',original_index=qid,split=split,question='Question'))
            (tmp_path/'source/trajectories'/f'{qid:05d}.json').write_text(json.dumps(dict(responses=[dict(response='Answer')])) )
            record=dict(seconds=.01,attempted=1,retained=1,readouts=[dict(block=l+1,response_index=0,
                prefix_lengths=[0,3],residual_energy_fraction=[.8,.7],active_distinct_atoms=[2,3]) for l in layers])
            (tmp_path/f'question-{qid:05d}.json').write_text(json.dumps(record))
    load=torch.load
    def checked_load(path,*args,**kwargs):
        if Path(path).name=='test.pt':assert (tmp_path/'selection.json').exists()
        return load(path,*args,**kwargs)
    monkeypatch.setattr(torch,'load',checked_load)
    m.fit(tmp_path,manifest)
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['dimension']==12
    assert len(summary['results'])==6
    assert len(summary['paired'])==5
    assert summary['selection']['concatenation_blocks']==[5,4,2]
    assert (tmp_path/'responses.html').exists()
