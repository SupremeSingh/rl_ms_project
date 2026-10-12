"""Readout algebra, coefficient accounting, leakage guards and cached viewer integration."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from math_rl.jspace_inspection import (association, concept_ids, direct_readout,
                                      html_payload, linear_contributions, visible_mentions)


class Tokenizer:
    words = ['error', ' Wrong', 'correct', 'maybe', 'but', 'x', 'faulty', 'ERROR']
    def decode(self, ids, **kwargs):
        return ''.join(self.words[int(i)] for i in ids)


def test_direct_readout_is_normalized_lens_not_sparse_coefficients():
    torch.manual_seed(7)
    h, j, u = torch.randn(3,4), torch.randn(4,4), torch.randn(8,4)
    gain = torch.tensor([1.,2.,3.,4.])
    groups, coverage = concept_ids(Tokenizer(), 8)
    assert groups['error'] == [0,1,7]  # "faulty" is not the exact word "fault".
    ids, p, mass = direct_readout(h,j,u,gain,1e-6,groups,top_k=8)
    target = torch.nn.functional.rms_norm(h@j.T, (4,), gain, eps=1e-6)@u.T
    target = target.softmax(-1)
    torch.testing.assert_close(p, target.gather(1,ids))
    torch.testing.assert_close(mass['error'],target[:,[0,1,7]].sum(1))
    assert coverage['error'][1]['text'] == ' Wrong'


def test_contributions_and_visible_words_and_safe_html():
    record = dict(state={'weight':torch.tensor([[2.,-3.,1.]]),'bias':torch.tensor([.2])},
                  mean=torch.tensor([1.,2.,3.]), scale=torch.tensor([2.,1.,4.]))
    h = torch.tensor([[2.,3.,4.],[1.,1.,1.]])
    predictions, detail = linear_contributions(record,h,count=2)
    for v,d in zip(predictions,detail):
        assert sum(d['contributions'])+d['other']+d['bias'] == pytest.approx(float(v))
    assert visible_mentions('A fault occurred')['error']
    assert not visible_mentions('faulty machinery')['error']
    encoded = html_payload({'text':'</script><script>alert(1)</script>'})
    assert '<' not in encoded and json.loads(encoded)['text'].startswith('</script>')


def test_parity_with_official_transport_and_qwen_unembed():
    jlens = pytest.importorskip('jlens')
    transformers = pytest.importorskip('transformers')
    config = transformers.Qwen2Config(vocab_size=8,hidden_size=4,intermediate_size=8,
        num_hidden_layers=2,num_attention_heads=1,num_key_value_heads=1)
    model = transformers.Qwen2ForCausalLM(config).float().eval()
    adapter = jlens.from_hf(model,Tokenizer(),force_bos=False)
    lens = jlens.JacobianLens({0:torch.randn(4,4)},n_prompts=2,d_model=4)
    transported = lens.transport(torch.randn(3,4),0)
    expected = adapter.unembed(transported).softmax(-1)
    ids,p,mass = direct_readout(transported,None,model.lm_head.weight,model.model.norm.weight,
                              config.rms_norm_eps,{'error':[0]},top_k=8)
    torch.testing.assert_close(p,expected.gather(1,ids))
    torch.testing.assert_close(mass['error'],expected[:,0])


def test_association_ignores_question_only_and_marks_one_class_undefined():
    rows = []
    for q,y in [('q1',0),('q2',1),('q3',0),('q4',1)]:
        for position in (0,10,20):
            rows.append(dict(question=q,response=0,position=position,block=19,correct=y,
                mass={k:(float(y) if position==0 else 1.-y) for k in ('error','uncertainty','revision')},
                mentions={k:y==1 for k in ('error','uncertainty','revision')}))
    result = association(rows, iterations=50)
    assert result['block19/error/all']['failure_auroc'] == 1.
    assert result['block19/error/all']['wrong_minus_correct_mean_mass'] == 1.
    assert result['block19/error/no_visible_mention']['failure_auroc'] is None


def test_cached_inspection_end_to_end(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('inspect_jspace',scripts/'inspect_jspace.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from math_rl.provenance import sha256
    source,old,model,trajectory = [tmp_path/name for name in ('source','old','model','data')]
    for p in (source,old,model,trajectory/'trajectories'):p.mkdir(parents=True)
    (old/'manifest.json').write_text('{}')
    (source/'status.json').write_text('{"state":"complete"}')
    (source/'extraction.json').write_text('{}')
    (model/'config.json').write_text('{"rms_norm_eps":1e-6}')
    for p in (old,source):
        (p/'lens.pt').write_bytes(b'test')
        (p/'lens-metadata.json').write_text(json.dumps({'sha256':sha256(p/'lens.pt')}))
    torch.manual_seed(5)
    lens = SimpleNamespace(jacobians={3:torch.eye(4),1:torch.eye(4)})
    lens.transport = lambda h,layer: h@lens.jacobians[layer].to(h.device).T
    monkeypatch.setitem(sys.modules,'jlens',SimpleNamespace(JacobianLens=SimpleNamespace(load=lambda path:lens)))
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:Tokenizer())
    monkeypatch.setattr(module.original,'MODEL',model)
    monkeypatch.setattr(module.original,'installed_lens',lambda:{'test':'1'})
    monkeypatch.setattr(module,'load_readout_weights',lambda p:{'lm_head.weight':torch.randn(8,4),'model.norm.weight':torch.ones(4)})
    monkeypatch.delenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',raising=False)
    inputs = {str(old/'manifest.json'):sha256(old/'manifest.json')}
    questions=[]
    for qi in range(2):
        q=dict(id=f'q{qi}',original_index=qi,split='val',question='Solve x',ground_truth='2')
        questions.append(q)
        t=trajectory/'trajectories'/f'{qi:05d}.json'
        t.write_text(json.dumps(dict(question_id=q['id'],responses=[dict(score=qi,verifier_status='correct' if qi else 'incorrect',token_ids=[5,5,5],response='xxx')])) )
        inputs[str(t)]=sha256(t)
        meta=dict(y=torch.full((4,),float(qi)),question=torch.full((4,),qi),response=torch.zeros(4,dtype=torch.long),position=torch.tensor([0,1,1,2]))
        files={}
        for name,dim in [('hidden_two_thirds',4),('hidden-block5',4),('hidden-block2',4),('jspace_concat',12)]:
            path=source/name/'sampled'/f'{qi:05d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(meta,x=torch.randn(4,dim)),path);files[str(path.relative_to(source))]=sha256(path)
        marker=dict(id=q['id'],split='val',attempted=1,retained=1,files=files,readouts=[])
        (source/f'question-{qi:05d}.json').write_text(json.dumps(marker))
    for name,dim in [('hidden_two_thirds',4),('jspace_concat',12)]:
        folder=source/name/'heads';folder.mkdir()
        torch.save(dict(state={'weight':torch.randn(1,dim),'bias':torch.zeros(1)},mean=torch.zeros(dim),scale=torch.ones(dim)),folder/'ridge.pt')
    manifest=dict(environment={'test':'1'},inputs=inputs,source=str(old),layers=[4,3,1],target_layer=4,dimension=4,questions=questions,trajectory_source=str(trajectory))
    (source/'manifest.json').write_text(json.dumps(manifest))
    before={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
    args=SimpleNamespace(source=source,out=tmp_path/'view',split='val',questions=0,batch_size=2,device='cpu')
    module.run(args)
    result=json.loads((args.out/'inspection.json').read_text())
    assert len(result['answers'])==2
    assert [p['position'] for p in result['answers'][0]['points']]==[0,1,2]
    assert result['summary']['baseline_block']==4
    assert (args.out/'inspector.html').exists()
    assert before=={str(p):sha256(p) for p in source.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='new output'):
        module.run(args)
