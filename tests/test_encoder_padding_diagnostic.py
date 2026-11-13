from pathlib import Path
import sys
import torch
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/encoder'))
from diagnose_padding import inputs,errors,diagnostic_inputs
import compare as c
from diagnose_precision import bidirectional_mask
from diagnose_precision import validate_indices,run_workers,CASES
import diagnose_precision
from types import SimpleNamespace


def test_padding_variants_preserve_real_tokens_and_align_positions():
    for side in ('left','right'):
        args,selection=inputs([3,4,5],99,padding=4,side=side,aligned=True,device='cpu')
        assert args['input_ids'][0,selection].tolist()==[3,4,5]
        assert args['position_ids'][0,selection].tolist()==[0,1,2]
        assert args['attention_mask'].sum()==3
    shifted,_=inputs([3,4,5],99,shift=16,device='cpu')
    assert shifted['position_ids'].tolist()==[[16,17,18]]


def test_diagnostic_reports_failure_without_approving_padding():
    a=torch.ones(4,8)
    assert errors(a,a)['passed']
    r=errors(a,a*5)
    assert not r['passed'] and r['fraction_outside_tolerance']==1 and r['max_abs']==4


def test_read_only_diagnostic_relocates_mount_alias_but_requires_same_model(tmp_path,monkeypatch):
    root=tmp_path/'rl_ms_project';model=root/'models/qwen-math';model.mkdir(parents=True)
    c.write_json(model/'config.json',{'model_type':'qwen2'})
    (model/'model.safetensors').write_bytes(b'fixed test weights')
    out=tmp_path/'old-study'
    m=dict(protocol=c.PROTOCOL,model='/missing/project/rl_ms_project/models/qwen-math',
           base_identity=c.model_identity(model),question_metadata_path='/missing/unrelated/manifest.json')
    c.seal(out,'manifest.json',m)
    monkeypatch.setattr(c,'ROOT',root)
    before=(out/'manifest.json').read_bytes()
    _,path=diagnostic_inputs(out)
    assert path==model and (out/'manifest.json').read_bytes()==before
    (model/'model.safetensors').write_bytes(b'changed weights')
    with pytest.raises(ValueError,match='differs'):diagnostic_inputs(out)


def test_dense_precision_reference_mask_is_bidirectional_and_excludes_pad_keys():
    mask=bidirectional_mask(torch.tensor([[0,0,1,1,1]]),torch.float32)
    assert mask.shape==(1,1,5,5)
    assert torch.all(mask[0,0,:,0:2]==torch.finfo(torch.float32).min)
    assert torch.all(mask[0,0,:,2:]==0)
    # In particular, the first real query can see the final real key.
    assert mask[0,0,2,4]==0


def test_eager_shift_rejected_on_cpu_before_invalid_rotary_index():
    args,_=inputs([3,4,5],99,shift=16,device='cpu')
    with pytest.raises(ValueError,match='rotary cache'):validate_indices(args,100,dense=True)
    validate_indices(args,100,dense=False)
    padded,_=inputs([3,4,5],99,padding=16,device='cpu')
    validate_indices(padded,100,dense=True)
    with pytest.raises(ValueError,match='vocabulary'):validate_indices(args,4,dense=False)


def test_precision_workers_continue_after_cuda_process_failure(tmp_path,monkeypatch):
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        name=cmd[cmd.index('--case')+1]
        if name!='fp32-eager':
            c.write_json(Path(cmd[cmd.index('--out')+1]),dict(completed=name))
        assert kwargs['env']['CUDA_LAUNCH_BLOCKING']=='1'
        return SimpleNamespace(returncode=1 if name=='fp32-eager' else 0)
    monkeypatch.setattr(diagnose_precision.subprocess,'run',run)
    report=run_workers(tmp_path/'study',tmp_path/'result.json',tmp_path/'reference.pt')
    assert len(calls)==4 and list(report)==list(CASES)
    assert report['fp32-eager']['exit_code']==1
    assert report['fp16-flash']['completed']=='fp16-flash'
