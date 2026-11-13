from pathlib import Path
import sys
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/encoder'))
from diagnose_padding import inputs,errors


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
