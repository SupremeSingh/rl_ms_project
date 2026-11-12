import importlib.util
from pathlib import Path
import json
from types import SimpleNamespace
import torch
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f'tools/encoder/{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class Tokenizer:
    def decode(self,ids,**kw):return ' '.join(map(str,ids))
    def encode(self,text,**kw):return list(map(int,text.split()))


def test_adaptation_corpus_never_contains_val_or_test():
    m=module('study')
    rows=[dict(split='train',prompt_ids=[],token_ids=list(range(128*8))),
          dict(split='val',prompt_ids=[],token_ids=[999]*256),
          dict(split='test',prompt_ids=[],token_ids=[999]*256)]
    lines=m.corpus(rows,Tokenizer())
    assert len(lines)==8 and not any(line.startswith('999 999') for line in lines)
    assert all(len(Tokenizer().encode(line))==128 for line in lines)


def test_official_configs_use_training_only_equal_length_blocks(tmp_path):
    m=module('study');a,b=m.trainer_configs(dict(adaptation_steps=2,model='/base'),tmp_path)
    assert a['train_file']==a['validation_file'] and not a['do_eval']
    assert a['attn_implementation']=='flash_attention_2' and not a['line_by_line']
    assert b['dataset_name']=='Wiki1M' and b['bidirectional']
    assert a['save_steps']==b['save_steps']==2
    assert a['max_seq_length']==b['max_seq_length']==128


def test_existing_slice_and_ridge_are_reused_and_test_is_not_fit(tmp_path,monkeypatch):
    m=module('study')
    # Deliberately invalid test labels: fitting must not read them.
    rows=[dict(x=torch.randn(12,10),y=torch.tensor([0.,1.]*6),split=s,question=s) for s in ('train','val')]
    rows.append(dict(x=torch.randn(2,10),y=torch.tensor([99.,99.]),split='test',question='test'))
    monkeypatch.setattr(m,'feature_rows',lambda *args:rows)
    for arm in m.ARMS:(tmp_path/arm).mkdir()
    m.fit(tmp_path,dict(seeds=[42],compressor_epochs=1))
    for arm in m.ARMS:
        head=torch.load(tmp_path/arm/'fit-42/ridge.pt',weights_only=True)
        assert head['folds']==1 and head['epsilon']==.01 and head['states']==12
        assert head['a'].shape==(769,769)
        torch.testing.assert_close((head['a']+.01*torch.eye(769))@head['weights'],head['b'],rtol=1e-6,atol=1e-6)


def test_submission_has_array_and_fail_closed_dependencies(tmp_path,monkeypatch):
    m=module('submit_study');calls=[]
    def run(cmd,**kw):calls.append(cmd);return SimpleNamespace(stdout=str(len(calls)))
    monkeypatch.setattr(m.subprocess,'run',run)
    m.submit(tmp_path/'out','smoke',tmp_path/'src',tmp_path/'data',tmp_path/'sanity',tmp_path/'official')
    assert len(calls)==5 and '--array=0-2%3' in calls[2]
    for i,cmd in enumerate(calls):
        assert '--gres=gpu:a5000:1' in cmd
        if i:assert '--dependency=afterok:'+str(i) in cmd
    with pytest.raises(FileExistsError):m.submit(tmp_path/'out','smoke',tmp_path/'src',tmp_path/'data',tmp_path/'sanity',tmp_path/'official')
