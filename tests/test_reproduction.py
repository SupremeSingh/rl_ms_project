import copy
import json
from pathlib import Path
import shutil

import pytest
import torch

from math_rl import experiment, reproduction
from math_rl.provenance import sha256, write_json
from math_rl.data import write_online_data, select_questions
from slice_critic.storage import publish, model_identity
from slice_critic.bottleneck import Bottleneck
from slice_critic.config import INSTRUCTION


def questions():
    return [dict(id=f'q{i}',question='Compute '+('a'*i)+'b quantity',ground_truth='1',source_index=i,
                 split='train' if i<16 else 'val' if i==16 else 'test') for i in range(18)]


def setup_model(tmp_path,monkeypatch):
    model=tmp_path/'models/qwen-math';model.mkdir(parents=True)
    (model/'config.json').write_text('{}');(model/'model.safetensors').write_bytes(b'weights')
    monkeypatch.setattr(reproduction,'ROOT',tmp_path)
    monkeypatch.setattr(experiment,'ROOT',tmp_path)
    monkeypatch.setattr(reproduction,'snapshot',lambda p:dict(files={'code':'same'}))
    monkeypatch.setattr(experiment,'snapshot',lambda p:dict(files={'code':'same'}))
    return model


def test_clean_artifact_to_online_preparation_and_overlap_rejection(tmp_path,monkeypatch):
    model=setup_model(tmp_path,monkeypatch)
    data=tmp_path/'prepared';data.mkdir();write_online_data(data/'online',questions())
    write_json(data/'manifest.json',dict(protocol='slice-numina-data-v1',revisions={},
        hashes={str(p.relative_to(data)):sha256(p) for p in (data/'online').iterdir()}))
    manifest=dict(version='fixture',model=dict(id='Qwen/Qwen2.5-Math-1.5B',files=model_identity(model)),
        prompt_instruction=INSTRUCTION,representation_training_families=['offline question'],selection_families=['validation question'],
        features=dict(width=768,layers_zero_based=[27,26,24,20,12],compression='sparse',rms_eps=1e-6))
    checkpoint=dict(compressor=Bottleneck(7680,768).state_dict(),mean=torch.zeros(7680),scale=torch.ones(7680),weights=torch.ones(769))
    artifact=publish(tmp_path/'artifact',manifest,checkpoint,dict(method='platt',slope=1.,intercept=0.),{})
    out=tmp_path/'study';reproduction.prepare_from_artifact(artifact,data,out)
    m=experiment.validate(out)
    assert m['steps']==120 and len(m['conditions'])==6 and m['critic_lambda']==.9
    exported=torch.load(out/'filters/sparse.pt',weights_only=True)
    assert 'weights' not in exported and 'screen_ridge_weights' not in exported
    assert not (out/'selection.json').exists()
    (out/'data/train.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Data changed'):experiment.validate(out)
    from slice_critic.text import family_key
    with pytest.raises(ValueError,match='Overlapping'):
        reproduction.validate_questions(questions(),{family_key(questions()[0]['question'])})


@pytest.mark.parametrize('actual_profile',['higher_lr','more_fitting'])
def test_historical_capture_is_portable_and_preserves_inputs(tmp_path,monkeypatch,actual_profile):
    model=setup_model(tmp_path,monkeypatch)
    source=tmp_path/'old';source.mkdir();write_online_data(source/'data',questions())
    (source/'filters').mkdir();(source/'filters/sparse.pt').write_bytes(b'frozen filter')
    m=dict(protocol='numina-online-v1',conditions=copy.deepcopy(experiment.METHODS),steps=120,critic_lambda=.9,
        seeds=[301,302,303,304,305],layers=[27,26,24,20,12],rms_eps=1e-6,revisions={},
        model_hashes=model_identity(model),data_hashes={p.name:sha256(p) for p in (source/'data').iterdir()},
        filters={'sparse':dict(path=str(source/'filters/sparse.pt'),sha256=sha256(source/'filters/sparse.pt'))})
    write_json(source/'manifest.json',m)
    write_json(source/'selection.json',dict(profile='higher_lr',tuning_training_gpu_hours=1))
    write_json(source/'summary.json',{'completed':True})
    for name in experiment.METHODS:
        for seed in m['seeds']:
            write_json(source/f'{name}-seed{seed}/result.json',dict(updates=120,answers=7680,seed=seed,ppo_profile=actual_profile))
            if experiment.METHODS[name]['method'] in ('ridge','lstd'):
                folder=source/f'{name}-seed{seed}/linear-critic';folder.mkdir()
                (folder/'metrics.jsonl').write_text(''.join(json.dumps(dict(step=step,
                    fitting_mode='cumulative_raw_sum',critic_lambda=.9,epsilon=.01,feature_mode='frozen'))+'\n'
                    for step in range(1,121)))
    before=sha256(source/'manifest.json')
    out=tmp_path/'replay';reproduction.import_final(source,out)
    assert sha256(source/'manifest.json')==before
    assert not list(out.glob('*-seed*')) and not (out/'smoke.json').exists()
    replay_selection=json.loads((out/'selection.json').read_text())
    assert replay_selection['profile']==actual_profile and replay_selection['declared_profile']=='higher_lr'
    shutil.rmtree(source)  # Prove the replay has no runtime dependency on old directories.
    relocated=tmp_path/'relocated';out.rename(relocated)
    saved=experiment.validate(relocated)
    assert saved['filters']['sparse']['path']==str(relocated/'filters/sparse.pt')
    assert (relocated/'historical-summary.json').exists()
    with pytest.raises(ValueError,match='separate'):reproduction.import_final(relocated,relocated)


def test_number_template_split_is_outcome_independent(monkeypatch):
    import math_rl.data as data
    class Tokenizer:
        def encode(self,text,**kw):return [1,2,3]
    monkeypatch.setattr(data,'candidate',lambda q:(dict(question=q['text'],ground_truth='1'),None))
    rows=[dict(text=t) for t in ['Buy 2 pears','Buy 3 pears','Count apples','Divide oranges','Multiply melons']]
    selected,_=select_questions(rows,{'buy NUMBER pears'},Tokenizer(),dict(train=1,val=1,test=1))
    assert len(selected)==3 and all('Buy' not in q['question'] for q in selected)
    assert len({data.family_key(q['question']) for q in selected})==3
    with pytest.raises(ValueError,match='Insufficient'):
        select_questions(rows,set(),Tokenizer(),dict(train=10))


def test_from_scratch_preparation_pins_data_and_emits_disjoint_trainable_config(tmp_path,monkeypatch):
    import datasets
    import huggingface_hub
    import transformers
    import math_rl.data as data
    from types import SimpleNamespace
    from slice_critic import TrainingConfig
    from slice_critic.text import family_key
    monkeypatch.setattr(data,'ROOT',tmp_path)
    write_json(tmp_path/'configs/assets.json',dict(model_revision='a'*40))
    rows=[dict(problem='Compute '+('a'*i)+'b quantity',solution=r'The result is \boxed{1}.',answer='1',
        source='cn_k12',synthetic=False,question_type='math-word-problem',problem_is_valid='Yes',solution_is_valid='Yes')
        for i in range(1,45)]
    revisions={name:'b'*40 for name in (data.DATASET,'openai/gsm8k','EleutherAI/hendrycks_math')}
    def load(name,*a,**kw):
        assert kw['revision']==revisions[name]
        if name==data.DATASET:return rows
        if name=='openai/gsm8k':return [{'question':'Separate excluded GSM problem'}]
        return [{'problem':'Separate excluded MATH problem'}]
    monkeypatch.setattr(datasets,'load_dataset',load)
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**kw:SimpleNamespace(encode=lambda *a,**kw:[1,2]))
    out=tmp_path/'prepared'
    data.prepare(out,tmp_path/'model',revisions,offline_count=20,counts=dict(train=16,val=1,test=1))
    config=TrainingConfig(**json.loads((out/'critic-config.json').read_text())).validate()
    assert config.max_questions==20 and config.responses_per_question==8 and config.model_revision=='a'*40
    offline=[json.loads(line) for line in (out/'critic.jsonl').read_text().splitlines()]
    online=json.loads((out/'online/questions.json').read_text())['questions']
    assert not {family_key(q['problem']) for q in offline} & {family_key(q['question']) for q in online}
    saved=json.loads((out/'manifest.json').read_text())
    assert saved['revisions']==revisions
    assert all(sha256(out/name)==digest for name,digest in saved['hashes'].items())
