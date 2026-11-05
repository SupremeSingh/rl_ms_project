"""Single-owner lifecycle, fail-closed submission, and parallel smoke aggregation."""
import hashlib
import importlib.util
import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from math_rl.neural_critic import NeuralCritic,NeuralCriticService
from slice_critic.bottleneck import Bottleneck

TOOLS=Path(__file__).resolve().parents[1]/'tools/diagnostics'


def load(name):
    spec=importlib.util.spec_from_file_location(name,TOOLS/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_service_keeps_exactly_one_fresh_rollout_and_does_not_change_rng():
    fitted=Bottleneck(7680,768).eval().requires_grad_(False),torch.zeros(7680),torch.ones(7680)
    rng=torch.get_rng_state().clone()
    service=NeuralCriticService(fitted,device='cpu',joint=False,minibatch=2,chunk=4)
    assert torch.equal(rng,torch.get_rng_state())
    x=[np.ones((4,768),dtype='float32') for _ in range(2)];mask=torch.ones(2,4)
    with pytest.raises(ValueError,match='Predict before'):
        service.fit(torch.ones(2,4),mask,1)
    reply=service.predict(x,mask,1)
    assert reply['owner_rank']==0 and reply['values'].device.type=='cpu'
    with pytest.raises(ValueError,match='previous fit'):
        service.predict(x,mask,1)
    with pytest.raises(ValueError,match='paired'):
        service.fit(torch.ones(2,4),mask,2)
    with torch.no_grad():
        result=service.fit(torch.ones(2,4),mask,1,checkpoint=True)
    assert result['metrics']['mse_after']<result['metrics']['mse_before']
    assert service.pending is None
    assert all(v.device.type=='cpu' for v in result['state'].values())
    service.predict(x,mask,2)
    result=service.fit(torch.zeros(2,4),mask,2)
    assert result['state'] is None and service.pending is None


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Cluster CUDA acceptance required')
def test_cuda_service_and_cuda_rng_isolation():
    fitted=Bottleneck(7680,768).eval().requires_grad_(False),torch.zeros(7680),torch.ones(7680)
    rng=torch.cuda.get_rng_state().clone()
    service=NeuralCriticService(fitted,device='cuda',joint=True,minibatch=2,chunk=4)
    assert torch.equal(rng,torch.cuda.get_rng_state())
    x=[np.ones((4,7680),dtype='float32') for _ in range(2)];mask=torch.ones(2,4)
    reply=service.predict(x,mask,1)
    assert reply['backend']=='cuda'
    result=service.fit(torch.ones(2,4),mask,1,True)
    assert result['backend']=='cuda' and result['metrics']['mse_after']<result['metrics']['mse_before']
    assert all(v.device.type=='cpu' for v in result['state'].values())


def test_parallel_smoke_requires_all_seven_matching_records(tmp_path,monkeypatch):
    module=load('online_critics')
    (tmp_path/'manifest.json').write_text('{}')
    def execute(out,m,name,seed,steps,smoke):
        folder=out/f'smoke-{name}';folder.mkdir()
        (folder/'result.json').write_text(json.dumps(dict(name=name,steps=steps)))
    monkeypatch.setattr(module,'execute',execute)
    monkeypatch.setattr(module,'audit',lambda *args,**kwargs:dict(passed=True))
    for i in range(6):module.smoke_task(tmp_path,{},i)
    with pytest.raises(FileNotFoundError):module.smoke_gate(tmp_path,{})
    assert not (tmp_path/'smoke.json').exists()
    module.smoke_task(tmp_path,{},6)
    module.smoke_gate(tmp_path,{})
    gate=json.loads((tmp_path/'smoke.json').read_text())
    assert gate['passed'] and set(gate['checks'])==set(module.ARMS)
    (tmp_path/'smoke-ridge/result.json').write_text('{}')
    with pytest.raises(ValueError,match='stale'):module.smoke_gate(tmp_path,{})


def submission_study(tmp_path):
    out=tmp_path/'study';out.mkdir()
    (out/'manifest.json').write_text(json.dumps(dict(protocol='slice-seven-critics-gpu-v1',
        conditions={str(i):{} for i in range(7)},seeds=[501,502,503])))
    return out


def test_submission_parallel_arrays_and_afterok_dependencies(tmp_path):
    module=load('submit_critic_study');out=submission_study(tmp_path);commands=[]
    def submit(command,**kw):
        commands.append(command)
        return SimpleNamespace(stdout=f'{100+len(commands)};duke\n')
    jobs=module.submit_chain(out,6,7,submit=submit)
    assert len(commands)==4 and jobs['report']=='104'
    assert '--array=0-6%7' in commands[0]
    assert '--dependency=afterok:101' in commands[1] and '--gres=none' in commands[1]
    assert '--dependency=afterok:102' in commands[2] and '--array=0-20%6' in commands[2]
    assert '--dependency=afterok:103' in commands[3]
    with pytest.raises(FileExistsError):module.submit_chain(out,submit=submit)


def test_uncertain_submission_never_launches_dependents_or_retries(tmp_path):
    module=load('submit_critic_study');out=submission_study(tmp_path);commands=[]
    def submit(command,**kw):
        commands.append(command)
        raise subprocess.TimeoutExpired(command,60)
    with pytest.raises(RuntimeError,match='may have reached Slurm'):
        module.submit_chain(out,submit=submit)
    assert len(commands)==1 and (out/'.submission-chain/smoke').exists()
    assert not (out/'jobs.tsv').exists()


def test_smoke_only_then_train_mode(tmp_path):
    module=load('submit_critic_study');out=submission_study(tmp_path);commands=[]
    def submit(command,**kw):
        commands.append(command)
        return SimpleNamespace(stdout=str(100+len(commands)))
    module.submit_chain(out,mode='smoke',submit=submit)
    assert len(commands)==2
    (out/'smoke.json').write_text(json.dumps(dict(passed=True,
        manifest_sha256=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest())))
    module.submit_chain(out,mode='train',submit=submit)
    assert len(commands)==4 and '--dependency=afterok:102' in commands[2]


def test_trainer_rpc_hook_preserves_pre_fit_gae_and_actor_order(tmp_path):
    from slice_critic.storage import write_json
    import time
    events=[]
    class Attr(dict): __getattr__=dict.__getitem__
    fitted=Bottleneck(7680,768).eval().requires_grad_(False),torch.zeros(7680),torch.ones(7680)
    service=NeuralCriticService(fitted,device='cpu',joint=False,minibatch=2,chunk=4)
    mask=torch.ones(2,4)
    data=SimpleNamespace(batch=dict(response_mask=mask,token_level_rewards=torch.zeros(2,4)),
        non_tensor_batch=dict(pretoken_features=np.ones((2,4,768),dtype='float32'),critic_encoder_epoch=np.zeros(2)))
    class Group:
        def neural_predict(self,features,mask,step,settings):
            events.append('predict')
            reply=service.predict(features,mask,step);reply['backend']='cuda'
            return reply
        def neural_fit(self,returns,mask,step,checkpoint):
            events.append('fit')
            assert not returns.requires_grad
            reply=service.fit(returns,mask,step,checkpoint);reply['backend']='cuda'
            return reply
    def gae(data,*args,**kwargs):
        events.append('gae')
        data.batch['advantages']=torch.full((2,4),.25)
        data.batch['returns']=torch.ones(2,4,requires_grad=True)
        return data
    class Parent:
        def __init__(self,config):
            self.config=config;self.global_steps=1;self.total_training_steps=1;self.actor_rollout_wg=Group()
        def fit(self):
            result=ray_trainer.compute_advantage(data)
            events.append('actor')
            torch.testing.assert_close(result.batch['values'],torch.full((2,4),.5))
            torch.testing.assert_close(result.batch['advantages'],torch.full((2,4),.25))
            self.global_steps=2
    ray_trainer=SimpleNamespace(RayPPOTrainer=Parent,compute_advantage=gae)
    path=TOOLS.parents[1]/'src/math_rl/trainer.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MathTrainer')
    scope=dict(ray_trainer=ray_trainer,Path=Path,torch=torch,json=json,time=time,write_json=write_json)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'trainer_hook','exec'),scope)
    config=Attr(experiment=Attr(method='mlp_frozen',neural_epochs=2,neural_head_lr=.0003,neural_compressor_lr=.000003),
        data=Attr(seed=42),actor_rollout_ref=Attr(model=Attr(critic_feature_mode='frozen')),
        trainer=Attr(default_local_dir=str(tmp_path)))
    trainer=scope['MathTrainer'](config)
    trainer.fit()
    assert events==['predict','gae','fit','actor']
    assert ray_trainer.compute_advantage is gae
    assert trainer.completed_updates==1
    assert (tmp_path/'neural-critic/latest.pt').exists()
