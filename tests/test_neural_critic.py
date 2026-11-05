import importlib.util
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from math_rl.neural_critic import NeuralCritic
from slice_critic.bottleneck import Bottleneck


@pytest.mark.parametrize('joint',[False,True])
def test_fresh_returns_update_correct_parameters_without_actor_gradients(joint):
    torch.manual_seed(42)
    fitted=Bottleneck(7680,768).eval().requires_grad_(False),torch.zeros(7680),torch.ones(7680)
    critic=NeuralCritic(fitted,joint=joint,epochs=2,chunk=4,minibatch=2)
    width=7680 if joint else 768
    features=[np.random.default_rng(i).normal(size=(5,width)).astype('float32') for i in range(2)]
    mask=torch.ones(2,5)
    xs=critic.inputs(features,mask)
    before=critic.values(xs,mask)
    torch.testing.assert_close(before,torch.full((2,5),.5))
    projection=critic.compressor.projection.weight.detach().clone()
    targets=torch.ones(2,5,requires_grad=True)
    threads=torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        metrics=critic.fit_returns(xs,targets,before,mask,1)
    finally: torch.set_num_threads(threads)
    assert targets.grad is None
    assert metrics['mse_after']<metrics['mse_before']
    assert metrics['cumulative_matrices'] is False
    assert torch.equal(projection,critic.compressor.projection.weight) != joint
    torch.testing.assert_close(fitted[0].projection.weight,projection)
    assert critic.compressor.readout.weight.grad is None
    with pytest.raises(ValueError,match='consecutive'):
        critic.fit_returns(xs,targets,before,mask,1)


def test_all_seven_command_settings_keep_actor_and_trace_lambdas_separate(tmp_path):
    path=Path(__file__).resolve().parents[1]/'tools/diagnostics/online_critics.py'
    spec=importlib.util.spec_from_file_location('online_critics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    m=dict(profile='higher_lr',filter=dict(path='/tmp/filter.pt',sha256='abc'),layers=[27,26,24,20,12],rms_eps=1e-6)
    assert len(module.ARMS)==7
    for name,arm in module.ARMS.items():
        cmd=module.command(tmp_path,m,name,501,120)
        if name.startswith('lstd'):
            assert f'experiment.critic_lambda={arm["trace_lambda"]}' in cmd
            assert 'experiment.critic_folds=1' in cmd
        if name=='mlp-joint':
            assert 'actor_rollout_ref.model.critic_feature_view=raw_concat' in cmd
            assert 'experiment.critic_statistics=fresh_neural' in cmd
        if name=='mlp-frozen':
            assert 'actor_rollout_ref.model.critic_feature_view=compressed' in cmd
        assert not any(c.startswith('algorithm.lam=') for c in cmd)
        # Validate each new command with the real Hydra validator, without Ray.
        root=path.parents[2]
        config=yaml.safe_load((root/'configs/training.yaml').read_text())
        def merge(a,b):
            for k,v in b.items():
                if isinstance(v,dict):merge(a.setdefault(k,{}),v)
                else:a[k]=v
        merge(config,yaml.safe_load((root/'configs/online.yaml').read_text()))
        config['critic'].update(strategy='fsdp',use_dynamic_bsz=False,ulysses_sequence_parallel_size=1)
        config['trainer'].update(critic_warmup=0,val_only=False)
        for item in cmd[3:]:
            key,value=item.split('=',1);current=config
            for part in key.split('.')[:-1]:current=current.setdefault(part,{})
            current[key.split('.')[-1]]=float(value) if key.endswith('.lr') else yaml.safe_load(value)
        class Attr(dict): __getattr__=dict.__getitem__
        def node(x):return Attr({k:node(v) for k,v in x.items()}) if isinstance(x,dict) else x
        tree=ast.parse((root/'src/math_rl/main.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate')
        commit='8d9e350ea58c7ad4b50dd14d9dcb50577242c55f'
        scope=dict(json=json,sha256=lambda p:'abc',VERL_COMMIT=commit,
            distribution=lambda n:SimpleNamespace(read_text=lambda n:json.dumps({'vcs_info':{'commit_id':commit}})))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'validator','exec'),scope)
        scope['validate'](node(config))
