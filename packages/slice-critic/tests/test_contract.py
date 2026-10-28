"""Public API, early artifact rejection and raw-sum mathematical contracts."""
import os
import subprocess
import sys

import pytest
import torch

from slice_critic import (Critic, CumulativeCritic, export_ppo_features, inspect_artifact,
                          load_ppo_features, encode_rollout_features)
from slice_critic.artifacts import validate_payload
from slice_critic.bottleneck import Bottleneck
from slice_critic.config import INSTRUCTION
from slice_critic.storage import publish, sha256


def payload():
    manifest = dict(version='contract-test', model=dict(id='toy', files={'config.json': 'hash'}),
        prompt_instruction=INSTRUCTION, representation_training_families=['train'], selection_families=['val'],
        features=dict(width=4, layers_zero_based=[2, 1], blocks_one_based=[3, 2], rms_eps=1e-6,
                      compression='sparse', input_dimension=6))
    checkpoint = dict(mean=torch.zeros(6), scale=torch.ones(6), weights=torch.zeros(5),
                      compressor=Bottleneck(6, 4).state_dict())
    return manifest, checkpoint, dict(method='platt', slope=1., intercept=0.)


@pytest.mark.parametrize('method',['ridge','lstd'])
def test_public_one_head_fits_all_answers_before_prediction(method):
    from slice_critic.cumulative import raw_design, trajectory_statistics
    critic=CumulativeCritic(['one question'],method=method,trace_lambda=.9,folds=1)
    features=[torch.tensor([[.1],[.7]]),torch.tensor([[.3],[.9]])]
    rewards=torch.tensor([[0.,1.],[0.,0.]])
    a=torch.zeros(2,2,dtype=torch.float64);b=torch.zeros(2,dtype=torch.float64)
    for features_i,outcome in zip(features,[1.,0.]):
        x=raw_design(features_i)
        ai,bi=(x.T@x,x.sum(0)*outcome) if method=='ridge' else trajectory_statistics(features_i,outcome,.9)[:2]
        a+=ai;b+=bi
    for step in (1,2):
        values,heads,metric=critic.update(features,rewards,torch.ones_like(rewards),['one question']*2,step)
        expected=torch.linalg.solve(step*a+.01*torch.eye(2,dtype=torch.float64),step*b)
        assert len(heads)==1 and not metric['cross_fitted']
        torch.testing.assert_close(values,torch.stack([(raw_design(x)@expected).float() for x in features]))


def test_inspect_is_cpu_only_and_preserves_artifact_and_rng(tmp_path):
    manifest, checkpoint, calibration = payload()
    artifact = publish(tmp_path/'artifact', manifest, checkpoint, calibration, {'test': {}})
    hashes = {p.name: sha256(p) for p in artifact.iterdir()}
    rng = torch.random.get_rng_state()
    assert inspect_artifact(artifact)['validated']
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert hashes == {p.name: sha256(p) for p in artifact.iterdir()}


@pytest.mark.parametrize('change', ['layers', 'numbering', 'scale', 'weights', 'calibration', 'families', 'width'])
def test_reject_malformed_artifacts_before_encoder_loading(tmp_path, monkeypatch, change):
    manifest, checkpoint, calibration = payload()
    if change == 'layers': manifest['features']['layers_zero_based'] = [2, 2]
    if change == 'numbering': manifest['features']['blocks_one_based'] = [2, 1]
    if change == 'scale': checkpoint['scale'][0] = 0
    if change == 'weights': checkpoint['weights'] = torch.zeros(4)
    if change == 'calibration': calibration['slope'] = float('inf')
    if change == 'families': manifest['selection_families'] = ['train']
    if change == 'width': manifest['features']['width'] = True
    # Nonfinite calibration cannot be serialized as JSON; test validator directly.
    if change == 'calibration':
        with pytest.raises(ValueError): validate_payload(manifest, checkpoint, calibration)
        return
    artifact = publish(tmp_path/'artifact', manifest, checkpoint, calibration, {})
    import slice_critic.backend as backend
    monkeypatch.setattr(backend, 'TransformersBackend', lambda *a, **kw: pytest.fail('Encoder loaded too early'))
    with pytest.raises(ValueError): Critic.load(artifact, device='cpu')


def test_public_ppo_exports_are_existing_implementations():
    from slice_critic.bridge import export_filter
    from slice_critic.online import bottleneck_prefixes, load_filter
    assert export_ppo_features is export_filter
    assert load_ppo_features is load_filter
    assert encode_rollout_features is bottleneck_prefixes


@pytest.mark.parametrize('method', ['ridge', 'lstd'])
def test_cumulative_public_api_matches_explicit_raw_sums_and_cross_fitting(method):
    from slice_critic.cumulative import trajectory_statistics, raw_design
    critic = CumulativeCritic(['a', 'b', 'c', 'd'], seed=42, epsilon=.01, trace_lambda=.99, method=method)
    ids = [next(q for q, fold in critic.assignment.items() if fold == f) for f in (0, 1)]
    features = [torch.tensor([[.1, .2], [.3, .4]]), torch.tensor([[.2, .7], [.5, .8]])]
    rewards = torch.tensor([[0., 1.], [0., 0.]])
    mask = torch.ones_like(rewards)
    for step in (1, 2):
        values, _, _ = critic.update(features, rewards, mask, ids, step)
        for f in (0, 1):
            phi = raw_design(features[f])
            outcome = float(rewards[f].sum())
            a, b = (phi.T@phi, phi.sum(0)*outcome) if method=='ridge' else trajectory_statistics(features[f], outcome, .99)[:2]
            torch.testing.assert_close(critic.stats[f]['a'], step*a)
            torch.testing.assert_close(critic.stats[f]['b'], step*b)
            other = critic.stats[1-f]
            weights = torch.linalg.solve(other['a']+.01*torch.eye(3, dtype=torch.float64), other['b'])
            torch.testing.assert_close(values[f].double(), phi@weights, rtol=1e-5, atol=1e-7)
        assert not values.requires_grad


def test_verifier_entrypoint_import_does_not_load_torch():
    # Match the verifier environment, where Torch is deliberately absent.
    code = "import sys; import slice_critic.verifier; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], env=dict(os.environ), check=True)
