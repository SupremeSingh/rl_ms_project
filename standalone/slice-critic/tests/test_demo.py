"""Small synthetic fixtures test the adapter, not research performance."""
import json
from types import SimpleNamespace

import pytest
import torch

from slice_critic import demo
from slice_critic.storage import model_identity, sha256


def save(path, obj):
    path.write_text(json.dumps(obj))


@pytest.fixture
def study(tmp_path):
    folder = tmp_path / 'study'
    folder.mkdir()
    save(folder / 'manifest.json', dict(protocol='fixed-policy-critics-v1', shards=1,
                                      recipe=dict(samples=128)))
    identity = dict(manifest_sha256=sha256(folder / 'manifest.json'))
    states = [dict(state_id='a', question_id='q1', prompt_ids=[1], prefix_ids=[]),
              dict(state_id='b', question_id='q1', prompt_ids=[1], prefix_ids=[2]),
              dict(state_id='c', question_id='q2', prompt_ids=[3], prefix_ids=[])]
    save(folder / 'bank.json', dict(identity, states=states))
    identity['bank_sha256'] = sha256(folder / 'bank.json')
    save(folder / 'reference-0.json', dict(identity, states=[
        dict(state_id=s['state_id'], samples=128, successes=0) for s in states]))
    for arm in demo.ARMS:
        save(folder / f'predictions-{arm}.json', dict(identity, states=[
            dict(state_id=s['state_id'], prediction=v) for s, v in zip(states, [0., 0., 1.])]))
    return folder


def test_replay_equal_question_metrics_and_no_source_writes(study, tmp_path):
    before = {p.name: sha256(p) for p in study.iterdir()}
    m, identity, states, refs, check = demo.load_study(study)
    p = demo.replay(study, states, check)
    assert demo.summarize(states, refs, p)['slice-token'] == pytest.approx(
        dict(mae=.5, rmse=2 ** -.5))
    out = tmp_path / 'demo-output'
    demo.main(['--study', str(study), '--out', str(out)])
    assert (out / 'predictions.csv').exists()
    assert before == {p.name: sha256(p) for p in study.iterdir()}


@pytest.mark.parametrize('kind', ['identity', 'missing', 'duplicate', 'counts'])
def test_reject_corrupt_input(study, kind):
    path = study / ('reference-0.json' if kind == 'counts' else 'predictions-slice-token.json')
    record = json.loads(path.read_text())
    if kind == 'identity': record['bank_sha256'] = 'wrong'
    if kind == 'missing': record['states'].pop()
    if kind == 'duplicate': record['states'].append(record['states'][0])
    if kind == 'counts': record['states'][0]['samples'] = 64
    save(path, record)
    with pytest.raises(ValueError):
        _, _, states, _, check = demo.load_study(study)
        demo.replay(study, states, check)


def test_reject_nonfinite_and_output_inside_source(study):
    _, _, states, refs, check = demo.load_study(study)
    p = demo.replay(study, states, check)
    p['slice-token']['a'] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'): demo.summarize(states, refs, p)
    with pytest.raises(SystemExit): demo.main(['--study', str(study), '--out', str(study / 'new')])


def test_recompute_adapter_uses_prefix_and_separate_heads(study, tmp_path, monkeypatch):
    from slice_critic import backend, inference
    import transformers
    encoder = tmp_path / 'encoder'
    encoder.mkdir()
    save(encoder / 'config.json', {})
    (encoder / 'model.safetensors').write_bytes(b'fixture')
    transformer = study / 'transformer'
    transformer.mkdir()
    save(transformer / 'config.json', {})
    (transformer / 'model.safetensors').write_bytes(b'fixture')
    heads = {a: dict(weights=torch.tensor([1., 1., float(i)])) for i, a in enumerate(demo.ARMS[:-1])}
    torch.save(heads, study / 'ridge.pt')
    save(study / 'linear.json', dict(manifest_sha256=sha256(study / 'manifest.json'),
                                   sha256=sha256(study / 'ridge.pt')))
    save(study / 'neural.json', dict(manifest_sha256=sha256(study / 'manifest.json'),
                                   model_identity=model_identity(transformer)))
    seen = []
    class FakeBackend:
        num_layers, hidden_size = 3, 2
        def __init__(self, *a, **kw): pass
        def state_features(self, tokens, layers):
            seen.append(tokens)
            return torch.tensor([[1., 2., 3., 4.]])
    class FakeModel:
        config = SimpleNamespace(num_labels=1, max_position_embeddings=100)
        def to(self, *a): return self
        def eval(self): return self
        def requires_grad_(self, *a): return self
        def __call__(self, input_ids, **kw):
            return SimpleNamespace(logits=input_ids.float().unsqueeze(-1))
    monkeypatch.setattr(backend, 'TransformersBackend', FakeBackend)
    monkeypatch.setattr(inference, 'load_filter', lambda *a, **kw:
        (SimpleNamespace(features=lambda x: torch.tensor([[.5, .5]])), torch.zeros(4), torch.ones(4)))
    monkeypatch.setattr(transformers.AutoModelForTokenClassification, 'from_pretrained',
        lambda *a, **kw: (FakeModel(), dict(missing_keys=[], unexpected_keys=[])))
    m, _, states, _, _ = demo.load_study(study)
    m.update(base_identity=model_identity(encoder), layers=[2, 1], filter_sha256='fixture', rms_eps=1e-6)
    p = demo.recompute(study, m, states, encoder, 'cpu')
    assert seen == [[1], [1, 2], [3]]
    assert p['slice-token']['a'] == 1.
    assert p['slice-chunk']['a'] == 2.
    assert p['rms-layer-token']['a'] == 9.
    assert p['transformer']['b'] == 2.
