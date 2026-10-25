import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from slice_critic import Critic, State, TrainingConfig, train_critic, CumulativeCritic
from slice_critic.config import select_layers
from slice_critic.storage import model_identity, read_artifact, read_json, sha256
from slice_critic.legacy import export_filter
from slice_critic.inference import load_filter


class ToyBackend:
    """Deterministic causal encoder and sampled answers; no network or GPU needed."""
    num_layers = 28
    hidden_size = 4
    max_context = 4096
    rms_eps = 1e-6

    def __init__(self, path):
        self.path = path
        path.mkdir(exist_ok=True)
        (path/'config.json').write_text('{}')
        (path/'model.safetensors').write_bytes(b'toy model')
        self.model_files = model_identity(path)
        self.generated = 0
        self.fail_at = None
        self.feature_calls = 0
        self.fail_feature_at = None

    def prompt_ids(self, question):
        return [3, len(question) % 17 + 4, 2]

    def prefix_ids(self, prefix):
        return [5] if prefix == 'good' else []

    def state_features(self, tokens, layers):
        h = torch.tensor([[tokens[1]/20, tokens[-1]/10, len(tokens)/10, 1.]])
        return torch.cat([h*(1+layer/28) for layer in layers], 1)

    def trajectory_features(self, prompt, answer, positions, layers):
        self.feature_calls += 1
        if self.feature_calls == self.fail_feature_at:
            raise RuntimeError('simulated interrupted feature extraction')
        return torch.cat([self.state_features(prompt+answer[:p], layers) for p in positions])

    def generate(self, prompt, count, max_tokens, seed):
        self.generated += 1
        if self.generated == self.fail_at:
            raise RuntimeError('simulated interrupted generation')
        return [dict(token_ids=[5 if i % 2 else 9, 7, 2],
                     response='Answer: 1' if i % 2 else 'Answer: 0', capped=False)
                for i in range(count)]


def grader(responses, reference):
    return [dict(status='correct' if r.endswith(reference) else 'incorrect',
                 diagnostic_reward=int(r.endswith(reference)), extracted=r[-1]) for r in responses]


def fixture(tmp_path, method='ridge'):
    data = tmp_path/'data.jsonl'
    data.write_text('\n'.join(json.dumps(dict(problem=f'Compute {"a"*i}b quantity.', answer='1'))
                              for i in range(1, 41)))
    config = TrainingConfig(dataset=str(data), max_questions=40, responses_per_question=2,
                            max_new_tokens=4, num_slices=3, compression_width=4,
                            epochs=2, batch_size=32, method=method)
    return config, ToyBackend(tmp_path/'model')


@pytest.mark.parametrize('base,count,expected', [
    (1,5,[27,26,25,24,23]), (2,5,[27,26,24,20,12]), (3,3,[27,25,19]),
])
def test_layers(base, count, expected):
    assert select_layers(28, base, count) == expected


@pytest.mark.parametrize('base,count', [(3,5), (.5,3), (float('nan'),3), (2,29)])
def test_invalid_layer_selection(base, count):
    with pytest.raises(ValueError): select_layers(28,base,count)


@pytest.mark.parametrize('method', ['ridge','lstd'])
def test_training_loading_scoring_export_and_resume(tmp_path, monkeypatch, method):
    config, backend = fixture(tmp_path,method)
    out = tmp_path/'trained'
    artifact = train_critic(config, out, device='cpu', backend=backend, grader=grader)
    m, checkpoint, calibration = read_artifact(artifact)
    assert not any(set(m['split_question_ids'][a]) & set(m['split_question_ids'][b])
                   for a,b in [('train','val'),('train','test'),('val','test')])
    assert m['features']['blocks_one_based'] == [28,27,25]
    evaluation = read_json(artifact/'evaluation.json')
    assert evaluation['generated_answers'] == 80
    assert evaluation['head_fit']['answers'] == 48
    assert evaluation['head_fit']['transitions'] == 144
    assert evaluation['head_fit']['relative_residual'] < 1e-8
    import slice_critic.backend as backend_module
    monkeypatch.setattr(backend_module, 'TransformersBackend', lambda *a,**kw: backend)
    critic = Critic.load(artifact, device='cpu')
    states = [State('Compute ab quantity.'), State('Compute ab quantity.',prefix='good')]
    features = critic.encode_batch(states)
    assert features.shape == (2,4) and not features.requires_grad
    predictions = critic.score_batch(states)
    phi = torch.cat([features.double(), torch.ones(2,1)],1)
    assert [p['value'] for p in predictions] == pytest.approx((phi@checkpoint['weights']).tolist())
    assert all(0 <= p['probability'] <= 1 for p in predictions)
    assert critic.score_batch([]) == []
    assert torch.equal(critic.encode_batch([State('Compute ab quantity.',prefix_token_ids=[5])]), features[1:])
    with pytest.raises(ValueError,match='OR exact'):
        critic.score_batch([State('Question',prefix='good',prefix_token_ids=[5])])
    before = sha256(out/'head-statistics.pt')
    assert train_critic(config,out,device='cpu',backend=backend,grader=grader) == artifact
    assert backend.generated == 40 and sha256(out/'head-statistics.pt') == before
    # The legacy PPO adapter intentionally supports only the established 768 basis.
    with pytest.raises(ValueError,match='width 768'): export_filter(artifact,tmp_path/'filter.pt')
    from dataclasses import replace
    with pytest.raises(ValueError,match='Configuration changed'):
        train_critic(replace(config,seed=12),out,device='cpu',backend=backend,grader=grader)
    (artifact/'calibration.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'): Critic.load(artifact,device='cpu')


def test_generation_recovery_does_not_double_count(tmp_path):
    config, backend = fixture(tmp_path)
    backend.fail_at = 3
    with pytest.raises(RuntimeError,match='interrupted'):
        train_critic(config,tmp_path/'run',device='cpu',backend=backend,grader=grader)
    backend.fail_at = None
    artifact = train_critic(config,tmp_path/'run',device='cpu',backend=backend,grader=grader)
    assert backend.generated == 41  # Forty questions plus one failed call; two saved calls reused.
    assert read_json(artifact/'evaluation.json')['head_fit']['answers'] == 48


def test_head_recovery_does_not_double_count_checkpointed_trajectories(tmp_path):
    config, backend = fixture(tmp_path)
    # 64 sampled train/validation extractions, then 32 complete all-token answers.
    backend.fail_feature_at = 97
    with pytest.raises(RuntimeError,match='interrupted feature'):
        train_critic(config,tmp_path/'run',device='cpu',backend=backend,grader=grader)
    from slice_critic.storage import load_tensor
    assert load_tensor(tmp_path/'run/head-statistics.pt')['answers'] == 32
    backend.fail_feature_at = None
    artifact = train_critic(config,tmp_path/'run',device='cpu',backend=backend,grader=grader)
    assert read_json(artifact/'evaluation.json')['head_fit']['answers'] == 48
    assert backend.generated == 40


def test_exported_768_basis_matches_existing_ppo_loader(tmp_path):
    from slice_critic.bottleneck import Bottleneck
    from slice_critic.storage import publish
    backend = ToyBackend(tmp_path/'model')
    model = Bottleneck(12,768).eval()
    manifest = dict(version='toy',model=dict(id='toy',files=backend.model_files),
                    features=dict(width=768,compression='sparse',layers_zero_based=[27,26,24],rms_eps=1e-6),
                    representation_training_families=['train'],selection_families=['val'])
    artifact = publish(tmp_path/'artifact',manifest,
        dict(compressor=model.state_dict(),mean=torch.zeros(12),scale=torch.ones(12),weights=torch.zeros(769)),
        dict(method='platt',slope=1.,intercept=0.),{})
    exported = export_filter(artifact,tmp_path/'filter.pt')
    restored,mean,scale = load_filter(exported['path'],exported['sha256'],
        layers=[27,26,24],eps=1e-6,model_path=backend.path)
    x = torch.randn(3,12)
    torch.testing.assert_close(restored.features((x-mean)/scale),model.features(x))
    with pytest.raises(ValueError,match='Configured layers'):
        load_filter(exported['path'],exported['sha256'],layers=[27,26,23])
    with pytest.raises(ValueError,match='new filter path'): export_filter(artifact,tmp_path/'filter.pt')


def test_cli_preview_no_gpu_or_download(tmp_path):
    config = tmp_path/'config.json'
    config.write_text(json.dumps(dict(dataset='organization/dataset')))
    result = subprocess.run([sys.executable,'-m','slice_critic','preview','--config',str(config)],
                            capture_output=True,text=True,check=True)
    preview = json.loads(result.stdout)
    assert preview['blocks_one_based'] == [28,27,25,21,13]
    assert preview['critic_dimension'] == 769


def test_huggingface_revision_is_resolved_before_loading(monkeypatch):
    from types import SimpleNamespace
    import datasets
    import huggingface_hub
    from slice_critic.data import load_rows
    calls = []
    class Hub:
        def dataset_info(self, name, revision=None):
            calls.append((name,revision))
            return SimpleNamespace(sha='a'*40)
    def load(name, subset, **kwargs):
        calls.append((name,subset,kwargs))
        return [{'prompt':'Question','number':'1'}]
    monkeypatch.setattr(huggingface_hub,'HfApi',Hub)
    monkeypatch.setattr(datasets,'load_dataset',load)
    config = TrainingConfig(dataset='organization/example',dataset_revision='main',dataset_config='numeric',
                            question_column='prompt',reference_column='number')
    rows, identity = load_rows(config)
    assert identity['revision'] == 'a'*40
    assert calls[1][2] == dict(split='train',revision='a'*40)
    assert rows == [{'prompt':'Question','number':'1'}]
    monkeypatch.setattr(Hub,'dataset_info',lambda *a,**kw: SimpleNamespace(sha=None))
    with pytest.raises(ValueError,match='immutable commit'): load_rows(config)
    with pytest.raises(ValueError,match='Local model_revision'):
        TrainingConfig(dataset='example',model_path='model',model_revision='main').validate()


def test_package_has_no_project_imports():
    import ast
    for path in (Path(__file__).parents[1]/'src/slice_critic').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            assert all(not name or not name.startswith(('math_rl','verl','jlens')) for name in names), path
