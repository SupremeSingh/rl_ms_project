import json
import subprocess
import sys

import pytest

from slice_critic import TrainingConfig, train_critic
from slice_critic.training import VerificationClient


def test_verifier_preserves_virtualenv_symlink(tmp_path, monkeypatch):
    from slice_critic.grading import PINS
    executable = tmp_path / 'venv' / 'bin' / 'python'
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert json.loads(kwargs['input']) == []
        return subprocess.CompletedProcess(argv, 0, json.dumps({
            'provenance': {'packages': PINS}, 'results': []}), '')

    monkeypatch.setattr(subprocess, 'run', run)
    client = VerificationClient(executable)
    client.check()
    assert client.python == str(executable)
    assert calls == [[str(executable), '-m', 'slice_critic.verifier']]


def test_verifier_reports_child_stderr(monkeypatch):
    def run(argv, **kwargs):
        raise subprocess.CalledProcessError(1, argv, stderr='No module named slice_critic')

    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='No module named slice_critic') as exc:
        VerificationClient('/example/venv/bin/python').check()
    assert '/example/venv/bin/python' in str(exc.value)
    assert 'exit 1' in str(exc.value)


def test_verifier_preflight_precedes_model_loading(tmp_path, monkeypatch):
    import slice_critic.backend as backend

    def fail(self):
        raise RuntimeError('verifier preflight failed')

    def unexpected_model_load(config):
        pytest.fail('Model loading must wait for verifier preflight')

    monkeypatch.setattr(VerificationClient, 'check', fail)
    monkeypatch.setattr(backend, 'resolve_model', unexpected_model_load)
    with pytest.raises(RuntimeError, match='verifier preflight failed'):
        train_critic(TrainingConfig(dataset='unused.jsonl'), tmp_path / 'training')
    status = json.loads((tmp_path / 'training' / 'status.json').read_text())
    assert status['state'] == 'failed'
    assert status['error'] == 'verifier preflight failed'
