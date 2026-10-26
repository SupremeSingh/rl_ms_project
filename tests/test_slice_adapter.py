"""Existing PPO imports resolve to the standalone package implementation."""
from slice_critic import CumulativeCritic


def test_package_smoke_preserves_verifier_executable(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import sys
    import pytest

    script = Path(__file__).resolve().parents[1] / 'scripts' / 'smoke_slice_package.py'
    spec = importlib.util.spec_from_file_location('slice_smoke', script)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    executable = tmp_path / 'venv-python'
    executable.symlink_to(sys.executable)
    source = tmp_path / 'source.json'
    source.write_text(json.dumps({'questions': [dict(split='train', question=str(i),
                                                   ground_truth=str(i)) for i in range(40)]}))
    configs = []

    def stop_after_config(config, out):
        configs.append(config)
        raise RuntimeError('stop before GPU')

    monkeypatch.setattr(smoke, 'train_critic', stop_after_config)
    monkeypatch.setattr(sys, 'argv', ['smoke', '--source', str(source), '--out',
                                    str(tmp_path / 'out'), '--verifier-python', str(executable)])
    with pytest.raises(RuntimeError, match='stop before GPU'):
        smoke.main()
    assert configs[0].verifier_python == str(executable)
