import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import math_comparison as comparison
from math_rl.provenance import sha256


def questions():
    return [dict(id=f'math/{split}/{i}', question=f'{split} question {i}', ground_truth='2',
                 split=split, official_split='test' if split == 'test' else 'train',
                 level=4 + i % 2, source_index=offset + i)
            for split, size, offset in [('train', 16, 0), ('val', 2, 16), ('test', 2, 18)]
            for i in range(size)]


def test_split_validation_rejects_leakage():
    rows = questions()
    assert comparison.validate_questions(rows) == dict(train=16, val=2, test=2)
    rows[-1]['question'] = rows[0]['question']
    with pytest.raises(ValueError, match='Duplicate'):
        comparison.validate_questions(rows)
    rows = questions()
    rows[0]['official_split'] = 'test'
    with pytest.raises(ValueError, match='Official test'):
        comparison.validate_questions(rows)


def test_data_reuses_saved_splits_without_old_labels(tmp_path, monkeypatch):
    from datasets import Dataset
    from transformers import AutoTokenizer
    root = tmp_path / 'repo'
    source = tmp_path / 'offline'
    (source / 'data').mkdir(parents=True)
    (root / 'models/qwen-math').mkdir(parents=True)
    weights = root / 'models/qwen-math/model.safetensors'
    weights.write_bytes(b'test-weights')
    path = source / 'data/questions.json'
    path.write_text(json.dumps(dict(questions=questions(), reserved_math500_questions=500)))
    manifest = dict(config=dict(protocol='frozen-math-critics-v1'), questions_sha256=sha256(path),
                    files={str(weights): sha256(weights)})
    (source / 'data/manifest.json').write_text(json.dumps(manifest))
    (source / 'report.txt').write_text('completed')
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            return [1, 2]
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **kw: Tokenizer())
    monkeypatch.setattr(comparison, 'ROOT', root)
    out = tmp_path / 'new'
    out.mkdir()
    result = comparison.prepare(source, out)
    assert result['counts'] == dict(train=16, val=2, test=2)
    test = Dataset.from_parquet(str(out / 'data/test.parquet'))
    assert [r['extra_info']['prompt_id'] for r in test] == ['math/test/0', 'math/test/1']
    assert test[0]['data_source'] == 'math_numeric'
    assert test[0]['reward_model']['ground_truth'] == '2'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='snapshot'):
        comparison.prepare(source, out)


def test_launch_commands_match_budgets_and_use_requested_lambda(tmp_path):
    import yaml
    config = yaml.safe_load((ROOT / 'configs/math_online.yaml').read_text())
    assert config['experiment']['critic_lambda'] == .99
    assert config['data']['train_batch_size'] * config['actor_rollout_ref']['rollout']['n'] == 64
    for method in comparison.METHODS:
        args = comparison.command(tmp_path, method, 42, 30)
        assert 'trainer.total_training_steps=30' in args and 'data.seed=42' in args
        assert f'algorithm.adv_estimator={"grpo" if method == "grpo" else "gae"}' in args
        assert f'experiment.method={method}' in args


def test_report_marks_missing_methods_and_checks_test_alignment(tmp_path):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/questions.json').write_text(json.dumps(dict(questions=questions())))
    manifest = dict(seeds=[42], methods=['ppo', 'ridge'], steps=2, data=dict(scope='exploratory'))
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    case = tmp_path / 'ppo-seed42'
    case.mkdir()
    rows = [dict(id=f'math/test/{i}', ground_truth='2', level=4 + i,
                 score=i, verifier_status='correct' if i else 'incorrect', capped=False) for i in range(2)]
    for label in ('base', 'final'):
        (case / f'{label}-test.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    result = dict(updates=2, answers=128, initial_test=dict(accuracy=.5), final_test=dict(accuracy=.5),
                  training_seconds=100, training_gpu_hours=200/3600)
    (case / 'result.json').write_text(json.dumps(result))
    (tmp_path / 'ppo-seed42.log').write_text('perf/max_memory_allocated_gb:20.0')
    report = comparison.report(tmp_path)
    assert report['missing'] == ['ridge-seed42'] and not report['complete']
    assert report['runs']['ppo-seed42']['versus_base']['net_correct'] == 0
    rows[0]['id'] = 'wrong-question'
    (case / 'final-test.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match='Different test'):
        comparison.report(tmp_path)


def test_pinned_config_and_worker_adapter_import():
    pytest.importorskip('verl')
    from hydra import compose, initialize_config_dir
    from math_rl.math_workers import FeatureActorWorker
    from math_rl.math_trainer import MathTrainer
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    assert issubclass(FeatureActorWorker, ActorRolloutRefWorker)
    assert issubclass(MathTrainer, RayPPOTrainer)
    with initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        config = compose(config_name='math_online')
    assert config.algorithm.lam == .95
    assert config.actor_rollout_ref.rollout.n == 4
    assert config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 1
    assert not config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
