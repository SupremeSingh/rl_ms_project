"""Exercise a published SLICE feature basis inside real PPO actor updates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from math_rl.provenance import sha256, snapshot, write_json
from slice_critic.data import family_key
from slice_critic.legacy import export_filter
from slice_critic.inference import load_filter
from slice_critic.storage import read_artifact
import math_comparison


def select_questions(rows, manifest, train_count=64, val_count=16):
    """Outcome-independent selection; never reuse representation fitting families."""
    blocked = set(manifest['representation_training_families']) | set(manifest['selection_families'])
    seen, selected = set(), []
    for split, count in [('train', train_count), ('val', val_count)]:
        candidates = []
        for q in rows:
            family = family_key(q['question'])
            if q['split'] != split or family in blocked or family in seen:
                continue
            seen.add(family)
            candidates.append(q)
            if len(candidates) == count:
                break
        if len(candidates) != count:
            raise ValueError(f'Need {count} eligible {split} questions outside compressor fitting/selection families')
        selected.extend(candidates)
    return selected


def command(out, method, exported):
    return math_comparison.command(out, method, 42, 2, feature_mode='frozen',
        evaluation_split='val') + [
        'experiment.critic_statistics=cumulative', 'experiment.cumulative_epsilon=0.01',
        'experiment.critic_lambda=0.99', 'actor_rollout_ref.model.critic_encoder_precision=fp32',
        'actor_rollout_ref.model.critic_feature_layer=bottleneck',
        f'actor_rollout_ref.model.critic_bottleneck_path={exported["path"]}',
        f'actor_rollout_ref.model.critic_bottleneck_sha256={exported["sha256"]}',
        f'actor_rollout_ref.model.critic_bottleneck_layers={json.dumps(exported["layers"], separators=(",", ":"))}',
        f'actor_rollout_ref.model.critic_bottleneck_rms_eps={exported["rms_eps"]}']


def verify_run(case, method):
    import torch
    result = json.loads((case / 'result.json').read_text())
    if result['updates'] != 2 or result['answers'] != 128 or result['evaluation_split'] != 'val':
        raise ValueError('Incomplete actor update/evaluation budget')
    stats = torch.load(case / 'linear-critic' / 'statistics.pt', map_location='cpu', weights_only=True)
    if (stats['step'] != 2 or stats['epoch'] != 0 or stats['method'] != method
            or stats['epsilon'] != .01 or stats['trace_lambda'] != .99):
        raise ValueError('Wrong cumulative fitting settings')
    if sum(s['answers'] for s in stats['stats']) != 128:
        raise ValueError('Statistics did not retain both updates')
    for s in stats['stats']:
        if (s['a'].shape != (769, 769) or s['b'].shape != (769,) or not s['answers']
                or not torch.isfinite(s['a']).all() or not torch.isfinite(s['b']).all()):
            raise ValueError('Invalid cross-fitted matrices')
    metrics = [json.loads(row) for row in (case / 'linear-critic' / 'metrics.jsonl').read_text().splitlines()]
    if len(metrics) != 2 or metrics[1]['reset'] or any(m['feature_epoch'] != 0 for m in metrics):
        raise ValueError('Feature basis or matrices unexpectedly reset')
    for step in (1, 2):
        record = torch.load(case / 'linear-critic' / f'{step}.pt', map_location='cpu', weights_only=True)
        if len(record['heads']) != 2 or any(h['weights'].shape != (769,)
                or not torch.isfinite(h['weights']).all() for h in record['heads']):
            raise ValueError('Invalid fresh cross-fitted heads')
    return dict(passed=True, updates=2, answers=128, matrix_dimension=769,
                folds=2, statistics_answers=[s['answers'] for s in stats['stats']],
                training_gpu_hours=result['training_gpu_hours'])


def run(args):
    from datasets import Dataset
    from transformers import AutoTokenizer
    from math_rl.prompts import make_prompt, encode_completion, AUDIT_INSTRUCTION
    artifact = args.artifact.resolve()
    manifest, _, _ = read_artifact(artifact)
    if manifest['prompt_instruction'] != AUDIT_INSTRUCTION:
        raise ValueError('Package and PPO prompt contracts differ')
    source = args.source.resolve()
    raw = json.loads(source.read_text())
    questions = select_questions(raw['questions'] if isinstance(raw, dict) else raw, manifest)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'status.json', dict(state='preparing'))
    try:
        exported = export_filter(artifact, out / 'ppo-filter.pt')
        # Includes encoder file identity, complete feature metadata and checksum checks.
        load_filter(exported['path'], exported['sha256'], layers=exported['layers'],
                    eps=exported['rms_eps'], model_path=ROOT / 'models/qwen-math')
        tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/qwen-math', local_files_only=True)
        (out / 'data').mkdir()
        for split in ('train', 'val'):
            records = []
            for i, q in enumerate(questions):
                if q['split'] != split:
                    continue
                if not 0 < len(encode_completion(tokenizer, q['question'])[1]) <= 512:
                    raise ValueError('Selected question exceeds the PPO prompt budget')
                records.append(dict(data_source='numina_numeric', prompt=make_prompt(q['question']), ability='math',
                    reward_model=dict(style='rule', ground_truth=q['ground_truth']),
                    extra_info=dict(index=i, prompt_id=q['id'], split=split, level=q.get('level', 0))))
            Dataset.from_list(records).to_parquet(str(out / 'data' / f'{split}.parquet'))
        write_json(out / 'manifest.json', dict(artifact=str(artifact),
            artifact_sha256=sha256(artifact / 'manifest.json'), source=str(source),
            source_sha256=sha256(source), selected_questions=questions, feature_export=exported,
            data_hashes={p.name: sha256(p) for p in (out / 'data').glob('*.parquet')},
            provenance=snapshot(ROOT), offline_value_head_loaded=False,
            scope='Package-to-PPO integration only; validation questions, no final test evaluation'))
        checks = {}
        for method in ('ridge', 'lstd'):
            write_json(out / 'status.json', dict(state='running', method=method))
            env = dict(os.environ, RAY_ADDRESS='local', MATH_RL_SEED='42')
            env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', None)
            with tempfile.TemporaryDirectory(prefix='sp-', dir='/tmp') as ray_tmp:
                env['RAY_TMPDIR'] = ray_tmp
                with (out / f'{method}.log').open('w') as log:
                    subprocess.run(command(out, method, exported), cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            checks[method] = verify_run(out / f'{method}-seed42', method)
        write_json(out / 'verification.json', dict(passed=True, methods=checks,
            offline_value_head_loaded=False, frozen_encoder=True, frozen_compressor=True,
            buffer_capacity=0, gamma=1, lstd_lambda=.99, gae_lambda=.95,
            scope='Two real PPO updates per method; integration check, not an accuracy ranking'))
        write_json(out / 'status.json', dict(state='complete'))
    except BaseException as exc:
        write_json(out / 'status.json', dict(state='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('artifact', type=Path)
    p.add_argument('--source', type=Path, default=Path('outputs/numina-online-768-large-v1/data/questions.json'))
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
