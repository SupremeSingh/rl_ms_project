"""Pinned VERL command construction and evaluation-file validation."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

def command(out, method, seed, steps, feature_mode='actor', ppo_profile='not_applicable',
            evaluation_split='test', case_name=None):
    from math_rl.ppo_baseline import PPO_PROFILES
    case = out / (case_name or f'{method}-seed{seed}')
    extras = []
    if method == 'ppo' and ppo_profile != 'not_applicable':
        profile = PPO_PROFILES[ppo_profile]
        extras = [f'experiment.ppo_profile={ppo_profile}',
            f'critic.optim.lr={profile["lr"]}', f'critic.ppo_epochs={profile["epochs"]}',
            'critic.loss_agg_mode=seq-mean-token-mean', 'critic.use_dynamic_bsz=false']
    return [sys.executable, '-m', 'math_rl.main',
        f'experiment.method={method}',
        f'actor_rollout_ref.model.critic_feature_mode={feature_mode if method in ("lstd", "ridge") else "actor"}',
        'experiment.replay_capacity=0',
        f'algorithm.adv_estimator={"grpo" if method == "grpo" else "gae"}',
        f'data.train_files={out / "data/train.parquet"}', f'data.val_files={out / "data/val.parquet"}',
        f'experiment.test_file={out / ("data/" + evaluation_split + ".parquet")}',
        f'experiment.evaluation_split={evaluation_split}', f'data.seed={seed}',
        f'trainer.total_training_steps={steps}', f'trainer.save_freq={steps}',
        f'trainer.default_local_dir={case}', f'trainer.experiment_name=math-{method}-seed{seed}',
        f'hydra.run.dir={out / (case.name + "-hydra")}'] + extras


def read_answers(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate test answers')
    for r in rows:
        if r['score'] not in (0, 1) or r['score'] != int(r['verifier_status'] == 'correct'):
            raise ValueError('Invalid evaluation score/status')
    return rows
