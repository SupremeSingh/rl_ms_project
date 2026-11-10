"""Matched six-arm training and independent Monte Carlo value diagnostics.

GPU stages live in value_audit_gpu; importing this module creates no GPU context.
Nothing is fitted on the prefix bank or the Monte Carlo reference outcomes.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from . import runner
from .provenance import sha256, snapshot, write_json
from slice_critic.storage import model_identity

ROOT = runner.ROOT
ARMS = {
    'slice-token': dict(method='ridge', layer='bottleneck', mode='token', normalization='none'),
    'rms-layer-token': dict(method='ridge', layer='two_thirds', mode='token', normalization='rms'),
    'slice-chunk': dict(method='ridge', layer='bottleneck', mode='math', normalization='none'),
    'rms-layer-chunk': dict(method='ridge', layer='two_thirds', mode='math', normalization='rms'),
    'ppo': dict(method='ppo', layer='final', mode='token', normalization='none'),
    'grpo': dict(method='grpo', layer='final', mode='token', normalization='none'),
}
TITLES = ['SLICE + Ridge (token)', 'RMS layer + Ridge (token)',
          'SLICE + Ridge (chunk)', 'RMS layer + Ridge (chunk)',
          'PPO value network', 'GRPO question baseline (no critic)']
BUDGETS = {'smoke': dict(questions=2, prefixes=2, samples=4, shards=1),
           'pilot': dict(questions=10, prefixes=4, samples=64, shards=1),
           'full': dict(questions=50, prefixes=4, samples=256, shards=4)}


def read(path):
    return json.loads(Path(path).read_text())


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], 'big') % (2**31)


def require_gate(out, name):
    record = read(out / name)
    if not record.get('passed') or record['manifest_sha256'] != sha256(out / 'manifest.json'):
        raise ValueError(f'Matching passed {name} required')
    folder = (out / name).parent
    if 'bank_sha256' in record and record['bank_sha256'] != sha256(folder / 'bank.json'):
        raise ValueError('Passed reference bank changed')
    for filename, digest in record.get('files', {}).items():
        if Path(filename).name != filename or sha256(folder / filename) != digest:
            raise ValueError('Passed audit artifact changed')


def prepare(source, out, updates=120, actor_seed=1001):
    if type(updates) is not int or updates < 2 or type(actor_seed) is not int or actor_seed < 0:
        raise ValueError('Require at least two training updates and a nonnegative actor seed')
    # Fail in the CPU preparation job, before reserving six training allocations.
    from importlib.metadata import distribution
    import matplotlib
    from transformers import AutoModelForTokenClassification  # noqa: F401
    direct = json.loads(distribution('verl').read_text('direct_url.json') or '{}')
    if direct.get('vcs_info', {}).get('commit_id') != '8d9e350ea58c7ad4b50dd14d9dcb50577242c55f':
        raise ValueError('Install the project-pinned VERL before preparing this audit')
    matplotlib.use('Agg')
    # Reuse the existing data/model/filter identity and family-exclusion audit.
    from .chunk_study import prepare as prepare_data
    prepare_data(source, out, fallback_tokens=32)
    m = read(out / 'manifest.json')
    m.update(protocol='slice-value-audit-v1', arms=ARMS, steps=updates, actor_seed=actor_seed,
             ppo_profile='higher_lr', budgets=BUDGETS, reference_seed=48371, grpo_group_size=4,
             reference_batch=16, response_cap=2048, max_model_len=2560,
             sampling=dict(temperature=1., top_p=1., top_k=-1),
             scope='One predeclared actor seed, matched fresh training. Reused inspected Numina test questions. '
                   'Shared mixture-prefix bank; each frozen actor defines its own MC target. '
                   'No test/MC tuning, no reference-based refitting, no multi-seed ranking claim.')
    m.pop('seeds', None)
    m.pop('primary', None)
    m['runtime'] = snapshot(ROOT)['files']
    write_json(out / 'manifest.json', m)


def validate(out):
    m = read(out / 'manifest.json')
    if m.get('protocol') != 'slice-value-audit-v1' or m['arms'] != ARMS or m['budgets'] != BUDGETS:
        raise ValueError('Unsupported/modified value audit protocol')
    expected = dict(gamma=1, gae_lambda=.95, epsilon=.01, critic_folds=1,
                    answers_per_update=64, chunk_fallback_tokens=32, response_cap=2048,
                    max_model_len=2560, grpo_group_size=4, reference_batch=16,
                    ppo_profile='higher_lr', reference_seed=48371,
                    sampling=dict(temperature=1., top_p=1., top_k=-1))
    if any(m.get(k) != v for k, v in expected.items()):
        raise ValueError('Locked training/reference recipe changed')
    if m['runtime'] != snapshot(ROOT)['files']:
        raise ValueError('Runtime changed since prepare; use a new study')
    for name, digest in m['data_hashes'].items():
        if sha256(out / name) != digest:
            raise ValueError('Study data changed')
    if sha256(out / 'filters/sparse.pt') != m['filter_sha256'] or model_identity(ROOT / 'models/qwen-math') != m['model_hashes']:
        raise ValueError('Frozen feature basis changed')
    return m


def command(out, m, arm, smoke=False):
    s = ARMS[arm]
    seed = 19 if smoke else m['actor_seed']
    steps = 2 if smoke else m['steps']
    case = ('smoke-' if smoke else '') + f'{arm}-seed{seed}'
    cmd = runner.command(out, s['method'], seed, steps, feature_mode='frozen',
                         ppo_profile=m['ppo_profile'] if s['method'] == 'ppo' else 'not_applicable',
                         evaluation_split='val', case_name=case)
    cmd += [f'experiment.chunking={s["mode"]}', 'experiment.chunk_fallback_tokens=32',
            'experiment.critic_folds=1', 'experiment.critic_statistics=cumulative',
            'experiment.cumulative_epsilon=0.01', 'algorithm.lam=0.95',
            f'actor_rollout_ref.model.critic_feature_layer={s["layer"]}',
            f'actor_rollout_ref.model.critic_feature_normalization={s["normalization"]}',
            'actor_rollout_ref.model.critic_encoder_precision=fp32',
            f'trainer.test_freq={steps}',
            'actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model]']
    if s['method'] == 'ppo':
        cmd += ['critic.checkpoint.save_contents=[model,hf_model]']
    if s['layer'] == 'bottleneck':
        cmd += [f'actor_rollout_ref.model.critic_bottleneck_path={out}/filters/sparse.pt',
                f'actor_rollout_ref.model.critic_bottleneck_sha256={m["filter_sha256"]}',
                f'actor_rollout_ref.model.critic_bottleneck_layers={json.dumps(m["layers"], separators=(",", ":"))}',
                f'actor_rollout_ref.model.critic_bottleneck_rms_eps={m["rms_eps"]}']
    if smoke:
        cmd += [f'data.train_files={out}/smoke-data/train.parquet',
                f'data.val_files={out}/smoke-data/val.parquet',
                f'experiment.test_file={out}/smoke-data/val.parquet']
    return cmd, out / case


def checkpoint_paths(out, m, arm, phase):
    _, case = command(out, m, arm, phase == 'smoke')
    steps = 2 if phase == 'smoke' else m['steps']
    checkpoint = case / f'global_step_{steps}'
    return case, checkpoint / 'actor/huggingface', checkpoint / 'critic/huggingface', case / f'linear-critic/{steps}.pt'


def audit_run(out, m, arm, smoke=False):
    case, actor, critic, head = checkpoint_paths(out, m, arm, 'smoke' if smoke else 'pilot')
    steps = 2 if smoke else m['steps']
    result = read(case / 'result.json')
    cfg = read(case / 'run.json')['config']
    s = ARMS[arm]
    model = cfg['actor_rollout_ref']['model']
    if (result['updates'] != steps or result['answers'] != steps * 64
            or result['seed'] != (19 if smoke else m['actor_seed'])
            or result['method'] != s['method'] or result['chunking'] != s['mode']
            or cfg['experiment']['critic_folds'] != 1 or cfg['algorithm']['lam'] != .95
            or cfg['experiment']['cumulative_epsilon'] != .01
            or model['critic_feature_layer'] != s['layer']
            or model.get('critic_feature_normalization', 'none') != s['normalization']
            or cfg['data']['max_prompt_length'] != 512 or cfg['data']['max_response_length'] != 2048
            or cfg['experiment']['evaluation_split'] != 'val'
            or model['critic_encoder_precision'] != 'fp32'):
        raise ValueError(f'Run recipe mismatch: {arm}')
    identity = dict(actor=model_identity(actor))  # config/tokenizer alone is not a model
    if s['method'] == 'ridge':
        saved = torch.load(head, map_location='cpu', weights_only=True)
        stats = torch.load(case / 'linear-critic/statistics.pt', map_location='cpu', weights_only=True)
        if (saved['step'] != steps or len(saved['heads']) != 1 or stats['step'] != steps
                or stats['folds'] != 1 or saved['method'] != 'ridge' or stats['method'] != 'ridge'
                or stats['stats'][0]['answers'] != steps * 64):
            raise ValueError('Wrong cumulative head')
        w = saved['heads'][0]['weights'].double()
        k = 769 if s['layer'] == 'bottleneck' else 1537
        a, b = stats['stats'][0]['a'].double(), stats['stats'][0]['b'].double()
        if w.shape != (k,) or not torch.isfinite(w).all() or stats['epsilon'] != .01:
            raise ValueError('Invalid head weights')
        residual = (a @ w + .01 * w - b).norm() / (a.norm()*w.norm()+b.norm()).clamp_min(1e-30)
        if not torch.isfinite(residual) or residual > 1e-8:
            raise ValueError('Ridge head does not solve its saved system')
        identity['head'] = sha256(head)
        identity['statistics'] = sha256(case / 'linear-critic/statistics.pt')
    elif s['method'] == 'ppo':
        identity['critic'] = model_identity(critic)
        from .ppo_baseline import audit_health
        health = audit_health(case / 'ppo-critic-audit', steps)
        if not all(health['gates'][k] for k in ('complete', 'finite', 'zero_initial_values', 'nonzero_updates')):
            raise ValueError('PPO critic failed functional checks')
        identity['health'] = health
    return dict(identity=identity, training_gpu_hours=result['training_gpu_hours'],
                actor_update=steps, linear_head_rollout_policy_update=steps-1 if s['method']=='ridge' else None,
                critic_last_rollout_policy_update=steps-1 if s['method']!='grpo' else None,
                scope='Ridge head fitted before final actor update; one-update lag is retained and reported.')


def train(out, m, arm, smoke=False):
    cmd, case = command(out, m, arm, smoke)
    if (case / 'result.json').exists():
        return audit_run(out, m, arm, smoke)
    if case.exists():
        raise ValueError(f'Partial training preserved: {case}; inspect before a fresh study')
    env = dict(os.environ, RAY_ADDRESS='local', MATH_RL_SEED=str(19 if smoke else m['actor_seed']))
    env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', None)
    with tempfile.TemporaryDirectory(prefix='value-ray-', dir='/tmp') as temp:
        env['RAY_TMPDIR'] = temp
        with (out / f'{case.name}.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return audit_run(out, m, arm, smoke)


def train_gate(out, m, phase):
    runs = {arm: audit_run(out, m, arm, phase == 'smoke') for arm in ARMS}
    write_json(out / f'{phase}-training.json', dict(passed=True, runs=runs,
               manifest_sha256=sha256(out / 'manifest.json')))


def check_models(out, m, arm, phase):
    saved = read(out / f'{"smoke" if phase=="smoke" else "train"}-training.json')['runs'][arm]
    actual = audit_run(out, m, arm, phase == 'smoke')
    if saved != actual:
        raise ValueError('Checkpoint or run identity changed after training gate')


def audit_questions(out, m, phase):
    questions = [q for q in read(out / 'data/questions.json')['questions'] if q['split'] == 'test']
    # Outcome-independent order and fixed balanced mixture; no cherry-picked trajectories.
    questions.sort(key=lambda q: stable_seed(m.get('question_seed',m['reference_seed']), 'questions', q['id']))
    budget = m['budgets'][phase]['questions']
    if len(questions) < budget:
        raise ValueError('Insufficient held-out audit questions')
    return [dict(q, bank_arm=list(ARMS)[i % len(ARMS)]) for i, q in enumerate(questions[:budget])]


def boundary_positions(starts, count):
    if not starts or starts[0] != 0 or sorted(set(starts)) != list(starts):
        raise ValueError('Invalid chunk starts')
    return [starts[i] for i in sorted(set(np.linspace(0, len(starts)-1, min(count,len(starts))).round().astype(int).tolist()))]


def pack_bank(out, m, phase):
    folder = out / phase
    planned = audit_questions(out, m, phase)
    records, tokenizer_hashes = {}, set()
    for arm in ARMS:
        data = read(folder / f'bank-{arm}.json')
        if data['manifest_sha256'] != sha256(out / 'manifest.json'):
            raise ValueError('Wrong bank manifest')
        tokenizer_hashes.add(data['tokenizer_sha256'])
        for row in data['questions']:
            if row['id'] in records or row['bank_arm'] != arm:
                raise ValueError('Repeated/misassigned bank question')
            records[row['id']] = row
    if len(tokenizer_hashes) != 1 or set(records) != {q['id'] for q in planned}:
        raise ValueError('Incompatible tokenizer or incomplete question bank')
    states = []
    for q in planned:
        row = records[q['id']]
        if row['ground_truth'] != q['ground_truth'] or not row['prompt_ids'] or not row['answer_ids']:
            raise ValueError('Invalid bank question')
        positions = row['positions']
        if (not positions or positions[0] != 0 or sorted(set(positions)) != positions
                or len(positions) > m['budgets'][phase]['prefixes']):
            raise ValueError('Invalid/duplicated boundary positions')
        for index, position in enumerate(row['positions']):
            if not 0 <= position < len(row['answer_ids']) or position >= m['response_cap']:
                raise ValueError('Terminal/out-of-budget prefix')
            states.append(dict(state_id=f'{q["id"]}/prefix-{index}', question_id=q['id'],
                               ground_truth=q['ground_truth'], bank_arm=q['bank_arm'],
                               prompt_ids=row['prompt_ids'], prefix_ids=row['answer_ids'][:position],
                               response_position=position, progress=position/max(1,len(row['answer_ids'])),
                               remaining_tokens=m['response_cap']-position))
    if len({r['state_id'] for r in states}) != len(states):
        raise ValueError('Duplicate bank states')
    write_json(folder / 'bank.json', dict(manifest_sha256=sha256(out/'manifest.json'),
               tokenizer_sha256=tokenizer_hashes.pop(), states=states, questions=planned))


def wilson(successes, count):
    p, z = successes/count, 1.959963984540054
    center = (p+z*z/(2*count))/(1+z*z/count)
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [max(0.,center-half),min(1.,center+half)]


def summarize(rows, seed=7231, draws=2000):
    if not rows:
        raise ValueError('Cannot summarize an empty prefix bank')
    groups = {}
    for r in rows:
        value, k, hits = r['prediction'], r['samples'], r['successes']
        if not math.isfinite(value) or k < 2 or type(hits) is not int or not 0 <= hits <= k:
            raise ValueError('Invalid reference/prediction')
        reference = hits/k
        groups.setdefault(r['question_id'], []).append([abs(value-reference), (value-reference)**2,
            float(not 0<=value<=1), (value-reference)**2-reference*(1-reference)/(k-1)])
    x = np.array([np.mean(v,axis=0) for v in groups.values()])
    rng = np.random.default_rng(seed)
    bootstrap = np.array([x[rng.integers(len(x),size=len(x))].mean(0) for _ in range(draws)])
    # A separate uncertainty sensitivity calculation integrates binomial-reference
    # uncertainty with Jeffreys Beta posteriors. This is not a frequentist CI and
    # must not be confused with the conditional question-bootstrap interval.
    question_ids = list(groups)
    grouped_rows = [[r for r in rows if r['question_id'] == q] for q in question_ids]
    sensitivity = []
    for _ in range(draws):
        chosen = rng.integers(len(grouped_rows), size=len(grouped_rows))
        # Draw each original state once, including when its question is resampled.
        per_question = []
        for group in grouped_rows:
            p = rng.beta([r['successes']+.5 for r in group],
                         [r['samples']-r['successes']+.5 for r in group])
            errors = np.array([r['prediction'] for r in group]) - p
            per_question.append([np.abs(errors).mean(), np.square(errors).mean()])
        sensitivity.append(np.array(per_question)[chosen].mean(0))
    sensitivity = np.array(sensitivity)
    avg = x.mean(0)
    return dict(questions=len(x), states=len(rows), mae=float(avg[0]), rmse=float(np.sqrt(avg[1])),
                outside_unit_interval=float(avg[2]), mc_noise_corrected_mse=float(avg[3]),
                question_bootstrap_mae_95=np.quantile(bootstrap[:,0],[.025,.975]).tolist(),
                question_bootstrap_rmse_95=np.sqrt(np.quantile(bootstrap[:,1],[.025,.975])).tolist(),
                question_mc_sensitivity_mae_95=np.quantile(sensitivity[:,0],[.025,.975]).tolist(),
                question_mc_sensitivity_rmse_95=np.sqrt(np.quantile(sensitivity[:,1],[.025,.975])).tolist(),
                reference_worst_case_se=max(.5/math.sqrt(r['samples']) for r in rows),
                scope='Equal-question metrics; intervals resample questions, conditional on sampled MC references. '
                      'Additional MC sensitivity uses Jeffreys Beta reference draws, not a frequentist CI. '
                      'Both exclude actor-seed uncertainty. Raw MSE includes MC noise; corrected MSE may be negative.')


def report(out, m, phase):
    folder = out / phase
    bank = read(folder/'bank.json')
    panels, metrics, timings = {}, {}, {}
    for arm in ARMS:
        prediction = read(folder / f'predictions-{arm}.json')
        references = [read(folder / f'reference-{arm}-{i}.json') for i in range(m['budgets'][phase]['shards'])]
        for data in (prediction, *references):
            if data['bank_sha256'] != sha256(folder/'bank.json') or data['manifest_sha256'] != sha256(out/'manifest.json'):
                raise ValueError('Stale audit outputs')
        wanted = {r['state_id'] for r in bank['states']}
        p = {r['state_id']:r['prediction'] for r in prediction['states']}
        reference_rows = [row for reference in references for row in reference['states']]
        r = {r['state_id']:r for r in reference_rows}
        if len(p)!=len(prediction['states']) or len(r)!=len(reference_rows) or set(p)!=wanted or set(r)!=wanted:
            raise ValueError('Incomplete/duplicated panel states')
        rows=[]
        for state in bank['states']:
            item=r[state['state_id']]
            if item['samples']!=m['budgets'][phase]['samples']:
                raise ValueError('Unequal Monte Carlo sample budgets')
            rows.append(dict(state,prediction=p[state['state_id']],samples=item['samples'],successes=item['successes'],
                             reference=item['successes']/item['samples'],reference_wilson_95=wilson(item['successes'],item['samples'])))
        panels[arm]=rows
        metrics[arm]=summarize(rows)
        statuses = Counter()
        for reference in references: statuses.update(reference['label_statuses'])
        timings[arm]=dict(prediction_seconds=prediction['seconds'],mc_seconds=sum(r['seconds'] for r in references),
                          response_tokens=sum(r['response_tokens'] for r in references),label_statuses=dict(statuses),
                          capped_continuations=sum(r['capped_continuations'] for r in references),
                          baseline_response_tokens=sum(r.get('baseline_response_tokens',0) for r in references),
                          cached_generation_and_verification_seconds=sum(r['cached_generation_and_verification_seconds'] for r in references))
    record=dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),phase=phase,metrics=metrics,
                panels=panels,timings=timings,scope=m['scope'],
                grpo='Four fresh prompt-level rewards, independent from reference samples; same baseline at all question prefixes.',
                target='Policy-specific finite-horizon verifier reward; no claim of exact ground truth or verifier infallibility.',
                head_lag='Cumulative Ridge heads precede the final actor by one update. '
                         'The final PPO critic targets also come from the last pre-update rollouts.',
                uncertainty='Wilson intervals describe per-state reference uncertainty; question bootstrap is conditional on references. '
                            'Separate question/MC sensitivity ranges use Jeffreys Beta draws, not frequentist CIs. No actor-seed uncertainty.')
    plot(folder,panels,metrics)
    write_json(folder/'summary.json',record)
    lines=['Monte Carlo value audit: one actor seed; equal-question metrics; no clipping.',
           'Shared prefixes, policy-specific continuation targets; GRPO has no value network.',
           'arm                     MAE       RMSE      outside [0,1]']
    lines += [f'{a:<23} {v["mae"]:.5f}   {v["rmse"]:.5f}   {100*v["outside_unit_interval"]:.2f}%' for a,v in metrics.items()]
    lines += [record['head_lag'],record['uncertainty'],m['scope'],
              m.get('report_note','Pilot/smoke are pipeline/cost diagnostics, not method rankings. Reference generation cost is additional.')]
    (folder/'report.txt').write_text('\n'.join(lines)+'\n')
    files = ['bank.json','summary.json','report.txt','value-scatter.png','value-scatter.pdf','error-by-progress.png']
    files += [f'predictions-{arm}.json' for arm in ARMS]
    files += [f'reference-{arm}-{i}.json' for arm in ARMS for i in range(m['budgets'][phase]['shards'])]
    write_json(folder/'verification.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),
               bank_sha256=sha256(folder/'bank.json'),files={name:sha256(folder/name) for name in files},
               scope='All six panels complete; does not establish method superiority.'))


def plot(folder, panels, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    predictions=[r['prediction'] for rows in panels.values() for r in rows]
    lo,hi=min(-.05,min(predictions)-.03),max(1.05,max(predictions)+.03)
    fig,axes=plt.subplots(2,3,figsize=(14,9),sharex=True,sharey=True,layout='constrained')
    colors = ['#6a3d8f','#2374ab','#9670ae','#64a9ce','#e69f00','#009e73']
    for ax,arm,title,color in zip(axes.flat,ARMS,TITLES,colors):
        rows=panels[arm]
        ax.plot([0,1],[0,1],color='black',ls='--',lw=1)
        ax.scatter([r['reference'] for r in rows],[r['prediction'] for r in rows],alpha=.4,s=24,color=color)
        ax.set(title=title,xlim=(-.05,1.05),ylim=(lo,hi))
        ax.text(.04,.94,f'MAE {metrics[arm]["mae"]:.3f}',transform=ax.transAxes,va='top')
        ax.grid(alpha=.25)
    fig.supxlabel('Monte Carlo reference value (each panel’s frozen actor)')
    fig.supylabel('Predicted value / GRPO baseline (unclipped)')
    fig.suptitle('Shared held-out prefixes • policy-specific targets • one actor seed')
    fig.savefig(folder/'value-scatter.png',dpi=180)
    fig.savefig(folder/'value-scatter.pdf')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for arm in ARMS:
        points=[]
        for left in np.arange(0,1,.2):
            chosen=[r for r in panels[arm] if left<=r['progress']<left+.2]
            if chosen:
                grouped = {}
                for row in chosen:
                    grouped.setdefault(row['question_id'], []).append(abs(row['prediction']-row['reference']))
                points.append((left+.1,float(np.mean([np.mean(v) for v in grouped.values()]))))
        if points: ax.plot(*zip(*points),marker='o',label=arm)
    ax.set(xlabel='Relative answer progress',ylabel='Equal-question MAE',title='Value error versus prefix progress')
    ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.savefig(folder/'error-by-progress.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['prepare','train','training-gate','bank','pack','evaluate','predict','mc','report','inventory'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase',choices=list(BUDGETS),default='smoke')
    parser.add_argument('--task',type=int,default=0)
    parser.add_argument('--updates',type=int,default=120)
    parser.add_argument('--actor-seed',type=int,default=1001)
    a=parser.parse_args();out=a.out.resolve();started=time.perf_counter()
    if a.stage=='prepare':
        if a.source is None:parser.error('prepare requires --source')
        prepare(a.source.resolve(),out,a.updates,a.actor_seed);return
    m=validate(out)
    arm=list(ARMS)[a.task % len(ARMS)] if a.task>=0 else None
    shard=a.task//len(ARMS)
    if a.stage=='mc':
        if not 0<=shard<m['budgets'][a.phase]['shards']:raise ValueError('Invalid MC shard task')
    elif not 0<=a.task<len(ARMS):raise ValueError('Invalid arm task')
    (out/a.phase).mkdir(exist_ok=True)
    status=out/a.phase/f'{a.stage}-{a.task}-status.json'
    try:
        if a.stage=='train':
            if a.phase!='smoke':require_gate(out,'smoke/verification.json')
            train(out,m,arm,a.phase=='smoke')
        elif a.stage in ('training-gate','inventory'):train_gate(out,m,'smoke' if a.phase=='smoke' else 'train')
        else:
            require_gate(out,f'{"smoke" if a.phase=="smoke" else "train"}-training.json')
            if a.phase=='full':require_gate(out,'pilot/verification.json')
            if a.stage=='pack':pack_bank(out,m,a.phase)
            elif a.stage=='report':report(out,m,a.phase)
            elif a.stage=='evaluate':
                for stage in ('predict','mc'):
                    subprocess.run([sys.executable,'-m','math_rl.value_audit','--stage',stage,'--out',str(out),
                                    '--phase',a.phase,'--task',str(a.task)],cwd=ROOT,check=True)
            else:
                if arm is None:raise ValueError('Invalid arm task')
                check_models(out,m,arm,a.phase)
                from . import value_audit_gpu as gpu
                if a.stage=='mc':gpu.monte_carlo(out,m,a.phase,arm,shard)
                else:{'bank':gpu.bank,'predict':gpu.predict}[a.stage](out,m,a.phase,arm)
        write_json(status,dict(state='complete',seconds=time.perf_counter()-started))
    except Exception as exc:
        write_json(status,dict(state='failed',error=str(exc)));raise


if __name__=='__main__':
    main()
