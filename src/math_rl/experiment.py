"""Final six-method Numina protocol: validation calibration, fresh actors, paired report."""
from collections import Counter
from pathlib import Path
import json
import os
import subprocess
import tempfile
import time
import numpy as np
import torch
from . import runner
from .provenance import sha256, snapshot, write_json
from .ppo_baseline import PPO_PROFILES, audit_health, select_profile
ROOT = runner.ROOT

METHODS = {
    'ppo': dict(method='ppo', representation='transformer'),
    'grpo': dict(method='grpo', representation='none'),
    'ridge-two_thirds': dict(method='ridge', representation='two_thirds'),
    'lstd-two_thirds': dict(method='lstd', representation='two_thirds'),
    'ridge-sparse': dict(method='ridge', representation='sparse'),
    'lstd-sparse': dict(method='lstd', representation='sparse'),
}
PRIMARY = (
    'ridge-sparse_minus_ridge-two_thirds', 'lstd-sparse_minus_lstd-two_thirds',
    'lstd-sparse_minus_ridge-sparse', 'lstd-two_thirds_minus_ridge-two_thirds',
    'ridge-sparse_minus_ppo', 'lstd-sparse_minus_ppo',
    'ridge-sparse_minus_grpo', 'lstd-sparse_minus_grpo',
)
FOLD_METHODS = {
    f'{method}-fold{folds}': dict(method=method, representation='sparse', critic_folds=folds)
    for method in ('ridge', 'lstd') for folds in (2, 1)
}
FOLD_PRIMARY = ('ridge-fold1_minus_ridge-fold2', 'lstd-fold1_minus_lstd-fold2')

def read(path):
    return json.loads(Path(path).read_text())

def validate(out):
    out = Path(out).resolve()
    m=read(out/'manifest.json')
    if m.get('protocol') == 'slice-numina-folds-v1':
        if (m['conditions'] != FOLD_METHODS or m['primary_comparisons'] != list(FOLD_PRIMARY)
                or m['steps'] != 120 or m['critic_lambda'] != .9 or m['epsilon'] != .01
                or m['seeds'] != [301,302,303,304,305]):
            raise ValueError('Fold study settings changed from the locked paired protocol')
    elif m.get('protocol') != 'slice-numina-online-v1' or m['conditions'] != METHODS:
        raise ValueError('Expected the final six-method or paired fold protocol')
    if m['provenance']['files'] != snapshot(ROOT)['files']:
        raise ValueError('Code/config changed since preparation; use a new study')
    for name,checksum in m['data_hashes'].items():
        if sha256(out/'data'/name)!=checksum: raise ValueError('Data changed')
    for f in m['filters'].values():
        f['path'] = str(out / 'filters' / Path(f['path']).name)
        if sha256(f['path'])!=f['sha256']: raise ValueError('Filter changed')
    for name,checksum in m['model_hashes'].items():
        if sha256(ROOT/'models/qwen-math'/name)!=checksum: raise ValueError('Model changed')
    return m


def command(out,m,name,seed,steps,profile='more_fitting',calibration=False,case=None):
    setting=m['conditions'][name];method=setting['method'];rep=setting['representation']
    cmd=runner.command(out,method,seed,steps,feature_mode='frozen',
        ppo_profile=profile if method=='ppo' else 'not_applicable',
        evaluation_split='val' if calibration else 'test',case_name=case or f'{name}-seed{seed}')
    if method in ('ridge','lstd'):
        cmd += ['experiment.critic_statistics=cumulative','experiment.cumulative_epsilon=0.01',
                f'experiment.critic_folds={setting.get("critic_folds", 2)}',
                f'experiment.critic_lambda={m.get("critic_lambda", .99)}','actor_rollout_ref.model.critic_encoder_precision=fp32',
                f'actor_rollout_ref.model.critic_feature_layer={"bottleneck" if rep in ("plain","sparse") else rep}']
        if rep in ('plain','sparse'):
            f=m['filters'][rep]
            cmd += [f'actor_rollout_ref.model.critic_bottleneck_path={f["path"]}',
                    f'actor_rollout_ref.model.critic_bottleneck_sha256={f["sha256"]}',
                    f'actor_rollout_ref.model.critic_bottleneck_layers={json.dumps(m["layers"],separators=(",",":"))}',
                    f'actor_rollout_ref.model.critic_bottleneck_rms_eps={m["rms_eps"]}']
    return cmd


def launch(out,m,name,seed,steps,profile='more_fitting',calibration=False,case=None):
    case=case or f'{name}-seed{seed}'
    result=out/case/'result.json'
    if result.exists():
        r=read(result)
        if (r['updates']!=steps or r['seed']!=seed or r['evaluation_split']!=('val' if calibration else 'test')
                or (m['conditions'][name]['method']=='ppo' and r['ppo_profile'] != profile)):
            raise ValueError('Incompatible completed run')
        return r
    if (out/case).exists():
        raise ValueError(f'Partial run {case}: preserve/rename its directory and log before retrying; no partial actor resume')
    with tempfile.TemporaryDirectory(prefix='nr-', dir='/tmp') as ray_tmp:
        env=dict(os.environ, RAY_ADDRESS='local', MATH_RL_SEED=str(seed), RAY_TMPDIR=ray_tmp)
        env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', None)
        with (out/f'{case}.log').open('w') as log:
            subprocess.run(command(out,m,name,seed,steps,profile,calibration,case),cwd=ROOT,env=env,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    r=read(result)
    if r['updates']!=steps or r['answers']!=steps*64: raise ValueError('Incomplete training')
    return r


def smoke(out,m):
    # Only validation questions, never final test scores, are read by smoke/calibration.
    selection = out/'selection.json'
    profile = read(selection)['profile'] if selection.exists() else 'more_fitting'
    if profile not in PPO_PROFILES:
        raise ValueError('Unknown smoke PPO profile')
    for name in m['conditions']:
        launch(out,m,name,19,2,profile=profile,calibration=True,case=f'smoke-{name}')
    health=None
    if 'ppo' in m['conditions']:
        health=audit_health(out/'smoke-ppo/ppo-critic-audit',2)
        if not all(health['gates'][k] for k in ('complete','finite','nonzero_updates','zero_initial_values')):
            raise RuntimeError('Conventional PPO failed functional smoke checks')
    linear_checks={}
    if m.get('protocol') == 'slice-numina-folds-v1':
        from math_rl.critic_checks import verify_run
        for name,setting in m['conditions'].items():
            linear_checks[name]=verify_run(out/f'smoke-{name}',setting['method'],folds=setting['critic_folds'])
    write_json(out/'smoke.json',dict(passed=True,conditions=list(m['conditions']),steps=2,
        manifest_sha256=sha256(out/'manifest.json'),ppo_profile=profile,ppo_health=health,
        linear_checks=linear_checks,
        scope='Integration check only; not a method ranking'))


def calibrate(out,m):
    if 'ppo' not in m['conditions']:
        raise ValueError('This study has no conventional PPO arm; calibration is not applicable')
    if not read(out/'smoke.json')['passed']: raise ValueError('Smoke must pass first')
    candidates=[]
    for profile in PPO_PROFILES:
        case=f'calibration-{profile}'
        r=launch(out,m,'ppo',17,m['calibration_steps'],profile,True,case)
        health=audit_health(out/case/'ppo-critic-audit',m['calibration_steps'])
        candidates.append(dict(r,profile=profile,health=health))
    # Explicitly fail rather than silently proceeding with a weak/broken baseline.
    profile=select_profile(candidates)
    write_json(out/'selection.json',dict(profile=profile,candidates=candidates,
        tuning_training_gpu_hours=sum(c['training_gpu_hours'] for c in candidates),
        scope='Validation-only profile selection; no test access'))

    gate = read(out/'smoke.json')
    if gate.get('ppo_profile') != profile:
        case=f'smoke-selected-{profile}'
        launch(out,m,'ppo',19,2,profile=profile,calibration=True,case=case)
        checks=audit_health(out/case/'ppo-critic-audit',2)
        if not all(checks['gates'][k] for k in ('complete','finite','nonzero_updates','zero_initial_values')):
            raise RuntimeError('Selected PPO profile failed its functional smoke')
        gate.update(ppo_profile=profile,selected_profile_check=case)
        write_json(out/'smoke.json',gate)


def run_task(out,m,task):
    gate = out/'smoke.json'
    if not gate.exists():
        raise ValueError('Smoke gate missing: refusing training')
    record = read(gate)
    if (not record.get('passed') or record.get('manifest_sha256') != sha256(out/'manifest.json')
            or set(record.get('conditions', [])) != set(m['conditions'])
            or record.get('ppo_profile') != read(out/'selection.json')['profile']):
        raise ValueError('Smoke gate failed or belongs to another protocol')
    # Rotate arm order across seeds; array index does not change random seeds.
    jobs=[];names=list(m['conditions'])
    for i,seed in enumerate(m['seeds']):
        jobs.extend((name,seed) for name in names[i:]+names[:i])
    if not 0<=task<len(jobs): raise ValueError('Invalid task index')
    name,seed=jobs[task]
    launch(out,m,name,seed,m['steps'],read(out/'selection.json')['profile'])


def report(out,m):
    expected=[(q['id'],q['ground_truth']) for q in read(out/'data/questions.json')['questions'] if q['split']=='test']
    scores,results,base_ref={}, {}, None
    audit=(out/'audit.jsonl').open('w')
    health={}
    try:
        for name,setting in m['conditions'].items():
            arrays=[];runs=[]
            for seed in m['seeds']:
                case=out/f'{name}-seed{seed}';r=read(case/'result.json')
                if r['updates']!=m['steps'] or r['answers']!=m['steps']*64: raise ValueError('Unequal budgets')
                base=runner.read_answers(case/'base-test.jsonl');rows=runner.read_answers(case/'final-test.jsonl')
                for batch in (base,rows):
                    if [(v['id'],v['ground_truth']) for v in batch]!=expected: raise ValueError('Unpaired test set')
                b=[v['score'] for v in base];a=[v['score'] for v in rows]
                if base_ref is None: base_ref=b
                if b!=base_ref: raise ValueError('Starting model scores differ')
                if not np.isclose(np.mean(a),r['final_test']['accuracy'],rtol=0,atol=1e-10): raise ValueError('Inconsistent accuracy')
                for row in rows:
                    audit.write(json.dumps(dict(row,method=name,seed=seed))+'\n')
                if setting['method']=='ppo':
                    health[str(seed)]=audit_health(case/'ppo-critic-audit',m['steps'])
                elif setting['method'] in ('ridge','lstd'):
                    metrics=[json.loads(x) for x in (case/'linear-critic/metrics.jsonl').read_text().splitlines()]
                    if [v['step'] for v in metrics]!=list(range(1,m['steps']+1)): raise ValueError('Missing critic metrics')
                    if m.get('protocol') == 'slice-numina-folds-v1':
                        from math_rl.critic_checks import verify_run
                        verify_run(case,setting['method'],folds=setting['critic_folds'],steps=m['steps'],evaluation_split='test')
                    r['critic_fit_seconds']=sum(v['fit_seconds'] for v in metrics)
                    r['last_critic']=metrics[-1]
                r['verifier_statuses']=dict(Counter(v['verifier_status'] for v in rows))
                r['mean_test_response_tokens']=float(np.mean([v['response_tokens'] for v in rows]))
                r['test_truncation_rate']=float(np.mean([v['capped'] for v in rows]))
                r['accounted_gpu_hours']=r['training_gpu_hours']+2*(r['initialization_seconds']+r['initial_test']['seconds']+r['final_test']['seconds'])/3600
                arrays.append(a);runs.append(r)
            a=np.array(arrays);scores[name]=a
            results[name]=dict(accuracy_mean=float(a.mean()),accuracy_seed_sd=float(a.mean(1).std(ddof=1)),
                accuracy_by_seed=dict(zip(map(str,m['seeds']),a.mean(1).tolist())),
                gain_pp=100*float(a.mean()-np.mean(base_ref)),
                training_gpu_hours_mean=float(np.mean([r['training_gpu_hours'] for r in runs])),
                accounted_gpu_hours_mean=float(np.mean([r['accounted_gpu_hours'] for r in runs])),runs=runs)
    finally: audit.close()
    paired={};names=list(scores)
    tail = .05 / (2 * len(m['primary_comparisons']))
    for i,name in enumerate(names):
        for ref in names[:i]:
            d=scores[name]-scores[ref];rng=np.random.default_rng(813)
            draws=np.array([d[np.ix_(rng.integers(len(d),size=len(d)),rng.integers(d.shape[1],size=d.shape[1]))].mean()*100 for _ in range(4000)])
            key=f'{name}_minus_{ref}'
            # Symmetric output for predeclared primary directions as well.
            paired[key]=dict(mean_pp=float(d.mean()*100),seed_differences_pp=(d.mean(1)*100).tolist(),
                descriptive_95_interval_pp=np.quantile(draws,[.025,.975]).tolist(),
                bonferroni_primary_interval_pp=np.quantile(draws,[tail,1-tail]).tolist())
            reverse=f'{ref}_minus_{name}'
            if reverse in m['primary_comparisons']:
                paired[reverse]=dict(mean_pp=float(-d.mean()*100),seed_differences_pp=(-d.mean(1)*100).tolist(),
                    descriptive_95_interval_pp=np.quantile(-draws,[.025,.975]).tolist(),
                    bonferroni_primary_interval_pp=np.quantile(-draws,[tail,1-tail]).tolist())
    summary=dict(methods=results,paired=paired,primary_comparisons=m['primary_comparisons'],
        base_accuracy=float(np.mean(base_ref)),ppo_health=health,selection=read(out/'selection.json'),
        filter_pretraining=m.get('filter_pretraining'),
        conditions=m['conditions'],scope=m['scope'],limitations=f'{len(m["seeds"])} actor seeds, one selected filter per arm. Approximate seed/question bootstrap; '
        f'{len(m["primary_comparisons"])} primary intervals use Bonferroni nominal coverage; other comparisons exploratory. '
        'No automatic superiority claim. Audit verifier answers before claims. Offline filter learning cost is additional.')
    write_json(out/'summary.json',summary)
    lines=['Numina online: fixed 768 bottlenecks and frozen single-layer controls; no buffer.',
        f'{m["steps"]} updates x 64 answers; {len(m["seeds"])} seeds; final greedy evaluation.',
        f'LSTD critic lambda: {m.get("critic_lambda", .99)}; GAE lambda: 0.95; gamma: 1.',
        'method                   accuracy    training GPUh   accounted GPUh']
    if m.get('protocol') == 'slice-numina-folds-v1':
        lines[0]='Paired fold ablation: frozen sparse-768 encoder/compressor; raw cumulative sums; no buffer.'
        lines.insert(3,'fold1: one head, fit then predict same batch; fold2: two opposite-question heads. Only the fold setting changes.')
    lines += [f'{name:25} {r["accuracy_mean"]:.4f}      {r["training_gpu_hours_mean"]:.3f}          {r["accounted_gpu_hours_mean"]:.3f}' for name,r in results.items()]
    lines += [f'{k}: {paired[k]}' for k in m['primary_comparisons']]
    calibration_label = ('Historical PPO calibration GPUh (not rerun; no PPO arm)' if m.get('protocol') == 'slice-numina-folds-v1'
                         else 'PPO calibration training GPUh (additional)')
    lines += [f'{calibration_label}: {summary["selection"]["tuning_training_gpu_hours"]:.3f}',
        'Offline filter training/generation/extraction are additional costs; see summary.json.',
        (f'PPO health passed for all seeds: {all(v["pass_checks"] for v in health.values())}' if health else
         'Conventional PPO health: not applicable; no conventional PPO arm.'),summary['scope'],summary['limitations']]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
