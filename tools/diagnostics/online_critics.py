"""Fresh seven-arm PPO comparison, with mandatory validation-only integration gate."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from math_rl import runner
from math_rl.experiment import launch
from math_rl.critic_checks import verify_run
from math_rl.ppo_baseline import audit_health, PPO_PROFILES
from math_rl.provenance import snapshot,sha256,write_json

ROOT=runner.ROOT
TOOLS=('online_critics.py','submit_online_critics.sh','submit_critic_study.py')
ARMS={
    'ridge':dict(method='ridge'),
    'mlp-frozen':dict(method='mlp_frozen'),
    'mlp-joint':dict(method='mlp_joint'),
    'lstd-005':dict(method='lstd',trace_lambda=.05),
    'lstd-099':dict(method='lstd',trace_lambda=.99),
    'ppo':dict(method='ppo'),
    'grpo':dict(method='grpo'),
}
PAIRS=[('mlp-frozen','ridge'),('mlp-joint','ridge'),('lstd-005','ridge'),('lstd-099','ridge'),
       ('mlp-joint','mlp-frozen'),('ridge','ppo'),('ridge','grpo')]


def read(path): return json.loads(Path(path).read_text())


def prepare(source,out):
    source,out=source.resolve(),out.resolve()
    if out==source or source in out.parents: raise ValueError('Separate new output required')
    m=read(source/'manifest.json')
    if m.get('protocol') not in ('slice-numina-no-folds-v1','slice-numina-online-v1'):
        raise ValueError('Expected an existing final Numina study')
    profile=read(source/'selection.json')['profile']
    if profile not in PPO_PROFILES: raise ValueError('Unknown PPO profile')
    for name,h in m['data_hashes'].items():
        if sha256(source/'data'/name)!=h: raise ValueError('Source data changed')
    for name,h in m['model_hashes'].items():
        if sha256(ROOT/'models/qwen-math'/name)!=h: raise ValueError('Model changed')
    f=m['filters']['sparse'];path=source/'filters'/Path(f['path']).name
    if sha256(path)!=f['sha256']: raise ValueError('Source filter changed')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'data',out/'data')
    (out/'filters').mkdir();shutil.copy2(path,out/'filters'/path.name)
    m=dict(protocol='slice-seven-critics-gpu-v1',conditions=ARMS,seeds=[501,502,503],steps=120,
        data_hashes=m['data_hashes'],model_hashes=m['model_hashes'],layers=m['layers'],rms_eps=m['rms_eps'],
        filter=dict(path=str(out/'filters'/path.name),sha256=f['sha256']),profile=profile,
        provenance=snapshot(ROOT),tools={name:sha256(Path(__file__).with_name(name)) for name in TOOLS},
        source=dict(path=str(source),manifest_sha256=sha256(source/'manifest.json')),
        scope='New actors, reused inspected questions and offline-selected sparse compressor. '
              'No output LayerNorm/clipping. One cumulative head for Ridge/LSTD; '
              'fresh GAE-return neural fits with fresh sigmoid heads, no offline MLP head initialization. '
              'Neural prediction/fitting uses a single owning actor-worker GPU, with streamed inputs. '
              'No driver GPU context, CPU fitting or unsynchronized critic replicas. '
              'Feature transport and RPC overhead are included in wall time. All arms allocate two A5000 GPUs. '
              'Fresh actor seeds, not an independent dataset benchmark. PPO validation profile reused.')
    write_json(out/'manifest.json',m)
    write_json(out/'status.json',dict(state='prepared'))


def validate(out):
    m=read(out/'manifest.json')
    if m['conditions']!=ARMS or m['protocol']!='slice-seven-critics-gpu-v1': raise ValueError('Wrong protocol')
    if m['seeds']!=[501,502,503] or m['steps']!=120: raise ValueError('Locked budget changed')
    if (snapshot(ROOT)['files']!=m['provenance']['files']
            or {name:sha256(Path(__file__).with_name(name)) for name in TOOLS}!=m['tools']):
        raise ValueError('Code changed after preparation; create a new study')
    for name,h in m['data_hashes'].items():
        if sha256(out/'data'/name)!=h: raise ValueError('Data changed')
    for name,h in m['model_hashes'].items():
        if sha256(ROOT/'models/qwen-math'/name)!=h: raise ValueError('Model changed')
    if sha256(m['filter']['path'])!=m['filter']['sha256']: raise ValueError('Filter changed')
    return m


def command(out,m,name,seed,steps,calibration=False,case=None):
    method=ARMS[name]['method']
    cmd=runner.command(out,method,seed,steps,feature_mode='frozen',
        ppo_profile=m['profile'] if method=='ppo' else 'not_applicable',
        evaluation_split='val' if calibration else 'test',case_name=case or f'{name}-seed{seed}')
    if method in ('ridge','lstd','mlp_frozen','mlp_joint'):
        cmd+=['actor_rollout_ref.model.critic_encoder_precision=fp32',
              'actor_rollout_ref.model.critic_feature_layer=bottleneck',
              f'actor_rollout_ref.model.critic_feature_view={"raw_concat" if method=="mlp_joint" else "compressed"}',
              f'actor_rollout_ref.model.critic_bottleneck_path={m["filter"]["path"]}',
              f'actor_rollout_ref.model.critic_bottleneck_sha256={m["filter"]["sha256"]}',
              f'actor_rollout_ref.model.critic_bottleneck_layers={json.dumps(m["layers"],separators=(",",":"))}',
              f'actor_rollout_ref.model.critic_bottleneck_rms_eps={m["rms_eps"]}',
              'experiment.critic_folds=1',
              f'experiment.critic_statistics={"fresh_neural" if method.startswith("mlp_") else "cumulative"}',
              f'experiment.critic_lambda={ARMS[name].get("trace_lambda",.99)}']
        if method=='mlp_joint': cmd+=['experiment.neural_head_lr=0.00003']
    return cmd


def execute(out,m,name,seed,steps,smoke=False):
    # Reuse the hardened launcher but temporarily supply this protocol's command.
    import math_rl.experiment as experiment
    original=experiment.command
    experiment.command=lambda out,m,name,seed,steps,profile=None,calibration=False,case=None: command(
        out,m,name,seed,steps,calibration,case)
    try:
        return launch(out,m,name,seed,steps,profile=m['profile'],calibration=smoke,
                      case=f'smoke-{name}' if smoke else None)
    finally: experiment.command=original


def audit(out,m,name,seed,steps,smoke=False):
    case=out/(f'smoke-{name}' if smoke else f'{name}-seed{seed}')
    method=ARMS[name]['method']
    if method in ('ridge','lstd'):
        return verify_run(case,method,folds=1,steps=steps,evaluation_split='val' if smoke else 'test',
                          trace_lambda=ARMS[name].get('trace_lambda',.99))
    if method=='ppo':
        health=audit_health(case/'ppo-critic-audit',steps)
        if not all(health['gates'][k] for k in ('complete','finite','zero_initial_values','nonzero_updates')):
            raise ValueError('PPO functional checks failed')
        return health
    if method.startswith('mlp_'):
        rows=[json.loads(r) for r in (case/'neural-critic/metrics.jsonl').read_text().splitlines()]
        if (len(rows)!=steps or [r['step'] for r in rows]!=list(range(1,steps+1))
                or any(r['answers']!=64 or r['cumulative_matrices'] or r['joint']!=(method=='mlp_joint')
                       or r['backend']!='cuda' or r['owner_rank']!=0 or not r['frozen_encoder']
                       or not np.isfinite(r['mse_after']) for r in rows)
                or (case/'linear-critic').exists()): raise ValueError('Neural critic audit failed')
        state=torch.load(case/'neural-critic/latest.pt',map_location='cpu',weights_only=True)
        if state['step']!=steps or any(not torch.isfinite(v).all() for v in state['state'].values()):
            raise ValueError('Invalid neural checkpoint')
        from slice_critic import load_ppo_features
        initial,mean,scale=load_ppo_features(m['filter']['path'],m['filter']['sha256'])
        saved=state['state']
        if (not torch.equal(saved['mean'],mean) or not torch.equal(saved['scale'],scale)
                or not torch.count_nonzero(saved['head.2.weight'])):
            raise ValueError('Normalization changed or neural head did not update')
        if method=='mlp_frozen' and any(not torch.equal(saved[f'compressor.projection.{k}'],v)
                for k,v in initial.projection.state_dict().items()):
            raise ValueError('Frozen compressor changed')
        if method=='mlp_joint' and all(torch.equal(saved[f'compressor.projection.{k}'],v)
                for k,v in initial.projection.state_dict().items()):
            raise ValueError('Trainable compressor did not change')
        return dict(passed=True,backend='cuda',owner_rank=0,steps=steps,joint=method=='mlp_joint',
                    fitting_seconds=sum(r['seconds'] for r in rows),
                    prediction_rpc_seconds=sum(r['prediction_rpc_seconds'] for r in rows),
                    fit_rpc_seconds=sum(r['fit_rpc_seconds'] for r in rows),
                    process_peak_cuda_bytes=max(r['process_peak_cuda_bytes'] for r in rows),
                    max_rollout_feature_bytes=max(r['feature_bytes'] for r in rows),cumulative_matrices=False)
    return dict(passed=True)


def report(out,m):
    scores,metrics,checks={}, {}, {}
    ids=None;base=None
    expected=[(q['id'],q['ground_truth']) for q in read(out/'data/questions.json')['questions'] if q['split']=='test']
    for name in ARMS:
        arrays=[];cost=[]
        checks[name]={}
        for seed in m['seeds']:
            case=out/f'{name}-seed{seed}'
            r=read(case/'result.json')
            if r['updates']!=m['steps'] or r['answers']!=m['steps']*64: raise ValueError('Unequal budgets')
            rows=runner.read_answers(case/'final-test.jsonl');initial=runner.read_answers(case/'base-test.jsonl')
            current=[(v['id'],v['ground_truth']) for v in rows]
            if ids is None: ids=current;base=[v['score'] for v in initial]
            if current!=expected or current!=ids or [(v['id'],v['ground_truth']) for v in initial]!=ids or [v['score'] for v in initial]!=base:
                raise ValueError('Unpaired evaluations or different base model scores')
            values=[v['score'] for v in rows]
            if not np.isclose(np.mean(values),r['final_test']['accuracy']): raise ValueError('Inconsistent result')
            arrays.append(values);cost.append(r['training_gpu_hours'])
            checks[name][str(seed)]=audit(out,m,name,seed,m['steps'])
        scores[name]=np.asarray(arrays)
        metrics[name]=dict(accuracy_mean=float(scores[name].mean()),
            accuracy_by_seed=dict(zip(map(str,m['seeds']),scores[name].mean(1).tolist())),
            accuracy_seed_sd=float(scores[name].mean(1).std(ddof=1)),training_gpu_hours_mean=float(np.mean(cost)))
    paired={}
    for a,b in PAIRS:
        d=scores[a]-scores[b];rng=np.random.default_rng(42)
        samples=np.array([d[rng.integers(len(d),size=len(d))][:,rng.integers(d.shape[1],size=d.shape[1])].mean()*100
                          for _ in range(4000)])
        paired[f'{a}_minus_{b}']=dict(mean_pp=float(d.mean()*100),
            seed_differences_pp=(d.mean(1)*100).tolist(),descriptive_95_interval_pp=np.quantile(samples,[.025,.975]).tolist(),
            adjusted_interval_pp=np.quantile(samples,[.025/len(PAIRS),1-.025/len(PAIRS)]).tolist())
    write_json(out/'summary.json',dict(methods=metrics,paired=paired,checks=checks,scope=m['scope'],
        limitations='Three seeds; approximate seed/question bootstrap; seven planned Bonferroni intervals. '
                    'Inspected test questions, verifier audit still required. Single-owner GPU neural fitting.'))
    lines=[m['scope'],'120 PPO updates x 64 answers; 3 seeds; GAE lambda=.95; gamma=1.',
           'LSTD trace lambdas=.05 and .99; Ridge epsilon=.01; one cumulative system; no buffer.']
    lines.extend(f'{k}: {v}' for k,v in metrics.items())
    lines.extend(f'{k}: {v}' for k,v in paired.items())
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    write_json(out/'status.json',dict(state='complete'))


def smoke_task(out,m,task):
    if not 0<=task<len(ARMS): raise ValueError('Smoke task must be 0..6')
    name=list(ARMS)[task]
    execute(out,m,name,19,2,True)
    checks=audit(out,m,name,19,2,True)
    folder=out/'smoke-records';folder.mkdir(exist_ok=True)
    case=out/f'smoke-{name}'
    write_json(folder/f'{name}.json',dict(passed=True,checks=checks,
        manifest_sha256=sha256(out/'manifest.json'),result_sha256=sha256(case/'result.json')))


def smoke_gate(out,m):
    checks={}
    for name in ARMS:
        row=read(out/'smoke-records'/f'{name}.json')
        if (not row['passed'] or row['manifest_sha256']!=sha256(out/'manifest.json')
                or row['result_sha256']!=sha256(out/f'smoke-{name}'/'result.json')):
            raise ValueError('Missing/stale smoke record')
        # Re-audit the saved artifacts, rather than trusting a passed boolean.
        checks[name]=audit(out,m,name,19,2,True)
    write_json(out/'smoke.json',dict(passed=True,checks=checks,manifest_sha256=sha256(out/'manifest.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','smoke','smoke-task','smoke-gate','run','report'],required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--task',type=int,default=0)
    a=p.parse_args();out=a.out.resolve()
    if a.stage=='prepare':
        if a.source is None:p.error('--source required')
        prepare(a.source,out);return
    m=validate(out)
    if a.stage=='smoke':
        for task in range(len(ARMS)):smoke_task(out,m,task)
        smoke_gate(out,m)
    elif a.stage=='smoke-task':smoke_task(out,m,a.task)
    elif a.stage=='smoke-gate':smoke_gate(out,m)
    elif a.stage=='run':
        gate=read(out/'smoke.json')
        if not gate['passed'] or gate['manifest_sha256']!=sha256(out/'manifest.json'): raise ValueError('Smoke gate failed')
        jobs=[];names=list(ARMS)
        for i,seed in enumerate(m['seeds']): jobs.extend((n,seed) for n in names[i:]+names[:i])
        if not 0<=a.task<len(jobs):p.error('Task must be 0..20')
        name,seed=jobs[a.task];execute(out,m,name,seed,m['steps'])
    else: report(out,m)


if __name__=='__main__': main()
