"""Same-allocation randomized fixed-workload timing of actual VERL update paths."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
from .test_followup import read,ARMS
from .provenance import sha256,snapshot,write_json
from .runner import ROOT


def reusable_workload(folder,source,feature_manifest):
    """Reuse only immutable generated data, never a failed trial or old runtime."""
    old=read(folder/'manifest.json')
    if old['protocol']!='fixed-workload-cost-v1' or old['source']!=str(source) or old['feature_manifest']!=feature_manifest:
        raise ValueError('Reusable workload source/features differ')
    if sha256(folder/'workload.json')!=old['workload_sha256']:raise ValueError('Reusable workload changed')
    for name,h in old['data_hashes'].items():
        if sha256(folder/name)!=h:raise ValueError('Reusable data changed')
    data=read(folder/'workload.json')
    if len(data['answers'])!=16 or set(data['answers'])!=set(data['rewards']):raise ValueError('Incomplete workload')
    for key,answers in data['answers'].items():
        if len(answers)!=4 or len(data['rewards'][key])!=4:raise ValueError('Wrong group size')
        if any(not 0<len(a)<=2048 or any(type(t) is not int or t<0 for t in a) for a in answers):raise ValueError('Invalid original actions')
        if any(v not in (0,1) for v in data['rewards'][key]):raise ValueError('Invalid rewards')
    return old


def prepare(source,out,quick=False,passed_smoke=None,reuse_workload=None):
    from datasets import Dataset
    source=source.resolve();out=out.resolve()
    if out==source or source in out.parents:raise ValueError('Separate benchmark directory required')
    if not quick and passed_smoke is None:raise ValueError('A passed benchmark smoke is required')
    if reuse_workload is not None and (not quick or passed_smoke is not None):raise ValueError('Workload-only recovery is for a fresh smoke')
    smoke_manifest=None
    if passed_smoke is not None:
        smoke_manifest=read(passed_smoke/'manifest.json');g=read(passed_smoke/'verification.json')
        if not g['passed'] or g['manifest_sha256']!=sha256(passed_smoke/'manifest.json') or not smoke_manifest['quick']:raise ValueError('Invalid benchmark smoke')
        if smoke_manifest['runtime']!=snapshot(ROOT)['files']:raise ValueError('Runtime changed after benchmark smoke')
    parent=read(source/'manifest.json'); verification=read(source/'main/verification.json')
    if not verification['passed'] or verification['manifest_sha256']!=sha256(source/'manifest.json'):raise ValueError('Unverified source')
    out.mkdir(parents=True,exist_ok=False);(out/'data').mkdir()
    child=source/'seed1101';m=read(child/'manifest.json')
    if sha256(child/'manifest.json')!=parent['children']['1101']:raise ValueError('Source child changed')
    from slice_critic.storage import model_identity
    if model_identity(ROOT/'models/qwen-math')!=m['model_hashes']:raise ValueError('Base actor changed')
    if sha256(child/'filters/sparse.pt')!=m['filter_sha256']:raise ValueError('Compressor changed')
    reused=passed_smoke or reuse_workload
    if reused:reusable_workload(reused,source,m)
    for name in ['train','val']:
        p=child/f'data/{name}.parquet'
        if sha256(p)!=m['data_hashes'][f'data/{name}.parquet']:raise ValueError('Changed source data')
        Dataset.from_parquet(str(p)).select(range(16)).to_parquet(str(out/f'data/{name}.parquet'))
    from .value_audit_gpu import Generator,grade
    from slice_critic.config import completion_prompt
    settings=dict(reference_seed=41017,response_cap=2048,max_model_len=2560)
    generator=None if reused else Generator(ROOT/'models/qwen-math',settings);answers={};rewards={}
    for i,row in enumerate([] if reused else Dataset.from_parquet(str(out/'data/train.parquet'))):
        ids=generator.tokenizer.encode(completion_prompt(row['prompt'][1]['content']),add_special_tokens=False)
        if not 0<len(ids)<=512:raise ValueError('Overlong benchmark prompt')
        samples=grade(generator,row['reward_model']['ground_truth'],generator.generate(ids,4,2048,41017+i),[])
        key=','.join(map(str,ids));answers[key]=[s['token_ids'] for s in samples];rewards[key]=[s['score'] for s in samples]
    if reused:
        import shutil
        shutil.copyfile(reused/'workload.json',out/'workload.json')
    else:
        if len(answers)!=16:raise ValueError('Duplicate prompts')
        write_json(out/'workload.json',dict(answers=answers,rewards=rewards,scope='64 fixed original-token answers from common base actor, training questions only.'))
    # Initial and accumulated matrices; same accumulated basis within each token/chunk pair.
    statistics={}
    for arm in ARMS[:4]:
        p=child/f'{arm}-seed1101/linear-critic/statistics.pt'
        statistics[arm]=dict(path=str(p),sha256=sha256(p))
    write_json(out/'manifest.json',dict(protocol='fixed-workload-cost-v1',source=str(source),feature_manifest=m,
        workload_sha256=sha256(out/'workload.json'),data_hashes={str(p.relative_to(out)):sha256(p) for p in (out/'data').glob('*.parquet')},
        runtime=snapshot(ROOT)['files'],reused_workload_from=str(reused) if reused else None,statistics=statistics,quick=quick,rounds=1 if quick else 3,warmup_updates=1 if quick else 2,measured_updates=2 if quick else 3,
        scope='Computational overhead only. Fixed base actor/actions, fresh optimizers each trial; accumulated regime restores only Ridge statistics, not learned actors/optimizers. No learning or time-to-quality claim.'))


def trial_command(out,m,arm,regime,round_index):
    from .value_audit import command
    child=m['feature_manifest']; config=dict(child,actor_seed=41017,steps=m['warmup_updates']+m['measured_updates'],ppo_profile='higher_lr')
    cmd,case=command(out,config,arm)
    case=out/'trials'/f'{regime}-round{round_index}-{arm}'
    # benchmark_entry installs replay in driver; external_lib installs it in workers.
    cmd[cmd.index('math_rl.main')]='math_rl.benchmark_entry'
    cmd += [f'trainer.default_local_dir={case}',f'hydra.run.dir={case}-hydra',
        'actor_rollout_ref.model.external_lib=math_rl.benchmark_runtime',
        'actor_rollout_ref.rollout.calculate_log_probs=false','trainer.val_before_train=false',
        'trainer.test_freq=-1','trainer.save_freq=-1','trainer.logger=[console]']
    # Filter was pinned by the source; no copying or mutation of its checkpoint.
    cmd=[v for v in cmd if not v.startswith('actor_rollout_ref.model.critic_bottleneck_path=')]
    if arm.startswith('slice'):
        cmd += [f'actor_rollout_ref.model.critic_bottleneck_path={Path(m["source"])/"seed1101/filters/sparse.pt"}']
    env=dict(os.environ,MATH_RL_BENCH_WORKLOAD=str(out/'workload.json'),MATH_RL_BENCH_TIMINGS=str(case/'benchmark-steps.jsonl'),MATH_RL_SEED='41017',RAY_ADDRESS='local')
    env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',None);env.pop('MATH_RL_BENCH_STATISTICS',None)
    if regime=='accumulated' and arm in m['statistics']:env['MATH_RL_BENCH_STATISTICS']=m['statistics'][arm]['path']
    return cmd,env,case


def execute(out):
    import tempfile
    from .benchmark_metrics import read_steps
    m=read(out/'manifest.json')
    if m['runtime']!=snapshot(ROOT)['files'] or sha256(out/'workload.json')!=m['workload_sha256']:raise ValueError('Benchmark changed')
    for name,h in m['data_hashes'].items():
        if sha256(out/name)!=h:raise ValueError('Workload changed')
    for v in m['statistics'].values():
        if sha256(v['path'])!=v['sha256']:raise ValueError('Statistics changed')
    (out/'.running').mkdir(exist_ok=False);(out/'trials').mkdir(exist_ok=True)
    gpu_query=['nvidia-smi','--query-gpu=index,uuid,name,power.draw,clocks.sm,utilization.gpu,memory.used','--format=csv,noheader,nounits']
    gpu_identity=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)
    if len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=2:raise ValueError('Benchmark requires two assigned CUDA devices')
    schedule=[];rng=random.Random(41017)
    for round_index in range(m['rounds']):
        order=[(a,r) for a in ARMS for r in (['initial'] if m['quick'] else ['initial','accumulated'])];rng.shuffle(order)
        schedule += [(round_index,a,r) for a,r in order]
    write_json(out/'schedule.json',dict(node=os.uname().nodename,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_identity=gpu_identity,schedule=schedule))
    trials=[]
    for round_index,arm,regime in schedule:
        if subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)!=gpu_identity:raise ValueError('GPU pair changed')
        cmd,env,case=trial_command(out,m,arm,regime,round_index)
        log=case.with_suffix('.log');telemetry=case.with_suffix('.telemetry.jsonl')
        if case.exists() or log.exists():raise FileExistsError('Trials are never silently resumed/replaced')
        # Ignore Slurm's nested TMPDIR: Ray appends a session/socket suffix and
        # Linux AF_UNIX filenames cannot exceed 107 bytes.
        with tempfile.TemporaryDirectory(prefix='b-',dir='/tmp') as temp:
            env['RAY_TMPDIR']=temp;start=time.perf_counter()
            with log.open('w') as f,telemetry.open('w') as t:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
                while True:
                    sample=subprocess.run(gpu_query,capture_output=True,text=True)
                    t.write(json.dumps(dict(elapsed=time.perf_counter()-start,data=sample.stdout,error=sample.stderr))+'\n');t.flush()
                    try:code=process.wait(timeout=2);break
                    except subprocess.TimeoutExpired:pass
            if code:raise RuntimeError(f'Failed trial retained: {log}')
        total_steps=m['warmup_updates']+m['measured_updates']
        result=read(case/'result.json')
        if result['updates']!=total_steps or result['answers']!=64*total_steps:raise ValueError('Incomplete benchmark updates')
        steps=read_steps(case/'benchmark-steps.jsonl',total_steps,m['workload_sha256'])
        measured=[steps[i] for i in range(m['warmup_updates']+1,total_steps+1)]
        trials.append(dict(arm=arm,regime=regime,round=round_index,steps=measured,total_process_seconds=time.perf_counter()-start))
        write_json(out/'trials.json',dict(trials=trials))
    rows=[f'Fixed-workload timing: same allocation; {m["rounds"]} rounds; discard {m["warmup_updates"]} warm-up updates; measure {m["measured_updates"]}.',m['scope'],'arm / statistics regime       median step s   between-round SD s   peak allocated GiB']
    metrics={}
    for arm in ARMS:
        for regime in (['initial'] if m['quick'] else ['initial','accumulated']):
            matches=[r for r in trials if r['arm']==arm and r['regime']==regime]
            values=[np.mean([s['timing_s/step'] for s in r['steps']]) for r in matches]
            memory=[s.get('perf/max_memory_allocated_gb') for r in matches for s in r['steps']]
            if any(v is None for v in memory):raise ValueError('Missing peak-memory metrics')
            value=dict(round_mean_step_seconds=list(map(float,values)),median_seconds=float(np.median(values)),sd_seconds=float(np.std(values,ddof=1)) if len(values)>1 else 0.,peak_memory_gib=float(max(memory)))
            metrics[f'{arm}/{regime}']=value
            rows.append(f'{arm+" / "+regime:<34} {value["median_seconds"]:.2f}           {value["sd_seconds"]:.2f}                 {value["peak_memory_gib"]:.2f}')
    (out/'report.txt').write_text('\n'.join(rows)+'\n');write_json(out/'summary.json',dict(metrics=metrics,scope=m['scope'],notes='No rollout generation timing: identical cached actions. Step includes replay/synchronization, reward, feature extraction, advantages and actual PPO actor/critic updates. Peak allocated memory is logged training memory, not total GPU reservation; nvidia-smi telemetry retained. No outcome-based exclusions.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    regimes=['initial'] if m['quick'] else ['initial','accumulated']
    for offset,regime in enumerate(regimes):
        x=np.arange(len(ARMS))+(offset-.5)*.2
        vs=[metrics[f'{a}/{regime}'] for a in ARMS]
        axes[0].errorbar(x,[v['median_seconds'] for v in vs],yerr=[v['sd_seconds'] for v in vs],fmt='o',capsize=4,label=regime)
        axes[1].plot(x,[v['peak_memory_gib'] for v in vs],'o',label=regime)
    for ax in axes:
        ax.set_xticks(range(len(ARMS)),ARMS,rotation=25);ax.grid(axis='y',alpha=.2);ax.legend()
    axes[0].set(ylabel='Step wall seconds',title='Median round mean; bars = between-round SD')
    axes[1].set(ylabel='Peak allocated GPU memory (GiB)',title='Training allocator metric, not total GPU reservation')
    fig.savefig(out/'controlled-cost.png',dpi=180);plt.close(fig)
    write_json(out/'verification.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json')))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','run'],required=True);p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--quick',action='store_true');p.add_argument('--passed-smoke',type=Path);p.add_argument('--reuse-workload',type=Path)
    a=p.parse_args();out=a.out.resolve()
    if a.stage=='prepare':prepare(a.source,out,a.quick,a.passed_smoke,a.reuse_workload)
    else:execute(out)

if __name__=='__main__':main()
