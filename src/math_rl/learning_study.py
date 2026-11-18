"""Five-method independently scheduled learning curves; fixed monitoring and final test."""
import argparse
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import numpy as np
from . import value_audit as base
from .runner import ROOT
from .provenance import sha256,snapshot,write_json
from .test_followup import read,paired_accuracy
from slice_critic.storage import model_identity

ARMS=['slice-token','slice-chunk','ppo','grpo','rms-layer-token']
SEEDS=[1201,1202,1203]
PRIMARY=[('slice-chunk','slice-token'),('slice-chunk','ppo')]

def actor_tasks():
    tasks=[dict(seed=seed,arm=arm) for seed in SEEDS for arm in ARMS]
    random.Random(20261118).shuffle(tasks)
    return tasks

def coordinates(task,smoke=False):
    tasks=[dict(seed=19,arm=a) for a in ARMS] if smoke else actor_tasks()
    if not 0<=task<len(tasks):raise ValueError('Invalid actor task')
    return tasks[task]['seed'],tasks[task]['arm']

def prepare(source,out):
    if out.exists():raise FileExistsError('New study directory required')
    # Use the existing reproducible data/filter preparation, not old actor weights.
    # Budget HF-only snapshots using the actual base weight size, with headroom.
    size=sum(p.stat().st_size for p in (ROOT/'models/qwen-math').glob('*.safetensors'))
    if not size:raise ValueError('Missing base-model weights')
    reference=ROOT/'outputs/numina-value-repeat-v1/seed1101/slice-token-seed1101/global_step_120/actor/huggingface'
    exported=sum(p.stat().st_size for p in reference.glob('*.safetensors'))
    # FP32 training exports can be larger than the BF16 source snapshot.
    size=max(size,exported) if exported else size*2
    needed=int(size*15*12*1.4)
    print(f'Retained-checkpoint disk estimate including headroom: {needed/2**30:.0f} GiB',flush=True)
    if shutil.disk_usage(out.parent).free<needed:raise ValueError(f'Insufficient space: need approximately {needed/2**30:.0f} GiB for retained HF actors and headroom')
    base.prepare(source,out,updates=120,actor_seed=SEEDS[0]);m=read(out/'manifest.json')
    m.update(protocol='parallel-learning-curves-v1',seeds=SEEDS,arms=ARMS,curve_arms=ARMS,interval=10,
        primary=PRIMARY,monitor_split='val',estimated_disk_bytes=needed,
        actor_tasks=actor_tasks(),max_concurrent_actors=6,hardware_sensitive_timing=True,runtime=snapshot(ROOT)['files'],
        scope='Five original training methods, three fresh seeds; validation learning curves and final test evaluation. Independently allocated actors; time curves are hardware-sensitive. No MC value audit or test-based selection.')
    from .ppo_reward import verifier_command
    m['verifier']=json.loads(subprocess.check_output(verifier_command()+['--provenance'],text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))))
    write_json(out/'manifest.json',m)

def validate(out):
    m=read(out/'manifest.json')
    if m['protocol']!='parallel-learning-curves-v1' or m['seeds']!=SEEDS or m['curve_arms']!=ARMS or m['interval']!=10 or m['steps']!=120 or m['actor_tasks']!=actor_tasks() or m['max_concurrent_actors']!=6:raise ValueError('Curve protocol changed')
    if m['runtime']!=snapshot(ROOT)['files']:raise ValueError('Runtime changed')
    for name,h in m['data_hashes'].items():
        if sha256(out/name)!=h:raise ValueError('Data changed')
    if model_identity(ROOT/'models/qwen-math')!=m['model_hashes'] or sha256(out/'filters/sparse.pt')!=m['filter_sha256']:raise ValueError('Feature basis changed')
    return m

def command(out,m,arm,seed,smoke=False):
    steps=2 if smoke else 120;interval=1 if smoke else 10
    config=dict(m,actor_seed=seed,steps=steps)
    cmd,_=base.command(out,config,arm,False)
    cmd[cmd.index('math_rl.main')]='math_rl.curve_entry'
    case=out/('smoke-runs' if smoke else 'runs')/f'{arm}-seed{seed}'
    cmd += [f'trainer.default_local_dir={case}',f'hydra.run.dir={case}-hydra',
        f'trainer.test_freq={interval}',f'trainer.save_freq={interval}',
        'trainer.max_actor_ckpt_to_keep=20','trainer.max_critic_ckpt_to_keep=1',
        'actor_rollout_ref.actor.checkpoint.save_contents=[hf_model]',
        'critic.checkpoint.save_contents=[model]','trainer.val_before_train=false']
    if smoke:
        cmd += [f'data.train_files={out}/smoke-data/train.parquet',f'data.val_files={out}/smoke-data/val.parquet',
            f'experiment.test_file={out}/smoke-data/val.parquet']
    return cmd,case

def curve_rows(case,steps,interval):
    rows=[json.loads(s) for s in (case/'learning-curve.jsonl').read_text().splitlines()]
    selected=[r for r in rows if r['label']=='base-validation' or r['label']=='periodic']
    if [r['step'] for r in selected]!=list(range(0,steps+1,interval)):raise ValueError(f'Missing/duplicate monitoring points: {case}')
    if any(not 0<=r['accuracy']<=1 for r in selected) or any(b['training_seconds']<a['training_seconds'] for a,b in zip(selected,selected[1:])):raise ValueError('Invalid curve metrics')
    return selected

def train_actor(out,m,task,smoke=False):
    seed,arm=coordinates(task,smoke)
    folder=out/('smoke-runs' if smoke else 'runs');folder.mkdir(exist_ok=True)
    placements=out/('smoke-placements' if smoke else 'placements');placements.mkdir(exist_ok=True)
    placement=placements/f'{arm}-seed{seed}';placement.mkdir(exist_ok=False)
    device=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True)
    if len(os.environ.get('CUDA_VISIBLE_DEVICES','').split(','))!=2:raise ValueError('Two assigned GPUs required per actor')
    metadata=dict(seed=seed,arm=arm,task=task,node=os.uname().nodename,devices=device,
        assigned_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),slurm_job_id=os.environ.get('SLURM_JOB_ID'),manifest_sha256=sha256(out/'manifest.json'))
    write_json(placement/'status.json',dict(metadata,state='started'))
    cmd,case=command(out,m,arm,seed,smoke);log=case.with_suffix('.log')
    if case.exists() or log.exists():raise FileExistsError('Partial actor retained; no silent resume')
    with tempfile.TemporaryDirectory(prefix='lc-',dir='/tmp') as temp:
        env=dict(os.environ,RAY_TMPDIR=temp,RAY_ADDRESS='local',MATH_RL_SEED=str(seed))
        env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',None)
        with log.open('w') as f:subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    steps=2 if smoke else 120;interval=1 if smoke else 10
    if read(case/'result.json')['updates']!=steps:raise ValueError('Incomplete actor')
    curve_rows(case,steps,interval)
    for step in range(interval,steps+1,interval):model_identity(case/f'global_step_{step}/actor/huggingface')
    write_json(placement/'status.json',dict(metadata,state='complete'))

def gate(out,m,smoke):
    seeds=[19] if smoke else SEEDS;cases=[]
    for seed in seeds:
        for arm in ARMS:
            record=read(out/('smoke-placements' if smoke else 'placements')/f'{arm}-seed{seed}'/'status.json')
            if record['state']!='complete' or record['seed']!=seed or record['arm']!=arm or record['manifest_sha256']!=sha256(out/'manifest.json'):raise ValueError('Incomplete actor placement')
            case=out/('smoke-runs' if smoke else 'runs')/f'{arm}-seed{seed}'
            curve_rows(case,2 if smoke else 120,1 if smoke else 10)
            path=case/f'global_step_{2 if smoke else 120}/actor/huggingface'
            cases.append(dict(arm=arm,seed=seed,path=str(path),identity=model_identity(path)))
    if smoke:
        write_json(out/'smoke.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json')));return
    cases.append(dict(arm='base',seed=0,path=str(ROOT/'models/qwen-math'),identity=model_identity(ROOT/'models/qwen-math')))
    test_questions=[q for q in read(out/'data/questions.json')['questions'] if q['split']=='test']
    if len(test_questions)!=2000:raise ValueError('Test cohort changed')
    write_json(out/'test-plan.json',dict(cases=cases,questions=test_questions,verifier=m['verifier'],manifest_sha256=sha256(out/'manifest.json')))

def report(out,m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curves={a:[curve_rows(out/'runs'/f'{a}-seed{s}',120,10) for s in SEEDS] for a in ARMS}
    colors=['#2765A7','#A064AA','#CC782A','#248A67','#858B93']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for arm,color in zip(ARMS,colors):
        ys=np.array([[r['accuracy']*100 for r in rows] for rows in curves[arm]])
        xs=np.array([[r['training_gpu_hours'] for r in rows] for rows in curves[arm]])
        steps=np.arange(0,121,10);mean=ys.mean(0);sd=ys.std(0,ddof=1)
        axes[0].plot(steps,mean,label=arm,color=color);axes[0].fill_between(steps,mean-sd,mean+sd,color=color,alpha=.13)
        grid=np.linspace(0,float(xs[:,-1].min()),100)
        sampled=np.array([np.interp(grid,x,y) for x,y in zip(xs,ys)])
        axes[1].plot(grid,sampled.mean(0),label=arm,color=color);axes[1].fill_between(grid,sampled.mean(0)-sampled.std(0,ddof=1),sampled.mean(0)+sampled.std(0,ddof=1),color=color,alpha=.13)
    for ax in axes:ax.set_ylabel('Monitoring accuracy (%)');ax.grid(alpha=.2)
    axes[0].set_xlabel('Training update');axes[1].set_xlabel('Training GPU-hours (evaluation/checkpoint time excluded)');axes[0].legend(fontsize=8)
    fig.suptitle('3 seeds · mean ± seed SD · validation monitoring · hardware-sensitive time axis')
    fig.savefig(out/'learning-curves.png',dpi=180);fig.savefig(out/'learning-curves.pdf');plt.close(fig)
    plan=read(out/'test-plan.json');results=[read(out/'test'/f'{i}.json') for i in range(16)]
    expected=[q['id'] for q in plan['questions']];arrays={};metrics={}
    for r,c in zip(results,plan['cases'],strict=True):
        if r['case']!=c or r['manifest_sha256']!=sha256(out/'manifest.json') or [a['id'] for a in r['answers']]!=expected:raise ValueError('Test result mismatch')
    for arm in ARMS:
        values=[r for r in results if r['case']['arm']==arm]
        arrays[arm]=np.array([[a['score'] for a in r['answers']] for r in values]);acc=arrays[arm].mean(1)*100
        metrics[arm]=dict(mean=float(acc.mean()),seed_sd=float(acc.std(ddof=1)),by_seed=acc.tolist())
    contrasts={a+'_minus_'+b:paired_accuracy(arrays[a],arrays[b]) for a,b in PRIMARY}
    write_json(out/'summary.json',dict(curves=curves,final_test=metrics,primary=contrasts,base_test_percent=results[-1]['accuracy']*100,
        placements=[read(out/'placements'/f'{a}-seed{s}'/'status.json') for s in SEEDS for a in ARMS],
        scope='15 independently scheduled actors, at most 6 concurrent. Paired seed/question accuracy contrasts; hardware-sensitive time curves, not a controlled speed benchmark. Interpolation within common seed coverage per method, without extrapolation. Reused data.'))
    rows=['Learning curves complete: five methods, three fresh seeds, 120 updates; monitoring every 10.',
        'Monitoring is validation; final accuracy below is TEST. GPUh curves exclude monitoring/checkpoint time and remain hardware-sensitive.']
    rows += [f'{a}: {v["mean"]:.2f}% test; seed SD {v["seed_sd"]:.2f} pp' for a,v in metrics.items()]
    rows += [f'{k}: {v}' for k,v in contrasts.items()]
    (out/'report.txt').write_text('\n'.join(rows)+'\n')
    write_json(out/'verification.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),files={n:sha256(out/n) for n in ['report.txt','summary.json','learning-curves.png','learning-curves.pdf']}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','smoke','smoke-gate','train','train-gate','test','report'],required=True)
    p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--task',type=int,default=0)
    a=p.parse_args();out=a.out.resolve()
    if a.stage=='prepare':prepare(a.source.resolve(),out);return
    m=validate(out)
    if a.stage in ['train','train-gate','test','report']:
        g=read(out/'smoke.json')
        if not g['passed'] or g['manifest_sha256']!=sha256(out/'manifest.json'):raise ValueError('Passed smoke required')
    if a.stage in ['smoke','train']:train_actor(out,m,a.task,a.stage=='smoke')
    elif a.stage in ['smoke-gate','train-gate']:gate(out,m,a.stage=='smoke-gate')
    elif a.stage=='report':report(out,m)
    else:
        from .test_followup import evaluate
        plan=read(out/'test-plan.json')
        if plan['manifest_sha256']!=sha256(out/'manifest.json'):raise ValueError('Changed test plan')
        evaluate(out,dict(m,**{k:plan[k] for k in ['cases','questions','verifier']}),a.task,False)

if __name__=='__main__':main()
