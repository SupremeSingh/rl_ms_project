"""Two-arm transferred-feature PPO. Validation pilots precede locked actor runs."""
import argparse
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
import torch
from slice_critic.storage import read_json as read,write_json,save_tensor,sha256,model_identity,load_tensor
from slice_critic.inference import load_filter
from slice_critic.config import completion_prompt
from slice_critic.text import family_key
from math_rl import runner
from math_rl.provenance import snapshot

ROOT=runner.ROOT
ARMS=('slice','rms_two_thirds')
SEEDS=(701,702,703,704,705)
EPS=(.01,100.,1000.,10000.)
# NumPy 2.4 removed trapz; Duke's NumPy 1.26 does not have trapezoid.
_trapezoid = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def tools():
    return {p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('submit_online_transfer.py'),Path(__file__).with_name('run_online_transfer.sh')]}


def validate(out):
    m=read(out/'manifest.json')
    if m['protocol']!='slice-transfer-ppo-v1' or m['runtime']!=snapshot(ROOT)['files'] or m['tools']!=tools():raise ValueError('Study code changed; prepare a new directory')
    for name,h in m['data_hashes'].items():
        if sha256(out/'data'/name)!=h:raise ValueError('Data changed')
    if model_identity(m['model_path'])!=m['model_hashes'] or sha256(out/'filter.pt')!=m['filter_sha256']:raise ValueError('Frozen model/filter changed')
    return m


def prepare(source,out):
    if out.exists():raise ValueError('Use a new output directory')
    sm=read(source/'manifest.json');locked=read(source/'fits/42/locked.json')
    if sm['protocol']!='slice-transfer-clarify-v1' or read(source/'status/evaluate-0.json')['state']!='complete':raise ValueError('Completed clarification study required')
    if locked['manifest']!=sha256(source/'manifest.json') or locked['packed']!=sha256(source/'packed/complete.json') or locked['models']!=sha256(source/'fits/42/models.pt'):raise ValueError('Source fit changed')
    if model_identity(ROOT/'models/qwen-math')!=sm['model_hashes']:
        raise ValueError('PPO preflight model at models/qwen-math differs from frozen source encoder')
    fitted=load_tensor(source/'fits/42/models.pt')['representations']['source']
    q=read(source/'questions.json')
    if sha256(source/'questions.json')!=sm['questions_sha256']:raise ValueError('Source questions changed')
    # Verify the source compressor never used any MATH family, independently of cached head fits.
    original=Path(sm['source']);oq=read(original/'questions.json')
    if sha256(original/'questions.json')!=read(original/'manifest.json')['questions_sha256']:raise ValueError('Original source questions changed')
    if sha256(original/'manifest.json')!=sm['source_manifest_sha256']:raise ValueError('Original source changed')
    blocked=set(sm['math_blocked_families']);source_families={v['family'] for v in oq if v['domain']=='numina' and v['split'] in ('train','val') and v['family'] not in blocked}
    if source_families & {v['family'] for v in q}:raise ValueError('Source feature/target overlap')
    from transformers import AutoTokenizer
    from math_rl.data import write_online_data
    tokenizer=AutoTokenizer.from_pretrained(sm['model_path'],local_files_only=True)
    rows=[]
    for v in q:
        if v['domain']!='math':continue
        if len(tokenizer.encode(completion_prompt(v['question']),add_special_tokens=False))>512:raise ValueError('Prompt mismatch')
        rows.append(dict(id=v['id'],question=v['question'],ground_truth=v['reference'],source_index=v['source_index'],split=v['split'],family=v['family']))
    # Calibration/monitoring share validation only; no final test scores select epsilon.
    if min(sum(v['split']==s for v in rows) for s in ('train','val','test'))<16:raise ValueError('Insufficient target questions')
    out.mkdir(parents=True);write_online_data(out/'data',rows,data_source='math_numeric')
    rms_eps=read(Path(sm['model_path'])/'config.json')['rms_norm_eps']
    record=dict(state=fitted['state'],mean=fitted['mean'],scale=fitted['scale'],width=768,kind='sparse',layers=sm['layers'][:5],
        rms_eps=rms_eps,model=dict(files=sm['model_hashes']),source_fit_sha256=sha256(source/'fits/42/models.pt'))
    save_tensor(out/'filter.pt',record)
    load_filter(out/'filter.pt',sha256(out/'filter.pt'),layers=sm['layers'][:5],eps=rms_eps,model_path=sm['model_path'])
    if model_identity(sm['model_path'])!=sm['model_hashes']:raise ValueError('Model mismatch')
    write_json(out/'manifest.json',dict(protocol='slice-transfer-ppo-v1',runtime=snapshot(ROOT)['files'],tools=tools(),
        model_path=sm['model_path'],model_hashes=sm['model_hashes'],rms_eps=rms_eps,layers=sm['layers'][:5],filter_sha256=sha256(out/'filter.pt'),
        data_hashes={p.name:sha256(p) for p in (out/'data').iterdir()},arms=list(ARMS),seeds=list(SEEDS),epsilons=list(EPS),steps=60,pilot_steps=20,
        source=dict(path=str(source),fit_seed=42,fit_sha256=sha256(source/'fits/42/models.pt')),
        scope='Frozen source SLICE vs frozen RMS two-thirds, cumulative fresh Ridge. MATH numeric subset, inspected data. 2048-token PPO cap differs from 1024-token offline generation. No source head or matrices loaded.'))


def command(out,m,arm,seed,steps,epsilon,calibration,case):
    cmd=runner.command(out,'ridge',seed,steps,feature_mode='frozen',evaluation_split='val' if calibration else 'test',case_name=case)
    cmd += ['experiment.critic_statistics=cumulative','experiment.critic_folds=1',f'experiment.cumulative_epsilon={epsilon}',
        'experiment.learning_curve=true','actor_rollout_ref.model.critic_encoder_precision=fp32',f'actor_rollout_ref.model.critic_bottleneck_rms_eps={m["rms_eps"]}',
        f'actor_rollout_ref.model.path={m["model_path"]}',f'trainer.test_freq={1 if steps==2 else 10}']
    if arm=='slice':cmd += ['actor_rollout_ref.model.critic_feature_layer=bottleneck',f'actor_rollout_ref.model.critic_bottleneck_path={out/"filter.pt"}',
        f'actor_rollout_ref.model.critic_bottleneck_sha256={m["filter_sha256"]}',f'actor_rollout_ref.model.critic_bottleneck_layers={json.dumps(m["layers"],separators=(",",":"))}']
    else:cmd += ['actor_rollout_ref.model.critic_feature_layer=two_thirds','actor_rollout_ref.model.critic_feature_normalization=rms']
    return cmd


def audit(case,steps,epsilon,arm,split):
    r=read(case/'result.json')
    if r['method']!='ridge' or r['updates']!=steps or r['answers']!=64*steps or r['evaluation_split']!=split:raise ValueError('Unequal run budget')
    cfg=read(case/'run.json')['config']
    expected_norm='rms' if arm=='rms_two_thirds' else 'none'
    if cfg['actor_rollout_ref']['model']['critic_feature_normalization']!=expected_norm:raise ValueError('Wrong feature normalization')
    stats=torch.load(case/'linear-critic/statistics.pt',map_location='cpu',weights_only=True)
    k=769 if arm=='slice' else 1537
    if stats['step']!=steps or stats['epsilon']!=epsilon or stats['method']!='ridge' or len(stats['stats'])!=1 or stats['folds']!=1:raise ValueError('Wrong cumulative critic')
    s=stats['stats'][0]
    if stats['epoch']!=0 or set(stats['assignment'].values())!={0}:raise ValueError('Feature epoch or shared head changed')
    if s['answers']!=64*steps or s['a'].shape!=(k,k) or not torch.isfinite(s['a']).all() or not torch.isfinite(s['b']).all():raise ValueError('Invalid statistics')
    head=torch.load(case/'linear-critic'/f'{steps}.pt',map_location='cpu',weights_only=True)['heads'][0]['weights'].double()
    system=s['a'].double()+epsilon*torch.eye(k,dtype=torch.float64)
    residual=torch.linalg.vector_norm(system@head-s['b'].double())
    scale=(torch.linalg.matrix_norm(system)*torch.linalg.vector_norm(head)+torch.linalg.vector_norm(s['b'])).clamp_min(1e-30)
    if not torch.isfinite(head).all() or residual/scale>1e-8:raise ValueError('Head does not solve cumulative system')
    metrics=[json.loads(v) for v in (case/'linear-critic/metrics.jsonl').read_text().splitlines()]
    if [v['step'] for v in metrics]!=list(range(1,steps+1)) or any(v['feature_epoch']!=0 or v['reset']!=(i==0) for i,v in enumerate(metrics)):raise ValueError('Reset or missing statistics')
    for step in (range(1,steps+1) if steps==2 else range(10,steps+1,10)):
        c=read(case/f'curve-{step}.json')
        if c['step']!=step or c['answers']!=step*64 or c['response_tokens']!=sum(v['transitions'] for v in metrics[:step]):raise ValueError('Invalid curve budget')
    return r


def execute(out,m,arm,seed,steps,eps,calibration,case):
    folder=out/case
    if (folder/'result.json').exists():
        r=audit(folder,steps,eps,arm,'val' if calibration else 'test')
        if r['seed']!=seed:raise ValueError('Wrong actor seed')
        return r
    if folder.exists():raise ValueError('Partial actor run; no resume. Preserve it and use a new study.')
    with tempfile.TemporaryDirectory(prefix='tr-',dir='/tmp') as raytmp:
        env=dict(os.environ,RAY_ADDRESS='local',RAY_TMPDIR=raytmp,MATH_RL_SEED=str(seed));env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',None)
        with (out/(case+'.log')).open('w') as log:subprocess.run(command(out,m,arm,seed,steps,eps,calibration,case),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    r=audit(folder,steps,eps,arm,'val' if calibration else 'test')
    if r['seed']!=seed:raise ValueError('Wrong actor seed')
    return r


def gate(out,m):
    for arm in ARMS:audit(out/f'smoke-{arm}',2,.01,arm,'val')
    write_json(out/'smoke.json',dict(passed=True,manifest=sha256(out/'manifest.json')))


def require_gate(out):
    r=read(out/'smoke.json')
    if not r['passed'] or r['manifest']!=sha256(out/'manifest.json'):raise ValueError('Smoke must pass')


def select(out,m):
    require_gate(out);selected={};candidates={}
    for arm in ARMS:
        values=[]
        for eps in EPS:
            case=out/f'pilot-{arm}-{eps}';r=audit(case,20,eps,arm,'val')
            ys=[r['initial_validation']['accuracy']]+[read(case/f'curve-{s}.json')['accuracy'] for s in (10,20)]
            auc=float(_trapezoid(ys,[0,10,20])/20);values.append((auc,eps));candidates[f'{arm}/{eps}']=dict(validation_auc=auc,final_accuracy=r['final_validation']['accuracy'],training_gpu_hours=r['training_gpu_hours'])
        # Deterministic ties favor smaller epsilon; no final test data read.
        selected[arm]=max(values,key=lambda x:(x[0],-x[1]))[1]
    write_json(out/'selection.json',dict(epsilons=selected,upper_boundary_selected={a:e==max(EPS) for a,e in selected.items()},candidates=candidates,manifest=sha256(out/'manifest.json'),scope='One pilot actor seed 601; validation-only accuracy AUC over 20 updates, grid [.01,100,1000,10000]. Limited online tuning, no claim of global optimum.'))


def report(out,m):
    results={};arrays={};areas={};base_scores=None
    for arm in ARMS:
        runs=[];scores=[];aucs=[]
        for seed in SEEDS:
            eps=read(out/'selection.json')['epsilons'][arm];case=out/f'{arm}-seed{seed}';r=audit(case,60,eps,arm,'test')
            initial=runner.read_answers(case/'base-test.jsonl');final=runner.read_answers(case/'final-test.jsonl')
            expected=[v['id'] for v in read(out/'data/questions.json')['questions'] if v['split']=='test']
            if [v['id'] for v in final]!=expected or [v['id'] for v in initial]!=expected:raise ValueError('Unpaired test questions')
            b=[v['score'] for v in initial]
            if base_scores is None:base_scores=b
            if b!=base_scores:raise ValueError('Different starting policies/evaluations')
            curves=[dict(step=0,accuracy=r['initial_test']['accuracy'],response_tokens=0,online_seconds=0)]+[read(case/f'curve-{s}.json') for s in range(10,61,10)]
            values=[]
            for s in range(10,61,10):
                rows=runner.read_answers(case/f'curve-{s}-test.jsonl')
                if [v['id'] for v in rows]!=expected:raise ValueError('Curve test mismatch')
                values.append([v['score'] for v in rows])
            # Per-question predeclared AUC gives a paired statistic without selecting a checkpoint.
            qauc=_trapezoid(np.array([b]+values),[0,10,20,30,40,50,60],axis=0)/60
            if not np.isclose(np.mean(values[-1]),r['final_test']['accuracy']):raise ValueError('Final curve/result mismatch')
            runs.append(dict(seed=seed,result=r,curve=curves,accounted_gpu_hours=2*(r['initialization_seconds']+r['initial_test']['seconds']+r['training_seconds']+r['final_test']['seconds'])/3600))
            scores.append([v['score'] for v in final]);aucs.append(qauc)
        arrays[arm]=np.array(scores);areas[arm]=np.array(aucs)
        results[arm]=dict(final_accuracy=float(np.mean(scores)),accuracy_seed_sd=float(np.std(np.mean(scores,axis=1),ddof=1)),
            accuracy_auc=float(np.mean(aucs)),training_gpu_hours_mean=float(np.mean([r['result']['training_gpu_hours'] for r in runs])),
            accounted_gpu_hours_mean=float(np.mean([r['accounted_gpu_hours'] for r in runs])),runs=runs)
    paired={};rng=np.random.default_rng(42)
    for name,scores in [('final_accuracy',arrays),('accuracy_auc',areas)]:
        d=(scores['slice']-scores['rms_two_thirds'])*100
        draws=np.array([d[rng.integers(5,size=5)][:,rng.integers(d.shape[1],size=d.shape[1])].mean() for _ in range(4000)])
        paired[name]=dict(mean_pp=float(d.mean()),descriptive_95_interval_pp=np.quantile(draws,[.025,.975]).tolist(),adjusted_interval_pp=np.quantile(draws,[.0125,.9875]).tolist())
    write_json(out/'summary.json',dict(methods=results,paired=paired,selection=read(out/'selection.json'),scope=m['scope'],
        limitations='Five actor seeds, one source compressor. Two planned paired contrasts; approximate Bonferroni nominal intervals. Test checkpoints predeclared, no test-based selection. All monitoring overhead included in online time. Offline source learning and pilots are additional; generation/validation questions reused and inspected. No guaranteed equivalence or superiority.'))
    (out/'report.txt').write_text('\n'.join(['Transferred-feature PPO: fresh cumulative Ridge, 60 updates x 64 answers, five seeds.',*[f'{a}: accuracy={r["final_accuracy"]:.4f}; AUC={r["accuracy_auc"]:.4f}; train GPUh={r["training_gpu_hours_mean"]:.3f}; accounted GPUh={r["accounted_gpu_hours_mean"]:.3f}' for a,r in results.items()],*[f'slice minus layer / {k}: {v}' for k,v in paired.items()],'Curves vs answers, response tokens and inclusive online GPU hours saved in summary.json. Prior source feature training and pilot costs are separate.'])+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['prepare','smoke','gate','pilot','select','run','report']);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--task',type=int,default=0)
    a=p.parse_args();out=a.out.absolute();torch.set_num_threads(8)
    if a.stage=='prepare':
        if a.source is None:p.error('--source required')
        prepare(a.source.absolute(),out);return
    m=validate(out)
    if a.stage=='smoke':execute(out,m,ARMS[a.task],19,2,.01,True,'smoke-'+ARMS[a.task])
    elif a.stage=='gate':gate(out,m)
    elif a.stage=='pilot':
        require_gate(out);arm,eps=[(a,e) for a in ARMS for e in EPS][a.task];execute(out,m,arm,601,20,eps,True,f'pilot-{arm}-{eps}')
    elif a.stage=='select':select(out,m)
    elif a.stage=='run':
        require_gate(out);sel=read(out/'selection.json')
        if sel['manifest']!=sha256(out/'manifest.json'):raise ValueError('Selection mismatch')
        arm,seed=[(a,s) for s in SEEDS for a in (ARMS if s%2 else ARMS[::-1])][a.task]
        execute(out,m,arm,seed,60,sel['epsilons'][arm],False,f'{arm}-seed{seed}')
    else:report(out,m)
if __name__=='__main__':main()
