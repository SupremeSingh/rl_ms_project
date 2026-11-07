"""Locked 2x2 Ridge study: representation x token/chunk time scale, five actors."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from . import runner
from .provenance import snapshot, sha256, write_json
from .reproduction import validate_questions
from slice_critic.storage import model_identity
from slice_critic.inference import load_filter

ROOT=runner.ROOT
ARMS={
    'slice-token':dict(layer='bottleneck',mode='token'),
    'slice-chunk':dict(layer='bottleneck',mode='math'),
    'layer-token':dict(layer='two_thirds',mode='token'),
    'layer-chunk':dict(layer='two_thirds',mode='math'),
}
SEEDS=[901,902,903,904,905]


def read(path): return json.loads(Path(path).read_text())


def prepare(source,out,fallback_tokens=64):
    from datasets import Dataset
    if type(fallback_tokens) is not int or fallback_tokens < 1:
        raise ValueError('chunk_fallback_tokens must be a positive integer')
    source=source.absolute();out=out.absolute();old=read(source/'manifest.json')
    if out==source or source in out.parents: raise ValueError('Output must be separate from source')
    for name,h in old['data_hashes'].items():
        if sha256(source/'data'/name)!=h: raise ValueError('Source data changed')
    questions=read(source/'data/questions.json')['questions']
    validate_questions(questions)
    counts=Counter(q['split'] for q in questions)
    if counts['train']<2000 or counts['val']<100 or counts['test']!=2000:
        raise ValueError('Require full study data: >=2000 train, >=100 val, exactly 2000 test; not smoke data')
    hashes=model_identity(ROOT/'models/qwen-math')
    if hashes!=old['model_hashes']: raise ValueError('Actor/frozen encoder differs from source')
    saved=old['filters']['sparse']
    filter_path=source/'filters'/Path(saved['path']).name
    if not filter_path.is_file(): filter_path=Path(saved['path'])
    if sha256(filter_path)!=saved['sha256']: raise ValueError('Source filter checksum mismatch')
    layers=old['layers'];eps=old['rms_eps']
    if layers!=[27,26,24,20,12]: raise ValueError('Expected final five-block SLICE')
    load_filter(filter_path,saved['sha256'],layers=layers,eps=eps,model_path=ROOT/'models/qwen-math')
    tensors=torch.load(filter_path,map_location='cpu',weights_only=True)
    blocked=set(tensors.get('representation_training_families',[]))|set(tensors.get('selection_families',[]))
    validate_questions(questions,blocked)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'data',out/'data');(out/'filters').mkdir()
    shutil.copyfile(filter_path,out/'filters/sparse.pt')
    (out/'smoke-data').mkdir()
    for split,n in [('train',64),('val',16),('test',16)]:
        dataset=Dataset.from_parquet(str(out/f'data/{split}.parquet'),cache_dir=str(out/'.prepare-cache'))
        expected=[q for q in questions if q['split']==split]
        if [(str(row['extra_info']['prompt_id']),str(row['reward_model']['ground_truth'])) for row in dataset] != [(str(q['id']),str(q['ground_truth'])) for q in expected]:
            raise ValueError('Parquet question identities/references differ from JSON')
        dataset.select(range(min(n,len(dataset)))).to_parquet(str(out/f'smoke-data/{split}.parquet'))
    manifest=dict(protocol='slice-chunk-ablation-v1',arms=ARMS,seeds=SEEDS,steps=120,
        chunk_fallback='fixed_tokens',chunk_fallback_tokens=fallback_tokens,
        answers_per_update=64,gamma=1,gae_lambda=.95,epsilon=.01,critic_folds=1,
        primary=['slice-chunk_minus_slice-token','layer-chunk_minus_layer-token'],
        counts=dict(counts),layers=layers,rms_eps=eps,model_hashes=hashes,
        filter_sha256=sha256(out/'filters/sparse.pt'),
        data_hashes={str(p.relative_to(out)):sha256(p) for directory in ('data','smoke-data') for p in (out/directory).iterdir()},
        source=dict(path=str(source),manifest_sha256=sha256(source/'manifest.json'),
            filter_sha256=saved['sha256'],has_explicit_family_exclusions=bool(blocked)),
        runtime=snapshot(ROOT)['files'],
        scope='Fresh actors; reused inspected Numina questions and one existing full-study compressor. No retraining or tuning. One shared head in every arm.')
    write_json(out/'manifest.json',manifest)


def validate(out):
    m=read(out/'manifest.json')
    if m['protocol']!='slice-chunk-ablation-v1' or m['arms']!=ARMS or m['seeds']!=SEEDS or m['steps']!=120:
        raise ValueError('Study protocol changed')
    if (m.get('chunk_fallback')!='fixed_tokens' or type(m.get('chunk_fallback_tokens')) is not int
            or m['chunk_fallback_tokens'] < 1):
        raise ValueError('Prepare a new study with an explicit fixed-token fallback')
    if snapshot(ROOT)['files']!=m['runtime']: raise ValueError('Code changed: prepare a new study')
    for name,h in m['data_hashes'].items():
        if sha256(out/name)!=h: raise ValueError('Study data changed')
    if sha256(out/'filters/sparse.pt')!=m['filter_sha256'] or model_identity(ROOT/'models/qwen-math')!=m['model_hashes']:
        raise ValueError('Frozen model/filter changed')
    return m


def command(out,m,arm,seed,smoke=False):
    setting=m['arms'][arm];case=('smoke-' if smoke else '')+f'{arm}-seed{seed}'
    cmd=runner.command(out,'ridge',seed,2 if smoke else m['steps'],feature_mode='frozen',
        evaluation_split='val' if smoke else 'test',case_name=case)
    cmd += [f'experiment.chunking={setting["mode"]}',f'actor_rollout_ref.model.critic_chunking={setting["mode"]}',
        f'experiment.chunk_fallback_tokens={m["chunk_fallback_tokens"]}',
        f'actor_rollout_ref.model.critic_chunk_fallback_tokens={m["chunk_fallback_tokens"]}',
        f'critic.chunk_fallback_tokens={m["chunk_fallback_tokens"]}',
        f'critic.chunking={setting["mode"]}','experiment.critic_folds=1','algorithm.lam=0.95',
        'experiment.critic_statistics=cumulative','experiment.cumulative_epsilon=0.01',
        'actor_rollout_ref.model.critic_encoder_precision=fp32',
        f'actor_rollout_ref.model.critic_feature_layer={setting["layer"]}',f'trainer.test_freq={2 if smoke else 30}']
    if setting['layer']=='bottleneck':
        cmd += [f'actor_rollout_ref.model.critic_bottleneck_path={out}/filters/sparse.pt',
            f'actor_rollout_ref.model.critic_bottleneck_sha256={m["filter_sha256"]}',
            f'actor_rollout_ref.model.critic_bottleneck_layers={json.dumps(m["layers"],separators=(",",":"))}',
            f'actor_rollout_ref.model.critic_bottleneck_rms_eps={m["rms_eps"]}']
    if smoke:
        cmd += [f'data.train_files={out}/smoke-data/train.parquet',f'data.val_files={out}/smoke-data/val.parquet',
                f'experiment.test_file={out}/smoke-data/val.parquet']
    return cmd,case


def audit(out,m,arm,seed,smoke=False):
    _,name=command(out,m,arm,seed,smoke);case=out/name;r=read(case/'result.json');n=2 if smoke else m['steps']
    setting=m['arms'][arm]
    if not np.isclose(r['training_gpu_hours'],2*r['training_seconds']/3600):
        raise ValueError('Training GPU-hour accounting mismatch')
    if (r['updates']!=n or r['answers']!=64*n or r['seed']!=seed or r['method']!='ridge'
        or r['chunking']!=setting['mode'] or r['chunk_fallback_tokens']!=m['chunk_fallback_tokens']
        or r['critic_folds']!=1 or r['gae_lambda']!=.95
        or r['evaluation_split']!=('val' if smoke else 'test')): raise ValueError('Unequal run protocol')
    cfg=read(case/'run.json')['config'];model=cfg['actor_rollout_ref']['model']
    if (model['critic_feature_layer']!=setting['layer'] or model['critic_feature_mode']!='frozen'
        or model['critic_encoder_precision']!='fp32' or cfg['experiment']['cumulative_epsilon']!=.01
        or any(v!=m['chunk_fallback_tokens'] for v in (model['critic_chunk_fallback_tokens'],
            cfg['experiment']['chunk_fallback_tokens'],cfg['critic']['chunk_fallback_tokens']))):
        raise ValueError('Feature/regularization settings differ')
    rows=[json.loads(v) for v in (case/'linear-critic/metrics.jsonl').read_text().splitlines()]
    if ([v['step'] for v in rows]!=list(range(1,n+1)) or any(v['reset']!=(i==0) or v['feature_epoch']!=0 for i,v in enumerate(rows))):
        raise ValueError('Missing/reset cumulative statistics')
    units=sum(v['transitions'] for v in rows);tokens=sum(v['response_tokens'] for v in rows)
    if units>tokens or (setting['mode']=='token' and units!=tokens): raise ValueError('Invalid state counts')
    fallback_answers=0
    if setting['mode']=='math':
        chunk=[json.loads(v) for v in (case/'chunk-metrics.jsonl').read_text().splitlines()]
        if ([v['step'] for v in chunk]!=list(range(1,n+1)) or sum(v['chunks'] for v in chunk)!=units
            or any(v['gae_lambda']!=.95 or v['chunk_fallback_tokens']!=m['chunk_fallback_tokens'] for v in chunk)):
            raise ValueError('Chunk and fit accounting differ')
        fallback_answers=sum(v['segmentation_fallback_answers'] for v in chunk)
        if not 0 <= fallback_answers <= 64*n: raise ValueError('Invalid segmentation fallback count')
    checkpoint=torch.load(case/'linear-critic/statistics.pt',map_location='cpu',weights_only=True)
    k=769 if setting['layer']=='bottleneck' else 1537
    if checkpoint['method']!='ridge' or checkpoint['step']!=n or checkpoint['epsilon']!=.01 or checkpoint['folds']!=1 or len(checkpoint['stats'])!=1:
        raise ValueError('Wrong cumulative matrix settings')
    state=checkpoint['stats'][0]
    if state['answers']!=64*n or state['transitions']!=units or state['a'].shape!=(k,k): raise ValueError('Wrong cumulative counts/dimensions')
    if not torch.isfinite(state['a']).all() or not torch.isfinite(state['b']).all(): raise ValueError('Nonfinite matrices')
    saved_heads=torch.load(case/f'linear-critic/{n}.pt',map_location='cpu',weights_only=True)['heads']
    if len(saved_heads)!=1: raise ValueError('Expected exactly one saved value head')
    weights=saved_heads[0]['weights'].double()
    system=state['a'].double()+.01*torch.eye(k,dtype=torch.float64);b=state['b'].double()
    residual=torch.linalg.vector_norm(system@weights-b)/(torch.linalg.matrix_norm(system)*weights.norm()+b.norm()).clamp_min(1e-30)
    if not torch.isfinite(weights).all() or residual>1e-8: raise ValueError('Bad head solve')
    return dict(result=r,states=units,response_tokens=tokens,segmentation_fallback_answers=fallback_answers,fit_seconds=sum(v['fit_seconds'] for v in rows),
        accounted_gpu_hours=2*(r['initialization_seconds']+r['training_seconds']+
            r['initial_validation' if smoke else 'initial_test']['seconds']+r['final_validation' if smoke else 'final_test']['seconds'])/3600)


def jobs(m):
    names=list(m['arms']);return [(arm,seed) for i,seed in enumerate(m['seeds']) for arm in names[i%4:]+names[:i%4]]


def execute(out,m,arm,seed,smoke=False):
    cmd,name=command(out,m,arm,seed,smoke);case=out/name
    if (case/'result.json').exists(): return audit(out,m,arm,seed,smoke)
    if case.exists(): raise ValueError('Partial actor run preserved: use a new output study or explicitly recover after inspection')
    env=dict(os.environ,RAY_ADDRESS='local',MATH_RL_SEED=str(seed));env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',None)
    with tempfile.TemporaryDirectory(prefix='ca-',dir='/tmp') as tmp:
        env['RAY_TMPDIR']=tmp
        with (out/f'{name}.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return audit(out,m,arm,seed,smoke)


def gate(out,m):
    for arm in m['arms']: audit(out,m,arm,19,True)
    write_json(out/'smoke.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),arms=list(m['arms']),
        scope='Four matched two-update checks on this study filter; no test-based selection'))


def require_gate(out):
    g=read(out/'smoke.json')
    if not g['passed'] or g['manifest_sha256']!=sha256(out/'manifest.json'): raise ValueError('Matching smoke gate required')


def paired_interval(d,tail=.025):
    rng=np.random.default_rng(813)
    draws=[d[np.ix_(rng.integers(d.shape[0],size=d.shape[0]),rng.integers(d.shape[1],size=d.shape[1]))].mean()*100 for _ in range(4000)]
    return np.quantile(draws,[tail,1-tail]).tolist()


def report(out,m):
    require_gate(out);arrays={};results={};base_ref=None
    expected=[(str(q['id']),str(q['ground_truth'])) for q in read(out/'data/questions.json')['questions'] if q['split']=='test']
    for arm in m['arms']:
        scores=[];runs=[]
        for seed in m['seeds']:
            run=audit(out,m,arm,seed);case=out/f'{arm}-seed{seed}';r=run['result']
            base=runner.read_answers(case/'base-test.jsonl');final=runner.read_answers(case/'final-test.jsonl')
            for batch in (base,final):
                if [(str(v['id']),str(v['ground_truth'])) for v in batch]!=expected: raise ValueError('Unpaired test identities')
            b=[v['score'] for v in base];a=[v['score'] for v in final]
            if base_ref is None: base_ref=b
            if b!=base_ref or not np.isclose(np.mean(a),r['final_test']['accuracy']): raise ValueError('Inconsistent baseline/results')
            run['verifier_statuses']=dict(Counter(v['verifier_status'] for v in final))
            run['test_truncation_rate']=float(np.mean([v['capped'] for v in final]))
            scores.append(a);runs.append(run)
        array=np.array(scores);arrays[arm]=array
        results[arm]=dict(accuracy_mean=float(array.mean()),accuracy_seed_sd=float(array.mean(1).std(ddof=1)),
            accuracy_by_seed=dict(zip(map(str,m['seeds']),array.mean(1).tolist())),
            training_gpu_hours_mean=float(np.mean([v['result']['training_gpu_hours'] for v in runs])),
            accounted_gpu_hours_mean=float(np.mean([v['accounted_gpu_hours'] for v in runs])),
            response_tokens_mean=float(np.mean([v['response_tokens'] for v in runs])),
            fitted_states_mean=float(np.mean([v['states'] for v in runs])),
            critic_fit_seconds_mean=float(np.mean([v['fit_seconds'] for v in runs])),
            segmentation_fallback_fraction=float(sum(v['segmentation_fallback_answers'] for v in runs)/(len(runs)*m['steps']*64)),runs=runs)
    primary={}
    for rep in ('slice','layer'):
        d=arrays[rep+'-chunk']-arrays[rep+'-token']
        primary[rep]=dict(mean_pp=float(d.mean()*100),seed_differences_pp=(d.mean(1)*100).tolist(),
            descriptive_95_interval_pp=paired_interval(d),adjusted_interval_pp=paired_interval(d,.0125),
            training_gpu_hours_change_percent=100*(results[rep+'-chunk']['training_gpu_hours_mean']/results[rep+'-token']['training_gpu_hours_mean']-1))
    interaction=(arrays['slice-chunk']-arrays['slice-token'])-(arrays['layer-chunk']-arrays['layer-token'])
    smoke_cost=sum(audit(out,m,a,19,True)['accounted_gpu_hours'] for a in m['arms'])
    summary=dict(protocol=m,methods=results,primary=primary,base_accuracy=float(np.mean(base_ref)),
        interaction_exploratory=dict(mean_pp=float(interaction.mean()*100),interval_95_pp=paired_interval(interaction)),
        smoke_accounted_gpu_hours=smoke_cost,
        limitations='Five fresh actor seeds but reused inspected questions and one frozen source compressor. Two primary accuracy contrasts use approximate paired seed/question bootstrap and Bonferroni nominal adjustment. No equivalence/superiority gate. GPU-hours are measured model-runtime phases times two GPUs, not utilization or a full Slurm allocation ledger; launcher/audit overhead and prior source training are additional. Chunking changes GAE horizon and state weighting; it is not an isolated segmentation-only intervention.')
    write_json(out/'summary.json',summary)
    lines=['Does reasoning-chunk learning improve cumulative Ridge PPO?',
        f'120 updates x 64 answers; 5 paired seeds; {len(expected)} greedy test questions; response cap 2048.',
        'Same frozen features, one head, epsilon .01, gamma 1, GAE .95 (per token or per chunk).',
        'arm                 actor accuracy   training GPUh   accounted GPUh   fitted states',
        *[f'{a:20} {v["accuracy_mean"]*100:6.2f}%          {v["training_gpu_hours_mean"]:.3f}          {v["accounted_gpu_hours_mean"]:.3f}          {v["fitted_states_mean"]:.0f}' for a,v in results.items()],
        *[f'{a}: segmentation fallback {v["segmentation_fallback_fraction"]:.2%} of training answers (fixed groups of at most {m["chunk_fallback_tokens"]} original tokens)' for a,v in results.items()],
        'Primary differences: chunk minus token; units are percentage points.',
        *[f'{rep}: {v}' for rep,v in primary.items()],f'Exploratory interaction: {summary["interaction_exploratory"]}',
        f'Additional smoke accounted GPUh: {smoke_cost:.3f}',summary['limitations']]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    with (out/'results.csv').open('w') as f:
        f.write('arm,seed,accuracy,training_gpu_hours,accounted_gpu_hours,response_tokens,fitted_states,critic_fit_seconds\n')
        for arm,r in results.items():
            for v in r['runs']:
                x=v['result'];f.write(f'{arm},{x["seed"]},{x["final_test"]["accuracy"]},{x["training_gpu_hours"]},{v["accounted_gpu_hours"]},{v["response_tokens"]},{v["states"]},{v["fit_seconds"]}\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=['prepare','smoke','gate','run','report'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--task',type=int,default=0)
    p.add_argument('--chunk-fallback-tokens',type=int,default=64,help='Prepare only: token group size when MATH segmentation fails')
    a=p.parse_args();out=a.out.absolute();torch.set_num_threads(8)
    if a.stage=='prepare':
        if a.source is None:p.error('prepare requires --source')
        prepare(a.source,out,a.chunk_fallback_tokens);return
    m=validate(out);status=out/f'{a.stage}-{a.task}-status.json';start=time.perf_counter()
    write_json(status,dict(state='running'))
    try:
        if a.stage=='smoke':execute(out,m,list(m['arms'])[a.task],19,True)
        elif a.stage=='run':
            require_gate(out);arm,seed=jobs(m)[a.task];execute(out,m,arm,seed)
        else:globals()[a.stage](out,m)
        write_json(status,dict(state='complete',seconds=time.perf_counter()-start))
    except BaseException as exc:write_json(status,dict(state='failed',error=str(exc)));raise


if __name__=='__main__':main()
