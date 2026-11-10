"""One frozen policy, shared rollout labels, six critic/baseline estimators.

This is offline policy evaluation. No actor optimizer, GAE or PPO actor update.
GPU extraction/generation and the FSDP transformer fit are separate entry points.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .provenance import sha256, snapshot, write_json
from .runner import ROOT
from .value_audit import read, stable_seed, summarize, wilson, boundary_positions, require_gate
from slice_critic.storage import model_identity

ARMS = ['slice-token','rms-layer-token','slice-chunk','rms-layer-chunk','transformer','grpo']
TITLES = ['SLICE + Ridge (token)','RMS layer + Ridge (token)',
          'SLICE + Ridge (chunk)','RMS layer + Ridge (chunk)',
          'Transformer critic (MC fit)','GRPO question baseline (no critic)']
RECIPES = {
    'smoke': dict(train=8,val=4,test=4,answers=4,samples=4,epochs=1),
    'main': dict(train=500,val=100,test=50,answers=4,samples=128,epochs=2),
}
SHARDS = 4


def prepare(source, out, phase):
    # Deliberately reuse one predeclared actor, not the actor with the best score.
    require_gate(source, 'pilot/verification.json')
    old = read(source/'manifest.json')
    if old['protocol'] != 'slice-value-audit-v1':
        raise ValueError('Source must be a completed six-actor value audit')
    from .value_audit import audit_run
    identity = audit_run(source, old, 'slice-token')
    if identity != read(source/'train-training.json')['runs']['slice-token']:
        raise ValueError('Source actor changed since its training gate')
    actor = source/f'slice-token-seed{old["actor_seed"]}/global_step_{old["steps"]}/actor/huggingface'
    if model_identity(ROOT/'models/qwen-math') != old['model_hashes']:
        raise ValueError('Frozen base encoder changed')
    for name, digest in old['data_hashes'].items():
        if sha256(source/name) != digest:
            raise ValueError('Source question data changed')
    if sha256(source/'filters/sparse.pt') != old['filter_sha256']:
        raise ValueError('Source compressor changed')
    recipe = RECIPES[phase]
    questions = read(source/'data/questions.json')['questions']
    selected = []
    for split in ('train','val','test'):
        rows = sorted((q for q in questions if q['split']==split),
                      key=lambda q: stable_seed(92017,'questions',q['id']))
        if len(rows) < recipe[split]: raise ValueError(f'Insufficient {split} questions')
        selected.extend(rows[:recipe[split]])
    if len({q['id'] for q in selected}) != len(selected):
        raise ValueError('Train/validation/test question overlap')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source/'filters/sparse.pt',out/'sparse.pt')
    write_json(out/'questions.json',dict(questions=selected))
    manifest = dict(protocol='fixed-policy-critics-v1',phase=phase,recipe=recipe,arms=ARMS,
        actor_path=str(actor),actor_identity=model_identity(actor),base_identity=old['model_hashes'],
        source_manifest=sha256(source/'manifest.json'),source_arm='slice-token',
        filter_sha256=old['filter_sha256'],layers=old['layers'],rms_eps=old['rms_eps'],
        questions_sha256=sha256(out/'questions.json'),epsilon=.01,seed=92017,
        response_cap=2048,max_model_len=2560,reference_seed=92017,reference_batch=16,
        sampling=dict(temperature=1.,top_p=1.,top_k=-1),prefixes=4,shards=SHARDS,
        neural=dict(lr=5e-5,epochs=recipe['epochs'],answer_batch=16,seed=1001),
        runtime=snapshot(ROOT)['files'],
        scope='One frozen, predeclared SLICE/token actor. Shared new rollout data and one MC reference bank. '
              'Offline critic fitting, not PPO or actor improvement. Reused inspected Numina data and one compressor.')
    write_json(out/'manifest.json',manifest)


def validate(out):
    m = read(out/'manifest.json')
    if (m['protocol']!='fixed-policy-critics-v1' or m['arms']!=ARMS
            or m['recipe']!=RECIPES[m['phase']] or m['epsilon']!=.01 or m['shards']!=SHARDS
            or m['seed']!=92017 or m['response_cap']!=2048 or m['max_model_len']!=2560
            or m['prefixes']!=4 or m['reference_batch']!=16 or m['reference_seed']!=92017
            or m['sampling']!=dict(temperature=1.,top_p=1.,top_k=-1)
            or m['neural']!=dict(lr=5e-5,epochs=m['recipe']['epochs'],answer_batch=16,seed=1001)
            or m['runtime']!=snapshot(ROOT)['files']):
        raise ValueError('Fixed-policy protocol/runtime changed')
    if (sha256(out/'questions.json')!=m['questions_sha256'] or sha256(out/'sparse.pt')!=m['filter_sha256']
            or model_identity(Path(m['actor_path']))!=m['actor_identity']
            or model_identity(ROOT/'models/qwen-math')!=m['base_identity']):
        raise ValueError('Actor, data or frozen basis changed')
    return m


def items(out, m):
    result=[]
    for q in read(out/'questions.json')['questions']:
        for answer in range(1 if q['split']=='test' else m['recipe']['answers']):
            key=f'{stable_seed(q["id"]):010d}-{answer}'
            result.append(dict(q,key=key,answer_index=answer))
    if len({r['key'] for r in result}) != len(result): raise ValueError('Answer key collision')
    return result


def answer(out, row):
    data=read(out/'answers'/f'{row["key"]}.json')
    if data['manifest_sha256']!=sha256(out/'manifest.json') or data['key']!=row['key']:
        raise ValueError('Stale/misassigned rollout')
    return data


def features(out, row):
    path=out/'features'/f'{row["key"]}.pt'
    sidecar=read(path.with_suffix('.json'))
    if sidecar!=dict(manifest_sha256=sha256(out/'manifest.json'),answer_sha256=sha256(out/'answers'/f'{row["key"]}.json'),sha256=sha256(path)):
        raise ValueError('Feature identity changed')
    return torch.load(path,map_location='cpu',weights_only=True)


def design(x):
    x=torch.as_tensor(x).double()
    return torch.cat([x,torch.ones(len(x),1,dtype=x.dtype)],1)


def linear_fit(out,m):
    systems={}
    for row in items(out,m):
        if row['split']!='train':continue
        data=answer(out,row);f=features(out,row)
        for arm in ARMS[:4]:
            x=f['slice' if arm.startswith('slice') else 'rms']
            if arm.endswith('chunk'):x=x[f['starts']]
            x=design(x)
            if arm not in systems:systems[arm]=[torch.zeros(x.shape[1],x.shape[1],dtype=torch.float64),torch.zeros(x.shape[1],dtype=torch.float64),0]
            a,b,count=systems[arm]
            a+=x.T@x;b+=x.sum(0)*data['score'];systems[arm][2]+=len(x)
    heads={}
    for arm,(a,b,count) in systems.items():
        system=a+m['epsilon']*torch.eye(len(b),dtype=a.dtype)
        w=torch.linalg.solve(system,b)
        residual=(system@w-b).norm()/(system.norm()*w.norm()+b.norm()).clamp_min(1e-30)
        if not torch.isfinite(w).all() or residual>1e-8:raise ValueError('Invalid Ridge solve')
        heads[arm]=dict(weights=w,states=count,residual=float(residual))
    if set(heads)!=set(ARMS[:4]):raise ValueError('Missing Ridge arms')
    torch.save(heads,out/'ridge.pt')
    torch.save(dict(systems=systems,epsilon=m['epsilon']),out/'ridge-statistics.pt')
    write_json(out/'linear.json',dict(manifest_sha256=sha256(out/'manifest.json'),sha256=sha256(out/'ridge.pt'),
        heads={a:dict(states=h['states'],residual=h['residual']) for a,h in heads.items()}))


def pack(out,m):
    states=[]
    for row in items(out,m):
        if row['split']!='test':continue
        data=answer(out,row);f=features(out,row)
        for i,pos in enumerate(boundary_positions(f['starts'],m['prefixes'])):
            states.append(dict(state_id=f'{row["id"]}/{i}',question_id=row['id'],key=row['key'],
                prompt_ids=data['prompt_ids'],prefix_ids=data['token_ids'][:pos],position=pos,
                ground_truth=row['ground_truth'],remaining_tokens=m['response_cap']-pos,
                progress=pos/len(data['token_ids'])))
    write_json(out/'bank.json',dict(manifest_sha256=sha256(out/'manifest.json'),states=states))


def output(out, states, **extra):
    return dict(manifest_sha256=sha256(out/'manifest.json'),bank_sha256=sha256(out/'bank.json'),states=states,**extra)


def predict_linear(out,m,arm):
    saved=read(out/'linear.json')
    if saved['manifest_sha256']!=sha256(out/'manifest.json') or saved['sha256']!=sha256(out/'ridge.pt'):
        raise ValueError('Ridge fit changed')
    w=torch.load(out/'ridge.pt',map_location='cpu',weights_only=True)[arm]['weights']
    by_key={r['key']:r for r in items(out,m)};cache={};rows=[]
    for state in read(out/'bank.json')['states']:
        if state['key'] not in cache:cache[state['key']]=features(out,by_key[state['key']])
        x=cache[state['key']]['slice' if arm.startswith('slice') else 'rms'][state['position']:state['position']+1]
        rows.append(dict(state_id=state['state_id'],prediction=float((design(x)@w).item())))
    write_json(out/f'predictions-{arm}.json',output(out,rows))


def paired(rows, arms=ARMS):
    # The SAME reference is used in every difference; resample questions, not dots.
    grouped={}
    for r in rows:
        ref=r['successes']/r['samples']
        grouped.setdefault(r['question_id'],[]).append([abs(r['predictions'][a]-ref) for a in arms])
    x=np.array([np.mean(v,axis=0) for v in grouped.values()]);rng=np.random.default_rng(92017)
    result={}
    for a,b in [('slice-token','rms-layer-token'),('slice-chunk','rms-layer-chunk'),
                ('slice-chunk','slice-token'),('transformer','slice-token'),('grpo','slice-token')]:
        d=x[:,arms.index(a)]-x[:,arms.index(b)]
        draws=[d[rng.integers(len(d),size=len(d))].mean() for _ in range(2000)]
        result[f'{a}_minus_{b}']=dict(mean=float(d.mean()),question_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),
            scope='Exploratory, unadjusted; conditional on shared MC references; one actor and one transformer fit seed.')
    return result


def report(out,m):
    bank=read(out/'bank.json')['states'];wanted={s['state_id'] for s in bank}
    refs=[]
    for task in range(SHARDS):
        r=read(out/f'reference-{task}.json')
        check_output(out,r);refs.extend(r['states'])
    reference={r['state_id']:r for r in refs}
    if len(refs)!=len(reference) or set(reference)!=wanted:raise ValueError('Incomplete/duplicate common references')
    predictions={}
    for arm in ARMS:
        r=read(out/f'predictions-{arm}.json');check_output(out,r)
        p={s['state_id']:s['prediction'] for s in r['states']}
        if len(p)!=len(r['states']) or set(p)!=wanted:raise ValueError('Incomplete predictions')
        predictions[arm]=p
    rows=[];panels={a:[] for a in ARMS};metrics={};clipped={}
    for s in bank:
        ref=reference[s['state_id']]
        if ref['samples']!=m['recipe']['samples']:raise ValueError('Reference sample budget mismatch')
        row=dict(s,**{k:ref[k] for k in ('samples','successes')},predictions={a:predictions[a][s['state_id']] for a in ARMS})
        rows.append(row)
        for arm in ARMS:
            panels[arm].append(dict(s,samples=ref['samples'],successes=ref['successes'],
                reference=ref['successes']/ref['samples'],prediction=row['predictions'][arm],
                reference_wilson_95=wilson(ref['successes'],ref['samples'])))
    for arm in ARMS:
        metrics[arm]=summarize(panels[arm])
        clipped[arm]=summarize([dict(r,prediction=float(np.clip(r['prediction'],0,1))) for r in panels[arm]])
    record=dict(protocol=m['protocol'],metrics=metrics,secondary_clipped_metrics=clipped,
                paired_mae=paired(rows),panels=panels,scope=m['scope'],
                note='Identical reference x-coordinates in all panels. MC-trained transformer, not a PPO actor run. '
                     'Question-bootstrap differences do not include actor/fit-seed or reference uncertainty.')
    write_json(out/'summary.json',record);plot(out,panels,metrics)
    lines=['One frozen policy; shared rollouts and MC references; offline critic comparison.',
           'arm                       MAE      RMSE     outside [0,1]']
    lines += [f'{a:<24} {v["mae"]:.5f}  {v["rmse"]:.5f}  {100*v["outside_unit_interval"]:.2f}%' for a,v in metrics.items()]
    lines += ['Transformer: supervised MC-return fitting; no GAE, no actor optimization.',m['scope'],
              'Clipping diagnostics and paired question-bootstrap contrasts are secondary/exploratory; see summary.json.']
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    files=['bank.json','ridge.pt','ridge-statistics.pt','linear.json','neural.json','summary.json','report.txt','value-scatter.png','value-scatter.pdf','error-by-progress.png','progress-counts.json']
    files += [f'predictions-{a}.json' for a in ARMS]+[f'reference-{i}.json' for i in range(SHARDS)]
    write_json(out/'verification.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),
        files={n:sha256(out/n) for n in files},scope='All six estimators use the same frozen actor and reference outcomes; not a method ranking gate.'))


def check_output(out,record):
    if (record['manifest_sha256']!=sha256(out/'manifest.json') or record['bank_sha256']!=sha256(out/'bank.json')):
        raise ValueError('Output identity mismatch')


def plot(out,panels,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=[r['prediction'] for rows in panels.values() for r in rows]
    lo,hi=min(-.05,min(values)-.03),max(1.05,max(values)+.03)
    colors=['#6a3d8f','#2374ab','#9670ae','#64a9ce','#e69f00','#009e73']
    fig,axes=plt.subplots(2,3,figsize=(14,9),sharex=True,sharey=True,layout='constrained')
    for ax,arm,title,color in zip(axes.flat,ARMS,TITLES,colors):
        rows=panels[arm];ax.plot([0,1],[0,1],'k--',lw=1)
        ax.scatter([r['reference'] for r in rows],[r['prediction'] for r in rows],color=color,alpha=.4,s=24)
        ax.set(title=title,xlim=(-.05,1.05),ylim=(lo,hi));ax.grid(alpha=.25)
        ax.text(.04,.94,f'MAE {metrics[arm]["mae"]:.3f}',transform=ax.transAxes,va='top')
    fig.suptitle('One frozen actor • shared training answers • identical MC reference values')
    fig.supxlabel('Shared Monte Carlo reference value');fig.supylabel('Prediction / GRPO baseline (unclipped)')
    fig.savefig(out/'value-scatter.png',dpi=180);fig.savefig(out/'value-scatter.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    counts={}
    for arm in ARMS:
        points=[]
        for index in range(5):
            chosen=[r for r in panels[arm] if min(int(r['progress']*5),4)==index]
            groups={}
            for r in chosen:groups.setdefault(r['question_id'],[]).append(abs(r['prediction']-r['reference']))
            if groups:points.append((index*.2+.1,float(np.mean([np.mean(v) for v in groups.values()]))))
            counts[str(index)]=dict(states=len(chosen),questions=len(groups))
        if points:ax.plot(*zip(*points),marker='o',label=arm)
    ax.set(xlabel='Relative sampled-answer token progress',ylabel='Equal-question MAE',title='Shared-policy value error by progress')
    ax.legend(fontsize=8);ax.grid(alpha=.25);fig.savefig(out/'error-by-progress.png',dpi=180);plt.close(fig)
    write_json(out/'progress-counts.json',counts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['prepare','generate','extract','linear','bank','neural','predict','mc','report'])
    p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=list(RECIPES),default='smoke');p.add_argument('--task',type=int,default=0)
    a=p.parse_args();out=a.out.resolve()
    if a.stage=='prepare':
        if a.source is None:p.error('prepare requires source')
        prepare(a.source.resolve(),out,a.phase);return
    m=validate(out);started=time.perf_counter()
    if m['phase']!=a.phase:raise ValueError('Wrong sealed phase')
    try:
        if a.stage=='linear':linear_fit(out,m)
        elif a.stage=='bank':pack(out,m)
        elif a.stage=='report':report(out,m)
        elif a.stage=='neural':
            from .fixed_policy_neural import train
            train(out,m)
        elif a.stage=='predict' and a.task<4:predict_linear(out,m,ARMS[a.task])
        else:
            from . import fixed_policy_gpu as gpu
            getattr(gpu,a.stage)(out,m,a.task)
        write_json(out/'status'/f'{a.stage}-{a.task}.json',dict(state='complete',seconds=time.perf_counter()-started))
    except Exception as exc:
        write_json(out/'status'/f'{a.stage}-{a.task}.json',dict(state='failed',error=str(exc)));raise


if __name__=='__main__':main()
