"""Multi-seed repeat of conventional PPO/GRPO and four cumulative Ridge actors.

Training delegates to the original value_audit commands. MC outcomes are ONLY
evaluation references. This module does not import the fixed-policy regression fit.
"""
import argparse
from pathlib import Path
import time

import numpy as np

from . import value_audit as base
from .provenance import sha256,snapshot,write_json
from .runner import ROOT
from slice_critic.storage import model_identity

SEEDS=[1101,1102,1103]
BUDGETS={'smoke':dict(questions=4,prefixes=4,samples=8,shards=1),
         'full':dict(questions=50,prefixes=4,samples=128,shards=4)}
PRIMARY=[('slice-token','ppo'),('rms-layer-token','ppo')]
SECONDARY=[('slice-chunk','slice-token'),('rms-layer-chunk','rms-layer-token'),
           ('slice-chunk','rms-layer-chunk'),('grpo','ppo')]


def child(out,seed):return out/f'seed{seed}'


def prepare(source,out):
    out.mkdir(parents=True,exist_ok=False)
    children={}
    for seed in SEEDS:
        folder=child(out,seed)
        base.prepare(source,folder,updates=120,actor_seed=seed)
        m=base.read(folder/'manifest.json')
        m.update(protocol='value-repeat-child-v1',budgets=BUDGETS,question_seed=48371,
            reference_seed=base.stable_seed(88213,'reference',seed),
            report_note='Per-seed component of a preregistered three-seed repeat. No automatic superiority claim; reference generation is additional.',
            scope='One of three fresh actor seeds. Original PPO GAE/value loss and GRPO; four original cumulative Ridge recipes. '
                  'Shared states within seed; actor-specific MC targets; same question IDs across seeds; inspected Numina data.')
        write_json(folder/'manifest.json',m)
        children[str(seed)]=sha256(folder/'manifest.json')
    write_json(out/'manifest.json',dict(protocol='value-repeat-v1',seeds=SEEDS,arms=list(base.ARMS),
        budgets=BUDGETS,steps=120,children=children,runtime=snapshot(ROOT)['files'],
        primary=[list(v) for v in PRIMARY],
        scope='Three actor seeds, 50 shared question IDs, up to four prefixes and 128 MC samples per state/actor. '
              'Each method updates its own actor using its original algorithm. Not a fixed-policy critic benchmark.'))


def validate(out):
    parent=base.read(out/'manifest.json')
    if (parent['protocol']!='value-repeat-v1' or parent['seeds']!=SEEDS or parent['budgets']!=BUDGETS
            or parent['arms']!=list(base.ARMS) or parent['primary']!=[list(v) for v in PRIMARY]
            or parent['runtime']!=snapshot(ROOT)['files']):raise ValueError('Repeat protocol/runtime changed')
    manifests={}
    for seed in SEEDS:
        folder=child(out,seed)
        if sha256(folder/'manifest.json')!=parent['children'][str(seed)]:raise ValueError('Child recipe changed')
        m=base.read(folder/'manifest.json')
        if (m['protocol']!='value-repeat-child-v1' or m['actor_seed']!=seed or m['arms']!=base.ARMS
                or m['steps']!=120 or m['budgets']!=BUDGETS or m['question_seed']!=48371
                or m['reference_seed']!=base.stable_seed(88213,'reference',seed)):
            raise ValueError('Invalid child protocol')
        for name,h in m['data_hashes'].items():
            if sha256(folder/name)!=h:raise ValueError('Child data changed')
        if sha256(folder/'filters/sparse.pt')!=m['filter_sha256']:raise ValueError('Filter changed')
        manifests[seed]=m
    identity=model_identity(ROOT/'models/qwen-math')
    if any(identity!=m['model_hashes'] for m in manifests.values()):raise ValueError('Model changed')
    return parent,manifests


def smoke_gate(out):
    gate=base.read(out/'smoke.json')
    path=child(out,SEEDS[0])/'smoke/verification.json'
    if (not gate.get('passed') or gate['manifest_sha256']!=sha256(out/'manifest.json')
            or gate['child_verification_sha256']!=sha256(path)):
        raise ValueError('Matching passed repeat smoke required')
    base.require_gate(child(out,SEEDS[0]),'smoke/verification.json')


def coordinates(task,stage,mode):
    seed_count=1 if mode=='smoke' else len(SEEDS)
    if stage in ('pack','training-gate','report'):
        if not 0<=task<seed_count:raise ValueError('Invalid seed task')
        return SEEDS[task],None,None
    shards=1 if mode=='smoke' or stage!='mc' else BUDGETS['full']['shards']
    count=len(base.ARMS)*shards
    if not 0<=task<seed_count*count:raise ValueError('Invalid arm/shard task')
    seed=SEEDS[task//count];within=task%count
    return seed,list(base.ARMS)[within%len(base.ARMS)],within//len(base.ARMS)


def question_arrays(summaries):
    expected=None;arrays={}
    for arm in base.ARMS:
        values=[]
        for summary in summaries:
            grouped={}
            for row in summary['panels'][arm]:
                grouped.setdefault(row['question_id'],[]).append((row['prediction']-row['reference']))
            ids=sorted(grouped)
            if expected is None:expected=ids
            if ids!=expected:raise ValueError('Question pairing differs between seeds/arms')
            values.append([[np.abs(grouped[q]).mean(),np.square(grouped[q]).mean()] for q in ids])
        arrays[arm]=np.asarray(values)
    return expected,arrays


def paired(arrays,a,b,tail=.025):
    d=arrays[a][:,:,0]-arrays[b][:,:,0];rng=np.random.default_rng(77311)
    draws=[]
    for _ in range(10000):
        seeds=rng.integers(d.shape[0],size=d.shape[0]);questions=rng.integers(d.shape[1],size=d.shape[1])
        draws.append(d[np.ix_(seeds,questions)].mean())
    return dict(mean=float(d.mean()),differences_by_seed=d.mean(1).tolist(),
        interval=np.quantile(draws,[tail,1-tail]).tolist(),
        scope='Approximate paired seed/question bootstrap; conditional on MC references. Three seeds limit precision.')


def ordinal_progress(summaries):
    # Four selected boundaries, not five token bins. Keep the SAME complete-case
    # question cohort at every position and across every seed; primary MAE retains all.
    groups=[]
    for summary in summaries:
        g={}
        for r in summary['panels']['slice-token']:g.setdefault(r['question_id'],[]).append(r)
        groups.append(g)
    ids=sorted(set.intersection(*[{q for q,rows in g.items() if len(rows)==4} for g in groups]))
    values={}
    for arm in base.ARMS:
        seed_values=[]
        for summary in summaries:
            g={}
            for r in summary['panels'][arm]:g.setdefault(r['question_id'],[]).append(r)
            ordered={q:sorted(g[q],key=lambda r:r['response_position']) for q in ids}
            seed_values.append([float(np.mean([abs(ordered[q][i]['prediction']-ordered[q][i]['reference']) for q in ids]))
                                for i in range(4)] if ids else [])
        values[arm]=seed_values
    return dict(questions=ids,questions_per_position=len(ids),states_per_seed=4*len(ids),values=values,
        scope='Secondary complete-case sampled-prefix order; same questions at all four positions and all seeds. Not token fractions.')


def aggregate(out,parent,mode):
    phase='smoke' if mode=='smoke' else 'full';seeds=SEEDS[:1] if mode=='smoke' else SEEDS
    summaries=[];actors={}
    for seed in seeds:
        folder=child(out,seed);base.require_gate(folder,f'{phase}/verification.json')
        summaries.append(base.read(folder/phase/'summary.json'))
        training=base.read(folder/f'{"smoke" if mode=="smoke" else "train"}-training.json')
        actors[str(seed)]=training['runs']
    folder=out/mode;folder.mkdir(exist_ok=True)
    if mode=='smoke':
        write_json(out/'smoke.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),
            child_verification_sha256=sha256(child(out,seeds[0])/'smoke/verification.json'),
            scope='Six original training algorithms, checkpoint restore, evaluation, references and per-seed plots completed. Not a ranking.'))
        return
    ids,arrays=question_arrays(summaries)
    metrics={}
    for arm,x in arrays.items():
        maes=x[:,:,0].mean(1);rmses=np.sqrt(x[:,:,1].mean(1))
        actor_results={}
        for seed in seeds:
            seed_folder=child(out,seed);m=base.read(seed_folder/'manifest.json')
            case=base.checkpoint_paths(seed_folder,m,arm,'full')[0]
            result=base.read(case/'result.json')
            actor_results[str(seed)]=dict(validation_accuracy=result['final_validation']['accuracy'],
                                         training_gpu_hours=result['training_gpu_hours'])
        metrics[arm]=dict(mae_mean=float(maes.mean()),mae_seed_sd=float(maes.std(ddof=1)),
            mae_by_seed=dict(zip(map(str,seeds),map(float,maes))),rmse_mean=float(rmses.mean()),
            rmse_by_seed=dict(zip(map(str,seeds),map(float,rmses))),actor_results=actor_results,
            actor_validation_accuracy_mean=float(np.mean([v['validation_accuracy'] for v in actor_results.values()])),
            training_gpu_hours_mean=float(np.mean([v['training_gpu_hours'] for v in actor_results.values()])))
    primary={f'{a}_minus_{b}':paired(arrays,a,b,.025/len(PRIMARY)) for a,b in PRIMARY}
    secondary={f'{a}_minus_{b}':paired(arrays,a,b) for a,b in SECONDARY}
    progress=ordinal_progress(summaries)
    rows=['Original six-actor algorithms: three seeds; 50 questions; 128 MC continuations/state.',
          'Each actor defines its own reference values. Shared prefixes within seed; question IDs match across seeds.',
          'arm                       MAE mean   seed SD    RMSE mean   actor val accuracy   train GPUh/run']
    rows += [f'{a:<24} {v["mae_mean"]:.5f}    {v["mae_seed_sd"]:.5f}    {v["rmse_mean"]:.5f}      {100*v["actor_validation_accuracy_mean"]:.2f}%              {v["training_gpu_hours_mean"]:.3f}' for a,v in metrics.items()]
    rows += ['Primary paired MAE differences (two contrasts; Bonferroni nominal adjustment):']
    rows += [f'{k}: {v}' for k,v in primary.items()]
    rows += [f'Ordinal progress: same {len(progress["questions"])} complete-case questions at each position.',
             'Raw predictions; no test tuning; GRPO is a four-answer question baseline, not a learned critic.',
             'References estimate verifier-defined stochastic values, not exact ground truth. Reused inspected data.']
    (folder/'report.txt').write_text('\n'.join(rows)+'\n')
    record=dict(seeds=seeds,questions=ids,metrics=metrics,primary=primary,secondary=secondary,
        progress=progress,training=actors,scope=parent['scope'],
        reference_worst_case_se=.5/np.sqrt(128),
        uncertainty='Seed/question bootstrap is conditional on MC references; not a universal ranking or architecture-only comparison.')
    write_json(folder/'summary.json',record);plot(folder,summaries,metrics,progress)
    write_json(folder/'verification.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),
        child_verification_sha256={str(seed):sha256(child(out,seed)/'full/verification.json') for seed in seeds},
        files={n:sha256(folder/n) for n in ('report.txt','summary.json','value-scatter.png','value-scatter.pdf','error-by-prefix-order.png')},
        scope='All three seeds complete; no automatic winner gate.'))


def plot(folder,summaries,metrics,progress):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#6a3d8f','#2374ab','#9670ae','#64a9ce','#e69f00','#009e73']
    v=[r['prediction'] for s in summaries for rows in s['panels'].values() for r in rows]
    fig,axes=plt.subplots(2,3,figsize=(14,9),sharex=True,sharey=True,layout='constrained')
    for ax,arm,title,color in zip(axes.flat,base.ARMS,base.TITLES,colors):
        ax.plot([0,1],[0,1],'k--',lw=1)
        for index,s in enumerate(summaries):
            rows=s['panels'][arm]
            ax.scatter([r['reference'] for r in rows],[r['prediction'] for r in rows],
                       marker=['o','s','^'][index],s=18,alpha=.25,color=color,label=str(SEEDS[index]))
        ax.set(title=title,xlim=(-.05,1.05),ylim=(min(-.05,min(v)-.03),max(1.05,max(v)+.03)))
        ax.text(.04,.94,f'MAE {metrics[arm]["mae_mean"]:.3f} (seed SD {metrics[arm]["mae_seed_sd"]:.3f})',transform=ax.transAxes,va='top',fontsize=9)
        ax.grid(alpha=.25)
    axes.flat[0].legend(title='Actor seed',fontsize=8)
    fig.suptitle('Original six-actor algorithms • three seeds • policy-specific MC values')
    fig.supxlabel('Monte Carlo reference value (panel/seed-specific actor)');fig.supylabel('Unclipped value / GRPO baseline')
    fig.savefig(folder/'value-scatter.png',dpi=180);fig.savefig(folder/'value-scatter.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    if progress['questions']:
        for arm,color in zip(base.ARMS,colors):
            x=np.array(progress['values'][arm]);mean=x.mean(0);sd=x.std(0,ddof=1)
            ax.errorbar([1,2,3,4],mean,yerr=sd,marker='o',label=arm,color=color,capsize=3)
    else:ax.text(.5,.5,'No questions have four sampled prefixes in all seeds',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Sampled prefix order (not token-progress percentage)',ylabel='Equal-question MAE',
        title=f'Matched question coverage: {len(progress["questions"])} questions/position; error bars = seed SD',xticks=[1,2,3,4])
    if progress['questions']:ax.legend(fontsize=8)
    ax.grid(alpha=.25);fig.savefig(folder/'error-by-prefix-order.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['prepare','train','training-gate','bank','pack','predict','mc','report','aggregate'])
    p.add_argument('--mode',choices=['smoke','main'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--task',type=int,default=0)
    a=p.parse_args();out=a.out.resolve()
    if a.stage=='prepare':
        if a.source is None:p.error('prepare needs source')
        prepare(a.source.resolve(),out);return
    parent,manifests=validate(out)
    if a.mode=='main':smoke_gate(out)
    if a.stage=='aggregate':aggregate(out,parent,a.mode);return
    seed,arm,shard=coordinates(a.task,a.stage,a.mode);folder=child(out,seed);m=manifests[seed]
    phase='smoke' if a.mode=='smoke' else 'full';(folder/phase).mkdir(exist_ok=True)
    status=folder/phase/f'{a.stage}-{a.task}-status.json';started=time.perf_counter()
    try:
        if a.stage=='train':base.train(folder,m,arm,a.mode=='smoke')
        elif a.stage=='training-gate':base.train_gate(folder,m,'smoke' if a.mode=='smoke' else 'train')
        else:
            base.require_gate(folder,f'{"smoke" if a.mode=="smoke" else "train"}-training.json')
            if a.stage=='pack':base.pack_bank(folder,m,phase)
            elif a.stage=='report':base.report(folder,m,phase)
            else:
                base.check_models(folder,m,arm,phase)
                from . import value_audit_gpu as gpu
                if a.stage=='mc':gpu.monte_carlo(folder,m,phase,arm,shard)
                else:getattr(gpu,a.stage)(folder,m,phase,arm)
        write_json(status,dict(state='complete',seconds=time.perf_counter()-started))
    except Exception as exc:
        write_json(status,dict(state='failed',error=str(exc)));raise


if __name__=='__main__':main()
