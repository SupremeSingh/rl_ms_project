"""Two-stage task-trained bottleneck study on saved Numina data.

Screen: matched plain/sparse BCE training; selection by validation Ridge Brier.
Stats: freeze encoders, re-extract complete training transitions, rebuild A/b.
Report: compare on original test prefixes only after selection is locked.
"""
import argparse
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from math_rl.sparse_bottleneck import Bottleneck, encode, loss
from math_rl.large_linear import statistics, accumulate, solve
from math_rl.large_probe import streaming_normalization, paired_errors
from math_rl.critic_probe import metrics
from math_rl.provenance import sha256
from math_rl.lstd import design
from large_jlens_study import read, write, dataset, trajectory, linear_features
import frozen_critics as frozen

ARM = 'rms_hidden_concat'


def save(path, record):
    frozen.save_tensor(path, record)
    write(path.with_suffix('.json'), {'sha256': sha256(path)})


def load(path):
    if sha256(path) != read(path.with_suffix('.json'))['sha256']:
        raise ValueError(f'Changed checkpoint: {path}')
    return torch.load(path, weights_only=True, map_location='cpu')


def ridge_fit(x, y, epsilon=.01):
    phi = design(x, torch.zeros(x.shape[1]), torch.ones(x.shape[1]))
    system = phi.T@phi
    system.diagonal().add_(epsilon)
    return torch.linalg.solve(system, phi.T@y.double())


def prediction(x, w):
    return (design(x, torch.zeros(x.shape[1]), torch.ones(x.shape[1]))@w).numpy()


def model_from(record, device):
    model = Bottleneck(len(record['mean']), record['width']).to(device)
    model.load_state_dict(record['state'])
    return model.eval().requires_grad_(False)


def setup(args):
    source, out = args.source.resolve(), args.out.resolve()
    if out == source or source in out.parents:
        raise ValueError('Output must be separate from source')
    m = read(source/'manifest.json')
    if m['config']['linear_gamma'] != 1 or m['config']['linear_lambda'] != .99:
        raise ValueError('Expected gamma=1, lambda=.99 source study')
    cfg = dict(source=str(source), width=args.width, epochs=args.epochs, seeds=[42,43,44],
        lrs=[.0003,.001], sparse_penalty=.01, batch=2048, epsilon=.01,
        source_manifest=sha256(source/'manifest.json'),
        packed={arm: sha256(source/'packed'/arm/'complete.json') for arm in (ARM,'hidden_two_thirds','final')},
        code={p.name:sha256(p) for p in (Path(__file__), ROOT/'src/math_rl/sparse_bottleneck.py',
                                        ROOT/'src/math_rl/large_linear.py')})
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists() and read(out/'protocol.json') != cfg:
        raise ValueError('Changed protocol: use a new output directory')
    write(out/'protocol.json', cfg)
    return source,out,m,cfg


def screen(source,out,m,cfg,device):
    if (out/'screen.json').exists():
        for item in read(out/'screen.json')['selected'].values(): load(out/item['path'])
        return
    train,val = dataset(source,ARM,'train'),dataset(source,ARM,'val')
    mean,scale = streaming_normalization(train['x'].numpy())
    md,sd = mean.to(device),scale.to(device)
    selected={}; trials=[]
    for kind,penalty in [('plain',0.),('sparse',cfg['sparse_penalty'])]:
        for seed in cfg['seeds']:
            candidates=[]
            for lr in cfg['lrs']:
                path=out/f'{kind}-{seed}-{lr}.pt'
                if path.exists(): rec=load(path)
                else:
                    torch.manual_seed(seed)
                    model=Bottleneck(len(mean),cfg['width']).to(device)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.001)
                    generator=torch.Generator().manual_seed(seed)
                    best=float('inf'); best_state=None; history=[]; start=time.perf_counter()
                    # Fixed epoch budget for every arm/trial; same ordering by seed.
                    for epoch in range(cfg['epochs']):
                        model.train()
                        for ids in torch.randperm(len(train['y']),generator=generator).split(cfg['batch']):
                            x=(train['x'][ids].to(device)-md)/sd
                            objective=loss(model,x,train['y'][ids].to(device).float(),penalty)
                            if not torch.isfinite(objective): raise ValueError('Nonfinite training loss')
                            optimizer.zero_grad(); objective.backward(); optimizer.step()
                        model.eval()
                        with torch.no_grad():
                            pv=torch.cat([model((x.to(device)-md)/sd)[0].sigmoid().cpu()
                                          for x in val['x'].split(cfg['batch'])]).numpy()
                        brier=metrics(val['y'].numpy(),pv)['brier']
                        history.append(brier)
                        if brier<best:
                            best=brier; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                        print(kind,seed,lr,epoch+1,brier,flush=True)
                    model.load_state_dict(best_state)
                    ztrain=encode(model,train['x'],mean,scale,device)
                    zval=encode(model,val['x'],mean,scale,device)
                    w=ridge_fit(ztrain,train['y'],cfg['epsilon'])
                    rec=dict(state=best_state,mean=mean,scale=scale,width=cfg['width'],seed=seed,kind=kind,
                        penalty=penalty,lr=lr,history=history,seconds=time.perf_counter()-start,
                        validation=metrics(val['y'].numpy(),prediction(zval,w)),
                        zero_fraction=float((zval==0).float().mean()),
                        dead_units=int((ztrain==0).all(0).sum()),screen_ridge_weights=w)
                    save(path,rec)
                candidates.append((rec['validation']['brier'],path,rec))
                trials.append(dict(kind=kind,seed=seed,lr=lr,validation=rec['validation'],
                                   zero_fraction=rec['zero_fraction'],dead_units=rec['dead_units']))
            _,path,rec=min(candidates,key=lambda r:r[0])
            selected[f'{kind}-{seed}']=dict(path=path.name,sha256=sha256(path),validation=rec['validation'])
    # Matched sampled-prefix Ridge baseline; no comparison to all-token fit at screen.
    base_train,base_val=dataset(source,'hidden_two_thirds','train'),dataset(source,'hidden_two_thirds','val')
    bm,bs=streaming_normalization(base_train['x'].numpy())
    w=ridge_fit((base_train['x']-bm)/bs,base_train['y'])
    baseline=metrics(base_val['y'].numpy(),prediction((base_val['x']-bm)/bs,w))
    means={kind:float(np.mean([r['validation']['brier'] for k,r in selected.items() if k.startswith(kind)]))
           for kind in ('plain','sparse')}
    passed=min(means.values()) <= baseline['brier']
    write(out/'screen.json',dict(selected=selected,trials=trials,baseline=baseline,mean_validation_brier=means,
        recommended_full=passed,scope='Validation only; sampled-prefix Ridge screen; BCE-trained features. '
        'Same budgets/seeds/LRs; sparse differs only by fixed activation penalty. No CRATE/rate-reduction or orthogonality claim.'))
    (out/'screen-report.txt').write_text(f'Validation Ridge Brier: {means}\nTwo-thirds matched baseline: {baseline["brier"]:.6f}\n'
        f'Recommend all-token comparison: {passed}\nInspect screen.json for sparsity, dead units and individual seeds.\n')


def stats(source,out,m,cfg,device,shard):
    from math_rl.critic_layers import sampled_block_states
    import compare_jspace_probes as original
    screening=read(out/'screen.json')
    if not screening['recommended_full']:
        raise ValueError('Validation screen did not pass; inspect screen-report.txt before further work')
    if original.installed_lens()!=read(source/'lens-metadata.json')['environment']:
        raise ValueError('Use the source feature extraction environment')
    paths={k:out/v['path'] for k,v in screening['selected'].items()}
    records={k:load(p) for k,p in paths.items()}
    if sha256(source/'questions.json') != m['questions_sha256']:
        raise ValueError('Source questions changed')
    for name,checksum in m['model_hashes'].items():
        if sha256(ROOT/'models/qwen-math'/name) != checksum:
            raise ValueError('Frozen model/tokenizer changed')
    if any(sha256(paths[k])!=screening['selected'][k]['sha256'] for k in paths):
        raise ValueError('Selection changed')
    dest=out/'statistics';dest.mkdir(exist_ok=True);path=dest/f'shard-{shard}.pt'
    lock={k:sha256(p) for k,p in paths.items()}
    old=load(path) if path.exists() else None
    if old and old['lock']!=lock: raise ValueError('Frozen bottleneck changed')
    if old and old.get('complete'): return
    models={k:model_from(r,device) for k,r in records.items()}
    totals=old['stats'] if old else {k:statistics(cfg['width']) for k in records}
    totals={k:{n:v.to(device) if torch.is_tensor(v) else v for n,v in s.items()} for k,s in totals.items()}
    processed=old['processed'] if old else {};del old
    hf,_=original.load_model()
    questions=read(source/'questions.json')
    questions=[q for q in questions if q['split']=='train' and q['index']%m['config']['shards']==shard]
    source_record=load_source_shard(source,shard)
    expected=source_record['processed']
    expected_counts={n:source_record['stats'][ARM][n] for n in ('transitions','trajectories')}
    del source_record
    zero=torch.zeros(cfg['width']);one=torch.ones(cfg['width'])
    def checkpoint(complete=False):
        save(path,dict(lock=lock,processed=processed,complete=complete,
             stats={k:{n:v.cpu() if torch.is_tensor(v) else v for n,v in s.items()} for k,s in totals.items()}))
    for i,q in enumerate(questions):
        file=trajectory(source,q);checksum=sha256(file)
        if expected.get(q['id'])!=checksum: raise ValueError('Source trajectory changed since original statistics')
        if q['id'] in processed:
            if processed[q['id']]!=checksum: raise ValueError('Changed resumed trajectory')
            continue
        for row in read(file)['responses']:
            if not frozen.usable_response(row):continue
            t=len(row['token_ids']);n=len(q['prompt_token_ids'])
            ids=torch.tensor([q['prompt_token_ids']+row['token_ids']],device=device)
            blocks=sampled_block_states(hf.model,ids,n,t,list(range(t)),m['layers'])
            x=linear_features(ARM,blocks,m,None,None,None)
            for k,model in models.items():
                r=records[k];z=encode(model,x,r['mean'],r['scale'],device)
                accumulate(totals[k],z,float(row['score']),zero,one,trace_lambda=.99)
        processed[q['id']]=checksum
        if (i+1)%32==0:checkpoint()
        print('Accumulated',q['id'],flush=True)
    if set(processed)!=set(expected): raise ValueError('Accumulated question set differs from source')
    for total in totals.values():
        if any(total[n]!=v for n,v in expected_counts.items()):
            raise ValueError('Accumulated answer/token counts differ from matched source')
    checkpoint(True)


def load_source_shard(source,shard):
    p=source/'linear-statistics'/f'shard-{shard}.pt'
    if sha256(p)!=read(p.with_suffix('.json'))['sha256']: raise ValueError('Source statistics changed')
    return torch.load(p,weights_only=True,map_location='cpu')


def report(source,out,m,cfg,device):
    screening=read(out/'screen.json');models={k:load(out/v['path']) for k,v in screening['selected'].items()}
    totals=None
    for shard in range(m['config']['shards']):
        record=load(out/'statistics'/f'shard-{shard}.pt')
        if not record['complete']:raise ValueError('Incomplete accumulation')
        if record['lock']!={k:v['sha256'] for k,v in screening['selected'].items()}:raise ValueError('Changed selection')
        if totals is None:totals=record['stats']
        else:
            for k in totals:
                for name in totals[k]:totals[k][name]+=record['stats'][k][name]
    weights={}
    for k,s in totals.items():
        for head in ('ridge','lstd-0.99'):
            w,checks=solve(s,head,cfg['epsilon']);weights[f'{k}/{head}']=w
            save(out/f'{k}-{head}-fit.pt',dict(weights=w,checks=checks))
    del totals
    data=dataset(source,ARM,'test');y=data['y'].numpy();q=data['question'].numpy()
    predictions={};results={}
    for k,r in models.items():
        z=encode(model_from(r,device),data['x'],r['mean'],r['scale'],device)
        for head in ('ridge','lstd-0.99'):
            key=f'{k}/{head}';p=prediction(z,weights[key]);predictions[key]=p
            results[key]=metrics(y,p)
    # Original all-token controls, with exact evaluation-state alignment.
    from math_rl.lstd import predict as predict_original
    for arm in (ARM,'hidden_two_thirds','final'):
        control=dataset(source,arm,'test')
        for field in ('y','question','position','response'):
            if not torch.equal(data[field],control[field]):raise ValueError('Unmatched test states')
        selected=read(source/'linear-fits'/arm/'selection.json')['selected']
        for head in ('ridge','lstd-0.99'):
            item=selected[head];path=source/item['path']
            if sha256(path)!=item['sha256']:raise ValueError('Changed baseline weights')
            r=torch.load(path,weights_only=True);p=predict_original(r['weights'],control,r['mean'],r['scale'])
            predictions[f'{arm}/{head}']=p;results[f'{arm}/{head}']=metrics(y,p)
    # Matched PCA from all-token training covariance, same width and epsilon.
    from compress_jlens_critics import merge
    from math_rl.critic_compression import pca_from_statistics,affine_projection,project_statistics
    s,(mean,scale)=merge(source,ARM,m['config']['shards'])
    center,eigenvalues,vectors=pca_from_statistics(s)
    transform=affine_projection(center,vectors[:,:cfg['width']]);small=project_statistics(s,transform)
    for head in ('ridge','lstd-0.99'):
        w,_=solve(small,head,cfg['epsilon'])
        p=predict_original(transform@w,data,mean,scale)
        predictions[f'pca/{head}']=p;results[f'pca/{head}']=metrics(y,p)
    for kind in ('plain','sparse'):
        for head in ('ridge','lstd-0.99'):
            rows=[results[f'{kind}-{seed}/{head}'] for seed in cfg['seeds']]
            results[f'{kind}-mean/{head}']={metric:float(np.mean([r[metric] for r in rows]))
                for metric in ('accuracy','brier','ece')}
    comparisons={}
    for head in ('ridge','lstd-0.99'):
        for kind in ('plain','sparse'):
            a=np.stack([predictions[f'{kind}-{seed}/{head}'] for seed in cfg['seeds']])
            for baseline in (ARM,'hidden_two_thirds','final','pca','plain'):
                if baseline==kind:continue
                b=(np.stack([predictions[f'plain-{seed}/{head}'] for seed in cfg['seeds']]) if baseline=='plain'
                   else np.repeat(predictions[f'{baseline}/{head}'][None],len(a),axis=0))
                comparisons[f'{kind}/{head}_minus_{baseline}']=paired_errors(y,a,b,q)
    scope='Offline exploratory inspected Numina test; BCE-trained frozen features; all-token Ridge/LSTD(.99). '
    scope+='Question bootstrap averages head seeds (not independent datasets). Not a CRATE replication, PPO result or cost superiority claim.'
    write(out/'summary.json',dict(scope=scope,results=results,comparisons=comparisons))
    lines=[scope,'representation/head                 accuracy    Brier']
    for k,r in results.items():lines.append(f'{k:38s} {r["accuracy"]:.4f}    {r["brier"]:.5f}')
    lines += [f'{k}: {v}' for k,v in comparisons.items()]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    np.savez_compressed(out/'predictions.npz',**predictions,y=y,question=q)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--stage',choices=['screen','stats','report'],default='screen')
    p.add_argument('--task',type=int,default=0);p.add_argument('--width',type=int,default=768)
    p.add_argument('--epochs',type=int,default=30)
    args=p.parse_args()
    if args.width<1 or args.epochs<1: p.error('Positive width and epochs required')
    torch.set_num_threads(8)
    source,out,m,cfg=setup(args)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start=time.perf_counter();status=out/f'{args.stage}-{args.task}-status.json'
    try:
        write(status,dict(state='running'))
        if args.stage=='screen':screen(source,out,m,cfg,device)
        elif args.stage=='stats':
            if not 0<=args.task<m['config']['shards']:raise ValueError('Invalid shard')
            stats(source,out,m,cfg,device,args.task)
        else:report(source,out,m,cfg,device)
    except Exception as exc:
        write(status,dict(state='failed',error=str(exc),seconds=time.perf_counter()-start));raise
    write(status,dict(state='complete',seconds=time.perf_counter()-start))

if __name__=='__main__':main()
