"""Validation-only head-first MLP tuning, followed by separately invoked evaluation."""
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch

from slice_critic.inference import load_filter
from slice_critic.storage import sha256, write_json
from slice_mlp import (ValueMLP, cached_split, train_one, predict, fit_ridge,
                       predict_ridge, evaluate)


@dataclass(frozen=True)
class Recipe:
    head_lrs: tuple = (1e-3, 3e-4)
    compressor_lrs: tuple = (1e-5, 3e-6)
    penalties: tuple = (0., .001, .01)
    seeds: tuple = (42,43,44)


def read(path):
    return json.loads(Path(path).read_text())


def load_inputs(args, include_test=False):
    source, online = args.features.resolve(), args.study.resolve()
    m, s = read(online/'manifest.json'), read(source/'manifest.json')
    if any(m[k] != s[k] for k in ('model_hashes','layers','rms_eps')):
        raise ValueError('Source/filter identities differ')
    info=m['filters']['sparse']
    fitted=load_filter(online/'filters'/Path(info['path']).name,info['sha256'],
                       layers=m['layers'],eps=m['rms_eps'])
    splits=('train','val','test') if include_test else ('train','val')
    data={split:cached_split(source,split) for split in splits}
    sets=[set(v[1]['question'].tolist()) for v in data.values()]
    if any(sets[i]&sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError('Question split overlap')
    return fitted,data


def protocol(args):
    source, online = args.features.resolve(),args.study.resolve()
    recipe=Recipe()
    return dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='stage'},
        recipe=json.loads(json.dumps(asdict(recipe))),
        source_manifest=sha256(source/'manifest.json'),filter_manifest=sha256(online/'manifest.json'),
        cache_manifest=sha256(source/'packed/rms_hidden_concat/complete.json'),
        filter_sha256=read(online/'manifest.json')['filters']['sparse']['sha256'],
        implementation={p.name:sha256(p) for p in [Path(__file__),Path(__file__).with_name('slice_mlp.py'),
                           Path(__file__).with_name('ridge_normalization.py'),
                           *sorted((Path(__file__).resolve().parents[2]/'packages/slice-critic/src/slice_critic').glob('*.py'))]},
        runtime=dict(torch=torch.__version__,numpy=np.__version__,
                     gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None),
        scope='Offline exploratory follow-up. Cached inspected Numina questions and verifier labels. '
              'Frozen Qwen features, no LayerNorm or RL. Pilot seed 42 tunes on validation only; '
              'locked recipes are repeated with head/shuffle seeds 43 and 44. '
              'No independent test benchmark or independent compressor fits.')


def verify_selection(out):
    selection=read(out/'selection.json')
    gate=read(out/'selection-lock.json')
    if gate['sha256'] != sha256(out/'selection.json') or gate['protocol_sha256'] != sha256(out/'protocol.json'):
        raise ValueError('Selection or protocol changed after locking')
    for rows in selection['models'].values():
        for row in rows:
            if sha256(out/row['checkpoint']) != row['sha256']:
                raise ValueError('Selected checkpoint changed')
    if sha256(out/'ridge.pt') != selection['ridge_sha256']:
        raise ValueError('Ridge baseline checkpoint changed')
    return selection


def args_for(args, head_lr, compressor_lr=1e-5, penalty=0.):
    return SimpleNamespace(hidden=256,device=args.device,batch_size=args.batch_size,
        head_lr=head_lr,compressor_lr=compressor_lr,sparsity=penalty,
        weight_decay=.001,epochs=args.epochs,patience=args.patience,ridge_epsilon=.01)


def fit_case(fitted,data,args,out,name,kind,seed,config,warm=None):
    folder=out/name
    folder.mkdir(parents=True,exist_ok=True)
    r=train_one(fitted,data['train'],data['val'],config,kind,seed,folder,initial_checkpoint=warm)
    checkpoint=folder/f'{kind}-seed{seed}/best.pt'
    return dict(case=str(checkpoint.parent.relative_to(out)),checkpoint=str(checkpoint.relative_to(out)),
                sha256=sha256(checkpoint),seed=seed,validation_brier=r['validation_brier'],
                best_epoch=r['best_epoch'],seconds=r['seconds'])


def tune(args,out):
    if (out/'selection.json').exists() and (out/'selection-lock.json').exists():
        verify_selection(out)
        print('Validation selection already locked; use --stage evaluate for test scores.',flush=True)
        return
    fitted,data=load_inputs(args)  # Intentionally never opens test feature/label files.
    recipe=Recipe()
    write_json(out/'status.json',dict(state='validation_tuning'))
    w,checks=fit_ridge(fitted,data['train'],args_for(args,1e-3),out)
    raw=predict_ridge(fitted,data['val'][0],w,args.batch_size)
    ridge_validation,_=evaluate(data['val'][1]['y'].numpy(),raw,data['val'][1]['question'].numpy())
    trials=[]
    for i,lr in enumerate(recipe.head_lrs):
        config=args_for(args,lr)
        row=fit_case(fitted,data,args,out,f'pilot/head-{i}','frozen_mlp',42,config)
        trials.append(dict(row,kind='frozen_mlp',head_lr=lr))
    head=min(trials,key=lambda r:r['validation_brier'])
    warm=out/head['checkpoint']
    for i,lr in enumerate(recipe.compressor_lrs):
        for j,penalty in enumerate(recipe.penalties):
            # Fine-tune both components gently after the head has learned its task.
            config=args_for(args,head['head_lr']/10,lr,penalty)
            row=fit_case(fitted,data,args,out,f'pilot/joint-{i}-{j}','joint_mlp',42,config,warm)
            trials.append(dict(row,kind='joint_mlp',head_lr=config.head_lr,
                               compressor_lr=lr,penalty=penalty))
    joint=min((r for r in trials if r['kind']=='joint_mlp'),key=lambda r:r['validation_brier'])
    # Persist the chosen hyperparameters before fits for additional seeds.
    write_json(out/'pilot-selection.json',dict(head=head,joint=joint,trials=trials))
    models={'frozen_mlp':[head],'joint_mlp':[joint]}
    for seed in recipe.seeds[1:]:
        config=args_for(args,head['head_lr'])
        row=fit_case(fitted,data,args,out,f'confirmation/seed-{seed}','frozen_mlp',seed,config)
        models['frozen_mlp'].append(row)
        config=args_for(args,joint['head_lr'],joint['compressor_lr'],joint['penalty'])
        row=fit_case(fitted,data,args,out,f'confirmation/seed-{seed}','joint_mlp',seed,
                     config,out/row['checkpoint'])
        models['joint_mlp'].append(row)
    selection=dict(head_recipe=dict(lr=head['head_lr']),
        joint_recipe={k:joint[k] for k in ('head_lr','compressor_lr','penalty')},
        models=models,trials=trials,ridge_validation=ridge_validation,ridge_checks=checks,
        ridge_sha256=sha256(out/'ridge.pt'),
        scope='Recipes selected only on seed-42 validation Brier. Each joint fit starts '
              'from its paired best frozen-head checkpoint. Epoch zero is eligible: '
              'joint best_epoch=0 means fine-tuning did not improve validation.')
    write_json(out/'selection.json',selection)
    write_json(out/'selection-lock.json',dict(sha256=sha256(out/'selection.json'),
                                            protocol_sha256=sha256(out/'protocol.json')))
    lines=['Validation-only tuning complete; no test predictions computed.',selection['scope'],
           f'Ridge validation: {ridge_validation}',f'Head recipe: {selection["head_recipe"]}',
           f'Joint recipe: {selection["joint_recipe"]}']
    lines.extend(f'{kind} seed={r["seed"]}: validation Brier={r["validation_brier"]:.6f}; '
                 f'best epoch={r["best_epoch"]}' for kind,rows in models.items() for r in rows)
    (out/'validation-report.txt').write_text('\n'.join(lines)+'\n')
    write_json(out/'status.json',dict(state='validation_complete'))
    print('\n'.join(lines),flush=True)


def bootstrap(delta):
    rng=np.random.default_rng(42)
    draws=[float(delta[rng.integers(len(delta),size=len(delta))][:,
          rng.integers(delta.shape[1],size=delta.shape[1])].mean()) for _ in range(2000)]
    return dict(question_mean_brier_difference=float(delta.mean()),
                descriptive_95_interval=np.quantile(draws,[.025,.975]).tolist(),
                scope='Exploratory seed/question bootstrap, unadjusted; includes pilot seed')


def evaluate_locked(args,out):
    selected=verify_selection(out)
    fitted,data=load_inputs(args,include_test=True)
    x,labels=data['test']
    w=torch.load(out/'ridge.pt',map_location='cpu',weights_only=True)['weights']
    raw=predict_ridge(fitted,x,w,args.batch_size)
    predictions={'ridge_raw':raw,'ridge_clipped':np.clip(raw,0,1)}
    for kind,rows in selected['models'].items():
        for row in rows:
            model=ValueMLP(fitted,256,joint=kind=='joint_mlp').to(args.device)
            record=torch.load(out/row['checkpoint'],weights_only=True,map_location='cpu')
            model.load_state_dict(record['state'],strict=True)
            predictions[f'{kind}-seed{row["seed"]}']=predict(model,x,args.batch_size,args.device)
            del model
    results,errors={},{}
    for key,pred in predictions.items():
        results[key],errors[key]=evaluate(labels['y'].numpy(),pred,labels['question'].numpy())
        np.save(out/f'test-{key}-predictions.npy',pred)
    paired,aggregate={},{}
    for kind,rows in selected['models'].items():
        keys=[f'{kind}-seed{r["seed"]}' for r in rows]
        aggregate[kind]=dict(brier_mean=float(np.mean([results[k]['raw_brier'] for k in keys])),
                            accuracy_mean=float(np.mean([results[k]['accuracy'] for k in keys])))
        for baseline in ('ridge_raw','ridge_clipped'):
            paired[f'{kind}_minus_{baseline}']=bootstrap(np.stack([errors[k]-errors[baseline] for k in keys]))
    paired['joint_minus_frozen']=bootstrap(np.stack([errors[f'joint_mlp-seed{s}']-errors[f'frozen_mlp-seed{s}']
                                                     for s in Recipe().seeds]))
    torch.save(labels,out/'test-labels.pt')
    write_json(out/'summary.json',dict(selection=selected,results=results,aggregate=aggregate,paired=paired,
        scope='Locked validation recipes; reused inspected test questions. Seeds 43/44 '
              'were not used to select hyperparameters; seed 42 was. No fresh-benchmark claim.'))
    lines=['Locked-model test evaluation; offline prediction, not RL.',
           f'Ridge raw: {results["ridge_raw"]}',f'Ridge clipped: {results["ridge_clipped"]}']
    lines.extend(f'{k}: {v}' for k,v in aggregate.items())
    lines.extend(f'{k}: {v}' for k,v in paired.items())
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    write_json(out/'status.json',dict(state='complete'))
    print('\n'.join(lines),flush=True)


def run(args):
    out=args.out.resolve()
    if any(out==p.resolve() or p.resolve() in out.parents for p in (args.features,args.study)):
        raise ValueError('Output must be separate from inputs')
    expected=protocol(args)
    # JSON canonicalization makes tuple/list metadata comparable on retries.
    expected=json.loads(json.dumps(expected))
    if (out/'protocol.json').exists():
        if read(out/'protocol.json')!=expected:
            raise ValueError('Protocol changed; use a new output directory')
    elif args.stage=='evaluate':
        raise ValueError('Run validation tuning before evaluation')
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError('Output is not empty')
        out.mkdir(parents=True,exist_ok=True)
        write_json(out/'protocol.json',expected)
    if args.stage=='tune': tune(args,out)
    else: evaluate_locked(args,out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','study','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['tune','evaluate'],default='tune')
    p.add_argument('--device',default='cuda')
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--patience',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=2048)
    args=p.parse_args()
    if min(args.epochs,args.patience,args.batch_size)<1: p.error('Positive budgets required')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    run(args)


if __name__=='__main__': main()
