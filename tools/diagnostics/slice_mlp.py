"""Offline Ridge vs frozen-SLICE MLP vs jointly trained SLICE+MLP.

Uses cached RMS-normalized hidden vectors; never loads or updates Qwen/VERL.
Incomplete neural runs restart from their seed. Completed validation fits can be
reused only under an identical protocol. Test evaluation follows all model fits.
"""
import argparse
import copy
import json
from pathlib import Path
import time
from importlib.metadata import version

import numpy as np
import torch
from torch import nn

from slice_critic.inference import load_filter
from slice_critic.storage import sha256, write_json, save_tensor
from ridge_normalization import evaluate, ridge_fit

KINDS = ('frozen_mlp', 'joint_mlp')


class ValueMLP(nn.Module):
    def __init__(self, fitted, hidden=256, joint=False):
        super().__init__()
        compressor, mean, scale = fitted
        self.compressor = copy.deepcopy(compressor)
        self.compressor.requires_grad_(joint)
        # This old temporary head is not used, trained or exported as the new head.
        self.compressor.readout.requires_grad_(False)
        self.register_buffer('mean', mean.clone())
        self.register_buffer('scale', scale.clone())
        self.head = nn.Sequential(nn.Linear(compressor.projection.out_features, hidden),
                                  nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        z = self.compressor.features((x-self.mean)/self.scale)
        probability = self.head(z).flatten().sigmoid()
        return probability, z


def objective(model, x, y, sparsity):
    probability, z = model(x)
    mse = (probability-y).square().mean()
    return mse+sparsity*z.mean(), mse


def cached_split(source, split):
    folder = source/'packed/rms_hidden_concat'
    complete = json.loads((folder/'complete.json').read_text())
    for suffix in ('x.npy', 'labels.pt'):
        path = folder/f'{split}-{suffix}'
        if sha256(path) != complete['files'][path.name]:
            raise ValueError(f'Cached data checksum mismatch: {path}')
    x = np.load(folder/f'{split}-x.npy', mmap_mode='c')
    labels = torch.load(folder/f'{split}-labels.pt', map_location='cpu', weights_only=True)
    if x.ndim != 2 or x.shape[1] != 7680 or not len(x):
        raise ValueError('Expected nonempty 7680-wide feature cache')
    if any(len(labels[k]) != len(x) for k in ('y','question','response','position')):
        raise ValueError('Feature/label metadata mismatch')
    if set(labels['y'].tolist()) != {0., 1.}:
        raise ValueError('Every split must contain both outcomes')
    return x, labels


def tensor_batch(x, indices, device):
    batch = torch.from_numpy(np.array(x[indices], dtype=np.float32, copy=True))
    if not torch.isfinite(batch).all():
        raise ValueError('Nonfinite cached features')
    return batch.to(device)


@torch.inference_mode()
def predict(model, x, batch_size, device):
    model.eval()
    return torch.cat([model(tensor_batch(x, slice(i,i+batch_size), device))[0].cpu()
                      for i in range(0,len(x),batch_size)]).double().numpy()


def train_one(fitted, train, val, args, kind, seed, out, initial_checkpoint=None):
    case = out/f'{kind}-seed{seed}'
    case.mkdir(exist_ok=True)
    initial_hash = sha256(initial_checkpoint) if initial_checkpoint else None
    if (case/'result.json').exists():
        r = json.loads((case/'result.json').read_text())
        if (sha256(case/'best.pt') != r['checkpoint_sha256']
                or r.get('initial_checkpoint_sha256') != initial_hash):
            raise ValueError('Saved neural checkpoint changed')
        return r
    # Deliberately restart incomplete fits from initialization, never mix epochs.
    torch.manual_seed(seed)
    model = ValueMLP(fitted, args.hidden, joint=kind=='joint_mlp').to(args.device)
    if initial_checkpoint:
        model.load_state_dict(torch.load(initial_checkpoint,map_location='cpu',weights_only=True)['state'],strict=True)
    groups = [dict(params=model.head.parameters(), lr=args.head_lr)]
    if kind == 'joint_mlp':
        groups.append(dict(params=model.compressor.projection.parameters(), lr=args.compressor_lr))
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    y = train[1]['y'].float()
    best, best_epoch, history = float('inf'), 0, []
    started = time.perf_counter()
    if initial_checkpoint:
        predictions = predict(model,val[0],args.batch_size,args.device)
        best = float(((predictions-val[1]['y'].numpy())**2).mean())
        history.append(dict(epoch=0,validation_brier=best,train_objective=None,train_mse=None))
        save_tensor(case/'best.pt',dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                    seed=seed,kind=kind,hidden=args.hidden,epoch=0,validation_brier=best))
    for epoch in range(1,args.epochs+1):
        model.train()
        total_loss, total_mse = 0., 0.
        order = torch.randperm(len(y), generator=generator)
        for ids in order.split(args.batch_size):
            x = tensor_batch(train[0], ids.numpy(), args.device)
            optimizer.zero_grad(set_to_none=True)
            loss, mse = objective(model, x, y[ids].to(args.device), args.sparsity)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training objective')
            loss.backward()
            norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradients')
            optimizer.step()
            total_loss += float(loss.detach())*len(ids)
            total_mse += float(mse.detach())*len(ids)
        predictions = predict(model, val[0], args.batch_size, args.device)
        error = float(((predictions-val[1]['y'].numpy())**2).mean())
        history.append(dict(epoch=epoch, train_objective=total_loss/len(y),
                            train_mse=total_mse/len(y), validation_brier=error))
        if error < best:
            best, best_epoch = error, epoch
            state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            save_tensor(case/'best.pt', dict(state=state, seed=seed, kind=kind,
                        hidden=args.hidden, epoch=epoch, validation_brier=best))
        print(f'{kind} seed={seed} epoch={epoch} validation Brier={error:.6f}; best={best:.6f}', flush=True)
        if epoch-best_epoch >= args.patience:
            break
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    result = dict(kind=kind, seed=seed, best_epoch=best_epoch, epochs_run=epoch,
                  initial_checkpoint_sha256=initial_hash,
                  validation_brier=best, seconds=time.perf_counter()-started, history=history,
                  checkpoint_sha256=sha256(case/'best.pt'))
    write_json(case/'result.json', result)
    return result


@torch.inference_mode()
def fit_ridge(fitted, train, args, out):
    compressor, mean, scale = fitted
    k = compressor.projection.out_features+1
    a,b = torch.zeros(k,k,dtype=torch.float64), torch.zeros(k,dtype=torch.float64)
    for i in range(0,len(train[0]),args.batch_size):
        batch = tensor_batch(train[0], slice(i,i+args.batch_size), 'cpu')
        z = compressor.features((batch-mean)/scale).double()
        x = torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1)
        a += x.T@x
        b += x.T@train[1]['y'][i:i+len(z)].double()
    w, checks = ridge_fit(a,b,'legacy',0.,epsilon=args.ridge_epsilon)
    save_tensor(out/'ridge.pt',dict(weights=w,checks=checks))
    return w,checks


@torch.inference_mode()
def predict_ridge(fitted, x, w, batch):
    compressor,mean,scale = fitted
    rows=[]
    for i in range(0,len(x),batch):
        raw = tensor_batch(x,slice(i,i+batch),'cpu')
        z = compressor.features((raw-mean)/scale).double()
        rows.append((torch.cat([z,torch.ones(len(z),1,dtype=z.dtype)],1)@w).numpy())
    return np.concatenate(rows)


def run(args):
    source,study,out = args.features.resolve(),args.study.resolve(),args.out.resolve()
    if any(out == p or p in out.parents for p in (source,study)):
        raise ValueError('Output must be separate from source inputs')
    sm = json.loads((source/'manifest.json').read_text())
    m = json.loads((study/'manifest.json').read_text())
    if any(sm[k] != m[k] for k in ('model_hashes','layers','rms_eps')):
        raise ValueError('Source features and frozen-filter identities disagree')
    f = m['filters']['sparse']
    filter_path = study/'filters'/Path(f['path']).name
    fitted = load_filter(filter_path,f['sha256'],layers=m['layers'],eps=m['rms_eps'])
    data = {split:cached_split(source,split) for split in ('train','val','test')}
    sets = [set(labels['question'].tolist()) for _,labels in data.values()]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Overlapping question splits')
    protocol = dict(architecture='SLICE -> Linear(768, hidden) -> ReLU -> Linear(hidden,1) -> sigmoid',
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_manifest_sha256=sha256(source/'manifest.json'), study_manifest_sha256=sha256(study/'manifest.json'),
        cache_sha256=sha256(source/'packed/rms_hidden_concat/complete.json'),filter_sha256=f['sha256'],
        runtime=dict(torch=version('torch'),numpy=version('numpy'),slice_critic=version('slice-critic'),
                     gpu=torch.cuda.get_device_name() if args.device.startswith('cuda') else None),
        implementation={str(p.name):sha256(p) for p in [Path(__file__),Path(__file__).with_name('ridge_normalization.py'),
                        *sorted((Path(__file__).resolve().parents[2]/'packages/slice-critic/src/slice_critic').glob('*.py'))]},
        scope='Offline sampled-prefix prediction on reused inspected Numina questions/labels. '
              'Same initial frozen filter; no Qwen, LayerNorm or RL updates. '
              'MLPs use sigmoid-bounded values, MSE plus declared activation penalty; Ridge is unbounded. '
              'Identical head initialization and shuffle seeds across frozen/joint arms. '
              'Validation Brier selects epochs only; test scores are computed after all fits. '
              'Previously selected compressor used these validation families; no fresh-benchmark claim.')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists():
        if json.loads((out/'protocol.json').read_text()) != protocol:
            raise ValueError('Protocol/code changed: use a new output directory')
    elif any(out.iterdir()):
        raise ValueError('Output is not empty and has no protocol')
    write_json(out/'protocol.json',protocol)
    write_json(out/'status.json',dict(state='fitting'))
    start = time.perf_counter()
    w,checks = fit_ridge(fitted,data['train'],args,out)
    runs={}
    for seed in args.seeds:
        for kind in KINDS:
            runs[f'{kind}-seed{seed}'] = train_one(fitted,data['train'],data['val'],args,kind,seed,out)
    write_json(out/'status.json',dict(state='evaluating'))
    results,errors={},{}
    for split in ('val','test'):
        x,labels = data[split]
        predictions = {'ridge':predict_ridge(fitted,x,w,args.batch_size)}
        for key,r in runs.items():
            checkpoint=torch.load(out/key/'best.pt',weights_only=True,map_location='cpu')
            model=ValueMLP(fitted,args.hidden,joint=r['kind']=='joint_mlp').to(args.device)
            model.load_state_dict(checkpoint['state'],strict=True)
            predictions[key]=predict(model,x,args.batch_size,args.device)
            del model
        results[split]={}
        for key,pred in predictions.items():
            results[split][key],grouped=evaluate(labels['y'].numpy(),pred,labels['question'].numpy())
            np.save(out/f'{split}-{key}-predictions.npy',pred)
            if split=='test': errors[key]=grouped
        save_tensor(out/f'{split}-labels.pt',labels)
    aggregate,paired={},{}
    for kind in KINDS:
        scores=[results['test'][f'{kind}-seed{s}'] for s in args.seeds]
        aggregate[kind]=dict(brier_mean=float(np.mean([r['raw_brier'] for r in scores])),
            accuracy_mean=float(np.mean([r['accuracy'] for r in scores])),
            brier_seed_sd=float(np.std([r['raw_brier'] for r in scores],ddof=1)) if len(scores)>1 else None)
        delta=np.stack([errors[f'{kind}-seed{s}']-errors['ridge'] for s in args.seeds])
        paired[f'{kind}_minus_ridge']=dict(question_mean_brier_difference=float(delta.mean()),
            differences_by_seed={str(s):float(delta[i].mean()) for i,s in enumerate(args.seeds)})
    delta=np.stack([errors[f'joint_mlp-seed{s}']-errors[f'frozen_mlp-seed{s}'] for s in args.seeds])
    paired['joint_minus_frozen']=dict(question_mean_brier_difference=float(delta.mean()),
        differences_by_seed={str(s):float(delta[i].mean()) for i,s in enumerate(args.seeds)})
    for key,value in paired.items():
        d = (delta if key=='joint_minus_frozen' else
             np.stack([errors[f'{key.split("_minus_")[0]}-seed{s}']-errors['ridge'] for s in args.seeds]))
        rng=np.random.default_rng(42)
        draws=[float(d[rng.integers(len(d),size=len(d))][:,rng.integers(d.shape[1],size=d.shape[1])].mean())
               for _ in range(2000)]
        value['descriptive_95_interval']=np.quantile(draws,[.025,.975]).tolist()
        value['scope']='Seed/question bootstrap; exploratory, unadjusted; one initial compressor'
    split_counts={split:dict(prefixes=len(labels['y']),questions=len(set(labels['question'].tolist())),
                            retained_success_rate=float(labels['y'].double().mean()))
                  for split,(_,labels) in data.items()}
    write_json(out/'summary.json',dict(protocol=protocol,ridge_checks=checks,runs=runs,split_counts=split_counts,
        results=results,aggregate=aggregate,paired=paired,seconds=time.perf_counter()-start))
    lines=[protocol['scope'],'','Test value prediction: Brier lower; accuracy higher.',
           f'Ridge: raw Brier={results["test"]["ridge"]["raw_brier"]:.6f}; '
           f'clipped Brier={results["test"]["ridge"]["clipped_brier"]:.6f}; '
           f'accuracy={results["test"]["ridge"]["accuracy"]:.5f}']
    rate=split_counts['test']['retained_success_rate']
    lines.append(f'Test prefixes={split_counts["test"]["prefixes"]}; '
                 f'questions={split_counts["test"]["questions"]}; majority-class accuracy={max(rate,1-rate):.5f}')
    lines.extend(f'{k}: {v}' for k,v in aggregate.items())
    lines.extend(f'{k}: {v}' for k,v in paired.items())
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    write_json(out/'status.json',dict(state='complete'))
    print('\n'.join(lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('features','study','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=2048)
    p.add_argument('--head-lr',type=float,default=1e-3)
    p.add_argument('--compressor-lr',type=float,default=1e-4)
    p.add_argument('--sparsity',type=float,default=.01)
    p.add_argument('--weight-decay',type=float,default=.001)
    p.add_argument('--ridge-epsilon',type=float,default=.01)
    args=p.parse_args()
    if (min(args.hidden,args.epochs,args.patience,args.batch_size)<1 or len(set(args.seeds))!=len(args.seeds)
            or any(not np.isfinite(v) or v<=0 for v in (args.head_lr,args.compressor_lr,args.ridge_epsilon))
            or any(not np.isfinite(v) or v<0 for v in (args.sparsity,args.weight_decay))):
        p.error('Invalid dimensions, seeds or optimization settings')
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    run(args)


if __name__=='__main__': main()
