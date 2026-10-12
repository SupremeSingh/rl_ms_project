"""Full geometric J-lens logit readouts versus matched hidden features; linear heads.

Uses every vocabulary direction through exact Gram-factor coordinates. No sparse
reconstruction or softmax. Existing geometric features and official lenses are
read-only. Ridge screens linear value features; this is not LSTD/PPO integration.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

import compare_jspace_probes as original
from refine_jspace_probes import paired, prediction
from math_rl.critic_probe import fit_ridge, metrics
from math_rl.direct_jlens import compact_readout, isotropic_normalization, readout_factor, rms_coordinates
from math_rl.jspace import load_readout_weights
from math_rl.provenance import sha256, snapshot, write_json

ARMS = ('direct_jlens_concat', 'hidden_concat', 'rms_hidden_concat', 'hidden_two_thirds')


def fit_and_report(out, manifest):
    records = {}; trials = {}
    for arm in ARMS:
        train, val = [torch.load(out/arm/f'{split}.pt', weights_only=True) for split in ('train','val')]
        mean, scale = isotropic_normalization(train['x'])
        record = dict(**fit_ridge(train,val,mean,scale),mean=mean,scale=scale)
        original.frozen.save_tensor(out/arm/'ridge.pt',record)
        records[arm] = record
        trials[arm] = {k:v for k,v in record.items() if k not in ('state','mean','scale')}
        print(f'Fitted {arm}; validation Brier {record["validation_brier"]:.5f}',flush=True)
    # Freeze every alpha before reading any test tensors in the fitting stage.
    write_json(out/'selection.json',dict(arms=trials,scope='Alpha selected on validation only; report all four arms.'))
    results = {}; predictions = {}; reference = None
    for arm in ARMS:
        data = torch.load(out/arm/'test.pt',weights_only=True)
        metadata = {k:data[k] for k in ('y','position','response','question')}
        if reference is None:
            reference = metadata
        if any(not torch.equal(reference[k],v) for k,v in metadata.items()):
            raise ValueError('Mismatched test prefixes')
        pred = prediction(records[arm],data)
        predictions[arm] = pred
        np.save(out/arm/'predictions.npy',pred)
        results[arm] = metrics(data['y'].numpy(),pred)
    comparisons = {f'direct_minus_{arm}':paired(predictions[ARMS[0]],predictions[arm],
        reference['y'].numpy(),reference['question'].numpy()) for arm in ARMS[1:]}
    scope = ('Full pre-softmax J-lens scores in exact compact coordinates; no vocabulary selection or sparse reconstruction. '
        'RMS normalization retained; not a linear head on softmax probabilities. '
        'All heads linear ridge with training-only centering and one isotropic scale per arm. '
        'Reused inspected MATH questions and retained verifier labels. Question intervals exploratory/unadjusted. '
        'No LSTD/PPO update, fresh benchmark or cheaper-critic claim.')
    summary = dict(scope=scope,blocks=[l+1 for l in manifest['layers']],
        dimension=manifest['dimension']*len(manifest['layers']),results=results,paired=comparisons,
        fitting=trials)
    write_json(out/'summary.json',summary)
    lines = [scope,f'Blocks: {summary["blocks"]}; complete-readout compact dimension: {summary["dimension"]}',
             'representation             Brier      accuracy']
    lines += [f'{arm:26} {v["brier"]:.5f}    {v["accuracy"]:.3f}' for arm,v in results.items()]
    lines += [f'{key}: {value}' for key,value in comparisons.items()]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')


def run(args):
    import jlens
    source = args.source.resolve(); out = args.out.resolve()
    m = json.loads((source/'manifest.json').read_text())
    if json.loads((source/'status.json').read_text())['state']!='complete':
        raise ValueError('Source geometric study must be complete')
    if original.installed_lens()!=m['environment']:
        raise ValueError('Use the original pinned J-lens environment')
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='1':
        raise ValueError('Unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); inputs={}
    def checked(path,digest=None):
        actual=sha256(path)
        if digest is not None and actual!=digest:raise ValueError(f'Changed input: {path}')
        inputs[str(path)]=actual
        return path
    checked(source/'manifest.json')
    extraction=json.loads(checked(source/'extraction.json').read_text())
    meta=json.loads(checked(source/'lens-metadata.json').read_text())
    lens=jlens.JacobianLens.load(str(checked(source/'lens.pt',meta['sha256'])))
    for path,digest in m['inputs'].items():
        if '/models/' in path:checked(Path(path),digest)
    config=json.loads(checked(original.MODEL/'config.json').read_text())
    eps=config.get('rms_norm_eps',1e-6)
    weights=load_readout_weights(original.MODEL)
    U=weights['lm_head.weight'].to(device);gain=weights['model.norm.weight'].to(device)
    del weights
    factor_record=readout_factor(U,gain)
    original.frozen.save_tensor(out/'readout-factor.pt',factor_record)
    factor=factor_record['factor'].to(device=device,dtype=torch.float64)
    write_json(out/'factorization.json',{k:v for k,v in factor_record.items() if not torch.is_tensor(v)})
    for arm in ARMS:(out/arm).mkdir()
    checks=[];counts={}
    for split in ('train','val','test'):
        def cached(arm):
            relative=f'{arm}/{split}.pt'
            return torch.load(checked(source/relative,extraction['files'][relative]),weights_only=True)
        base=cached('hidden_two_thirds')
        reference={key:base[key] for key in ('y','position','response','question')}
        counts[split]=dict(prefixes=len(base['y']),questions=len(base['question'].unique()))
        stacks={name:[] for name in ARMS[:3]}
        for layer in m['layers']:
            data=cached(f'hidden-block{layer+1}')
            if any(not torch.equal(data[k],v) for k,v in reference.items()):
                raise ValueError('Unpaired source features')
            hidden=data['x'].float();direct=[];normalized=[]
            for start in range(0,len(hidden),args.batch_size):
                h=hidden[start:start+args.batch_size].to(device)
                transported=h if layer==m['target_layer'] else lens.transport(h,layer)
                phi=compact_readout(transported.double(),factor,eps)
                if start==0:
                    x=rms_coordinates(transported[:4].double(),eps)
                    full=(x*gain.double())@U.double().T
                    actual=full@full.T;compact=phi[:4]@phi[:4].T
                    torch.testing.assert_close(compact,actual,rtol=1e-7,atol=1e-7)
                    checks.append(dict(split=split,block=layer+1,gram_relative_error=float(
                        torch.linalg.vector_norm(compact-actual)/torch.linalg.vector_norm(actual).clamp_min(1e-20))))
                direct.append(phi.float().cpu())
                normalized.append(rms_coordinates(h,eps).cpu())
            stacks['direct_jlens_concat'].append(torch.cat(direct))
            stacks['hidden_concat'].append(hidden)
            stacks['rms_hidden_concat'].append(torch.cat(normalized))
        for arm in ARMS:
            x=base['x'].float() if arm=='hidden_two_thirds' else torch.cat(stacks[arm],dim=1)
            original.frozen.save_tensor(out/arm/f'{split}.pt',dict(reference,x=x))
        print(f'Prepared {split}: {counts[split]}',flush=True)
    write_json(out/'readout-checks.json',checks)
    write_json(out/'counts.json',counts)
    del U,gain,factor
    if device.type=='cuda':torch.cuda.empty_cache()
    write_json(out/'manifest.json',dict(inputs=inputs,code=snapshot(original.ROOT)['files'],
        source=str(source),layers=m['layers'],dimension=m['dimension'],environment=m['environment'],
        feature_recipe='Full J-lens logits via untruncated vocabulary Gram factor; per-state RMS; no softmax.',
        normalization='Training mean and one scalar per arm; preserves Euclidean geometry up to a scalar.',
        config={'batch_size':args.batch_size,'device':args.device}))
    fit_and_report(out,m)
    write_json(out/'status.json',dict(state='complete',seconds=time.perf_counter()-started))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=parser.parse_args()
    if args.out.exists():parser.error('Use a new output directory')
    if args.batch_size<1:parser.error('Batch size must be positive')
    try:run(args)
    except Exception as exc:
        if args.out.exists():write_json(args.out/'status.json',dict(state='failed',error=str(exc)))
        raise


if __name__=='__main__':main()
