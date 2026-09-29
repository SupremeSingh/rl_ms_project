"""Reuse saved frozen-policy answers; compare matched heads at three causal depths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

import frozen_critics as frozen
from math_rl.critic_layers import layer_spec, sampled_layer_states
from math_rl.critic_probe import HEADS, PREFIX_SAMPLING
from math_rl.provenance import sha256, snapshot, write_json

LAYERS = ('third', 'two_thirds', 'final')


def positions(question_index, response_index, length):
    rng=np.random.default_rng(np.random.SeedSequence([PREFIX_SAMPLING['seed'],question_index,response_index]))
    return [0]+(rng.integers(1,length,size=7).tolist() if length>1 else [0]*7)


def initialize(source, out, seeds, epochs):
    source=source.resolve()
    if (source/'data/questions.json').exists():
        source=source/'data'
    saved=json.loads((source/'manifest.json').read_text())
    questions=json.loads((source/'questions.json').read_text())['questions']
    if saved['questions_sha256']!=sha256(source/'questions.json'):
        raise ValueError('Source question snapshot changed')
    if saved['config'].get('multistage'):
        raise ValueError('Use ordinary completion trajectories, not staged planning')
    model_hashes={}
    for path,digest in saved['files'].items():
        if '/models/qwen-math/' in path:
            local=frozen.ROOT/'models/qwen-math'/Path(path).name
            if sha256(local)!=digest:
                raise ValueError('Model differs from the generating base model')
            model_hashes[local.name]=digest
    if not any(p.endswith('.safetensors') for p in model_hashes):
        raise ValueError('Missing generating-model weight provenance')
    paths=[source/'trajectories'/f'{i:05d}.json' for i in range(len(questions))]
    hashes={str(p):sha256(p) for p in paths}
    config=dict(saved['config'],seeds=seeds,epochs=epochs,min_epochs=min(30,epochs),
                patience=30,learning_rates=[1e-2,1e-3,1e-4])
    manifest=dict(protocol='critic-depth-v1',source=str(source),questions=questions,config=config,
        source_manifest_sha256=sha256(source/'manifest.json'),source_hashes=hashes,model_hashes=model_hashes,
        normalization='Train-split coordinate standardization at each depth; intermediate block output versus final RMSNorm.',
        prefix_sampling=PREFIX_SAMPLING,provenance=snapshot(frozen.ROOT))
    out.mkdir(parents=True,exist_ok=False)
    for name in LAYERS:
        folder=out/name;folder.mkdir();(folder/'heads').mkdir();(folder/'sampled').mkdir()
        (folder/'trajectories').symlink_to(source/'trajectories',target_is_directory=True)
    write_json(out/'manifest.json',manifest)
    return manifest


def extract(out, manifest):
    from transformers import AutoModel
    if not torch.cuda.is_available():
        raise RuntimeError('Layer extraction needs the cluster GPU')
    model=AutoModel.from_pretrained(frozen.ROOT/'models/qwen-math',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    write_json(out/'layers.json',dict(block_count=len(model.layers),depths=layer_spec(len(model.layers)),
        feature_dimension=model.config.hidden_size,
        note='One-based completed block counts; final includes the existing learned final RMSNorm.'))
    for i,q in enumerate(manifest['questions']):
        targets={name:out/name/'sampled'/f'{i:05d}.pt' for name in LAYERS}
        if all(p.exists() for p in targets.values()):
            continue
        rows=json.loads((Path(manifest['source'])/'trajectories'/f'{i:05d}.json').read_text())
        if rows['question_id']!=q['id']:
            raise ValueError('Trajectory/question mismatch')
        xs={name:[] for name in LAYERS};ys=[];ps=[];rs=[]
        for ri,row in enumerate(rows['responses']):
            if not frozen.usable_response(row):
                continue
            if 'action_mask' in row or 'context_token_ids' in row:
                raise ValueError('Controlled/staged trajectories are not supported')
            length=len(row['token_ids']);prefix=positions(i,ri,length)
            tokens=torch.tensor([q['prompt_token_ids']+row['token_ids']],device='cuda')
            captured=sampled_layer_states(model,tokens,len(q['prompt_token_ids']),length,prefix)
            for name in LAYERS:xs[name].append(captured[name])
            ys += [float(row['score'])]*len(prefix);ps+=prefix;rs += [ri]*len(prefix)
        for name,path in targets.items():
            frozen.save_tensor(path,dict(x=torch.cat(xs[name]) if xs[name] else torch.empty((0,model.config.hidden_size),dtype=torch.bfloat16),
                y=torch.tensor(ys),position=torch.tensor(ps,dtype=torch.long),response=torch.tensor(rs,dtype=torch.long),
                question=torch.full((len(ys),),i,dtype=torch.long)))
        print(f'Extracted matched layers: {i+1}/{len(manifest["questions"])}',flush=True)
    for name in LAYERS:
        for split in ('train','val','test'):
            parts=[torch.load(out/name/'sampled'/f'{i:05d}.pt',weights_only=True) for i,q in enumerate(manifest['questions']) if q['split']==split]
            if not parts or not sum(len(p['y']) for p in parts):
                raise ValueError(f'No usable {split} answers')
            frozen.save_tensor(out/name/f'{split}.pt',{k:torch.cat([p[k] for p in parts]) for k in parts[0]})


def summarize(out, manifest):
    summaries={name:json.loads((out/name/'summary.json').read_text()) for name in LAYERS}
    data={name:torch.load(out/name/'test.pt',weights_only=True) for name in LAYERS}
    for name in LAYERS:
        for key in ('question','response','position','y'):
            if not torch.equal(data[name][key],data['final'][key]):
                raise ValueError('Unpaired layer evaluation')
    y=data['final']['y'].numpy();question=data['final']['question'].numpy()
    results={};paired={};lines=['Frozen base-model depth comparison; same answers, prefixes, labels and fitting budgets.',
        'Layer depth includes extraction convention: intermediate block output; final learned RMSNorm.',
        'layer/head             Brier       accuracy']
    for kind in list(HEADS)+['ridge_value']:
        errors={}
        for name in LAYERS:
            rows=[summaries[name]['ridge']] if kind=='ridge_value' else [r for r in summaries[name]['heads'] if r['kind']==kind]
            brier=float(np.mean([r['test']['brier'] for r in rows]));accuracy=float(np.mean([r['test']['accuracy'] for r in rows]))
            files=[out/name/'heads/ridge_value-predictions.npy'] if kind=='ridge_value' else [out/name/'heads'/f'{kind}-{seed}-predictions.npy' for seed in manifest['config']['seeds']]
            errors[name]=np.mean([(np.load(p)-y)**2 for p in files],axis=0)
            results[f'{name}/{kind}']=dict(brier=brier,accuracy=accuracy)
            lines.append(f'{name+"/"+kind:27} {brier:.5f}     {accuracy:.3f}')
        for name in ('third','two_thirds'):
            delta=errors[name]-errors['final']
            qdelta=np.array([delta[question==q].mean() for q in np.unique(question)])
            rng=np.random.default_rng(912)
            draws=[float(rng.choice(qdelta,size=len(qdelta),replace=True).mean()) for _ in range(2000)]
            paired[f'{name}_minus_final/{kind}']=dict(question_mean_brier_difference=float(qdelta.mean()),
                descriptive_95_interval=np.quantile(draws,[.025,.975]).tolist())
    scope='Conditional on verifiable answers; inspected questions; exploratory unadjusted question intervals. Head seeds share data. No PPO or LSTD learning claim.'
    write_json(out/'summary.json',dict(results=results,paired=paired,scope=scope))
    lines += [f'{key}: {value}' for key,value in paired.items()]+[scope]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path,nargs='?');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['extract','fit']);p.add_argument('--layer',choices=LAYERS)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]);p.add_argument('--epochs',type=int,default=200)
    args=p.parse_args();out=args.out.resolve()
    if args.stage:
        m=json.loads((out/'manifest.json').read_text())
        if args.stage=='extract':extract(out,m)
        else:
            if not args.layer:p.error('--layer required for fitting')
            frozen.fit(out/args.layer,m['config'],m['questions'])
        return
    if args.epochs<1 or len(set(args.seeds))!=len(args.seeds):p.error('Invalid fitting budget/seeds')
    if out.exists():
        m=json.loads((out/'manifest.json').read_text())
        if m['config']['seeds']!=args.seeds or m['config']['epochs']!=args.epochs:raise ValueError('Resume config differs')
        for path,digest in m['source_hashes'].items():
            if sha256(path)!=digest:raise ValueError('Source trajectories changed')
        for name,digest in m['model_hashes'].items():
            if sha256(frozen.ROOT/'models/qwen-math'/name)!=digest:raise ValueError('Base model changed')
        old=m['provenance']['files'];current=snapshot(frozen.ROOT)['files']
        if old!=current:raise ValueError('Code/config changed; use a fresh output directory')
    else:
        if args.source is None:p.error('Source directory required')
        m=initialize(args.source,out,args.seeds,args.epochs)
    state=dict(state='extracting');write_json(out/'status.json',state);started=time.perf_counter()
    try:
        subprocess.run([sys.executable,__file__,'--out',str(out),'--stage','extract'],check=True)
        for name in LAYERS:
            state.update(state='fitting',layer=name);write_json(out/'status.json',state)
            subprocess.run([sys.executable,__file__,'--out',str(out),'--stage','fit','--layer',name],check=True)
        summarize(out,m);state.update(state='complete')
    except BaseException as exc:
        state.update(state='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        state['invocation_seconds']=time.perf_counter()-started;write_json(out/'status.json',state)

if __name__=='__main__':main()
