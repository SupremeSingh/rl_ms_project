"""Concatenate layer-specific J-space components at geometric depths from the end.

Official J-lens fitting for earlier blocks; exact identity transport at the final
block. Our sparse reconstruction and ridge heads are separate extensions.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import compare_jspace_probes as original
from refine_jspace_probes import fit_head, prediction, paired
from math_rl.critic_layers import sampled_block_states, layer_spec
from math_rl.critic_probe import metrics
from math_rl.jspace import geometric_layers, pursuit_path, rotated_dictionary, vocabulary_directions
from math_rl.provenance import sha256, snapshot, write_json


def names(layers):
    return ['jspace_concat','hidden_concat','random_concat','hidden_two_thirds'] + [
        f'{kind}-block{layer+1}' for kind in ('jspace','hidden') for layer in layers]


def select_questions(questions, limits):
    selected=[]
    for split in ('train','val','test'):
        pool=sorted((q for q in questions if q['split']==split),
            key=lambda q:hashlib.sha256(('geometric-jspace-814:'+q['id']).encode()).hexdigest())
        selected.extend(pool[:limits[split]] if limits[split] else pool)
    if any(not any(q['split']==split for q in selected) for split in limits):
        raise ValueError('Empty split')
    return selected


def initialize(source, out, config, environment):
    old=json.loads((source/'manifest.json').read_text())
    if old['environment']!=environment:raise ValueError('Use the original pinned J-lens environment')
    if json.loads((source/'status.json').read_text())['state']!='complete':
        raise ValueError('Source must be a completed original J-space study')
    calibration=old['calibration'][:config['calibration_prompts']]
    if len(calibration)!=config['calibration_prompts']:
        raise ValueError('Source does not reserve enough calibration questions')
    questions=select_questions(old['questions'],config['question_limits'])
    if {q['id'] for q in questions}&{q['id'] for q in calibration}:
        raise ValueError('Calibration/probe question overlap')
    model_config=json.loads((original.MODEL/'config.json').read_text())
    layers=geometric_layers(model_config['num_hidden_layers'],config['geometric_base'])
    if len(layers)<2:raise ValueError('Need at least two selected blocks')
    inputs={str(source/'manifest.json'):sha256(source/'manifest.json')}
    for q in questions+calibration:
        path=Path(old['source'])/'trajectories'/f'{q["original_index"]:05d}.json'
        inputs[str(path)]=old['source_hashes'][str(path)]
    for name,digest in old['model_hashes'].items():inputs[str(original.MODEL/name)]=digest
    for path,digest in inputs.items():
        if sha256(path)!=digest:raise ValueError(f'Changed input: {path}')
    manifest=dict(config=config,environment=environment,inputs=inputs,
        code=snapshot(original.ROOT)['files'],questions=questions,calibration=calibration,
        source=str(source),trajectory_source=old['source'],layers=layers,
        dimension=model_config['hidden_size'],target_layer=model_config['num_hidden_layers']-1,
        convention='All selected block outputs before final RMSNorm; concatenation final-to-earlier. '
                   'Final block has identity transport; earlier J matrices from official jlens.fit.',
        scope='New layer aggregation on existing inspected questions, saved rewards and frozen base weights; no policy updates.')
    out.mkdir(parents=True,exist_ok=False)
    for name in names(layers):(out/name/'sampled').mkdir(parents=True)
    original.frozen.atomic_json(out/'manifest.json',manifest)
    return manifest


def validate(m):
    if original.installed_lens()!=m['environment'] or snapshot(original.ROOT)['files']!=m['code']:
        raise ValueError('Code/environment changed; use a fresh output')
    for path,digest in m['inputs'].items():
        if sha256(path)!=digest:raise ValueError(f'Changed input: {path}')


def fit_lenses(out,m):
    import jlens
    model,tokenizer=original.load_model()
    adapter=jlens.from_hf(model,tokenizer,force_bos=False)
    if adapter.n_layers-1!=m['target_layer'] or adapter.d_model!=m['dimension']:
        raise ValueError('Model dimensions changed')
    prompts=[]
    for q in m['calibration']:
        rows=json.loads((Path(m['trajectory_source'])/'trajectories'/f'{q["original_index"]:05d}.json').read_text())
        if rows['question_id']!=q['id']:raise ValueError('Calibration question mismatch')
        ids=q['prompt_token_ids']+rows['responses'][0]['token_ids']
        prompts.append(tokenizer.decode(ids[:128],skip_special_tokens=False))
    original.frozen.atomic_json(out/'calibration.json',dict(
        ids=[q['id'] for q in m['calibration']],prompts=prompts,
        rule='First 128 tokens, first saved answer, reserved training questions; unchanged calibration recipe.'))
    jlens.configure_logging()
    started=time.perf_counter()
    # Upstream requires source < target. At target itself the derivative is I.
    sources=[layer for layer in m['layers'] if layer!=m['target_layer']]
    lens=jlens.fit(adapter,prompts,source_layers=sources,target_layer=m['target_layer'],
        max_seq_len=128,skip_first=16,dim_batch=m['config']['dim_batch'],
        checkpoint_path=str(out/'lens-checkpoint.pt'))
    if lens.n_prompts!=len(prompts):raise ValueError('Official fitter skipped calibration prompts')
    if any(not torch.isfinite(j).all() for j in lens.jacobians.values()):raise ValueError('Nonfinite lens')
    temporary=out/'lens.tmp.pt';lens.save(str(temporary),dtype=torch.float32);temporary.replace(out/'lens.pt')
    original.frozen.atomic_json(out/'lens-metadata.json',dict(sha256=sha256(out/'lens.pt'),
        fitted_layers=sources,identity_layer=m['target_layer'],n_prompts=lens.n_prompts,
        seconds_this_invocation=time.perf_counter()-started))


def extract(out,m):
    import jlens
    metadata=json.loads((out/'lens-metadata.json').read_text())
    if sha256(out/'lens.pt')!=metadata['sha256']:raise ValueError('Lens changed')
    model,tokenizer=original.load_model()
    lens=jlens.JacobianLens.load(str(out/'lens.pt'))
    if model.lm_head.bias is not None:raise ValueError('Expected unbiased Qwen unembedding')
    dictionaries={}
    baseline_layer=layer_spec(len(model.model.layers))['two_thirds']-1
    capture_layers=list(dict.fromkeys(m['layers']+[baseline_layer]))
    with torch.no_grad():
        for layer in m['layers']:
            J=torch.eye(m['dimension'],device='cuda') if layer==m['target_layer'] else lens.jacobians[layer].cuda()
            dictionaries[layer]=vocabulary_directions(model.lm_head.weight,model.model.norm.weight,J)
    checked=False
    for i,q in enumerate(m['questions']):
        marker=out/f'question-{i:05d}.json'
        if marker.exists():
            record=json.loads(marker.read_text())
            if any(sha256(out/path)!=digest for path,digest in record['files'].items()):
                raise ValueError('Feature shard changed')
            continue
        started=time.perf_counter()
        trajectory=json.loads((Path(m['trajectory_source'])/'trajectories'/f'{q["original_index"]:05d}.json').read_text())
        if trajectory['question_id']!=q['id']:raise ValueError('Question mismatch')
        collected={name:[] for name in names(m['layers'])}
        ys=[];ps=[];rs=[];readouts=[];attempted=retained=0;statuses={}
        # First attempts are chosen before reading their labels; failures stay counted.
        for ri,row in enumerate(trajectory['responses'][:m['config']['answers']]):
            attempted+=1
            status=row['verifier_status'];statuses[status]=statuses.get(status,0)+1
            if not original.frozen.usable_response(row):continue
            if row['score'] not in (0,1) or row['score']!=(status=='correct'):
                raise ValueError('Inconsistent binary reward')
            if 'action_mask' in row or 'context_token_ids' in row:raise ValueError('Staged responses unsupported')
            retained+=1
            length=len(row['token_ids']);prompt_length=len(q['prompt_token_ids'])
            prefix=original.positions(q['original_index'],ri,length)
            tokens=torch.tensor([q['prompt_token_ids']+row['token_ids']],device='cuda')
            captured=sampled_block_states(model.model,tokens,prompt_length,length,prefix,capture_layers)
            if not checked:
                cut=prefix[-1];changed=tokens.clone();boundary=prompt_length+cut
                changed[:,boundary:]=(changed[:,boundary:]+1)%model.config.vocab_size
                alternative=sampled_block_states(model.model,changed,prompt_length,length,[cut],capture_layers)
                differences={}
                for layer in capture_layers:
                    torch.testing.assert_close(captured[layer][-1],alternative[layer][0],atol=1e-4,rtol=1e-4)
                    differences[str(layer+1)]=float((captured[layer][-1]-alternative[layer][0]).abs().max())
                original.frozen.atomic_json(out/'causality-check.json',dict(question=q['id'],prefix=cut,
                    max_absolute_difference_by_block=differences,tf32=False,atol=1e-4,rtol=1e-4))
                checked=True
            js=[];hs=[];randoms=[]
            for layer in m['layers']:
                h=captured[layer].cuda().float()
                budget=m['config']['sparsity'];method=m['config']['pursuit']
                j,ids,weights=pursuit_path(h,dictionaries[layer],[budget],method)[budget]
                control=rotated_dictionary(dictionaries[layer],seed=913+layer)
                random,_,_=pursuit_path(h,control,[budget],method)[budget]
                del control
                j=j.cpu();random=random.cpu();hidden=h.cpu()
                js.append(j);hs.append(hidden);randoms.append(random)
                collected[f'jspace-block{layer+1}'].append(j)
                collected[f'hidden-block{layer+1}'].append(hidden)
                ids=ids.cpu();weights=weights.cpu()
                readouts.append(dict(block=layer+1,response_index=ri,prefix_lengths=prefix,
                    token_ids=ids.tolist(),coefficients=weights.tolist(),
                    tokens=[[tokenizer.decode([int(t)]) for t in line] for line in ids],
                    active_distinct_atoms=[len(line[w>1e-8].unique()) for line,w in zip(ids,weights)],
                    residual_energy_fraction=((hidden-j).square().sum(1)/hidden.square().sum(1).clamp_min(1e-12)).tolist()))
            collected['jspace_concat'].append(torch.cat(js,1))
            collected['hidden_concat'].append(torch.cat(hs,1))
            collected['random_concat'].append(torch.cat(randoms,1))
            collected['hidden_two_thirds'].append(captured[baseline_layer].float())
            ys.extend([float(row['score'])]*len(prefix));ps.extend(prefix);rs.extend([ri]*len(prefix))
        files={}
        for name,chunks in collected.items():
            dimension=m['dimension']*(len(m['layers']) if name.endswith('_concat') else 1)
            path=out/name/'sampled'/f'{i:05d}.pt'
            original.frozen.save_tensor(path,dict(x=torch.cat(chunks) if chunks else torch.empty(0,dimension),
                y=torch.tensor(ys),position=torch.tensor(ps,dtype=torch.long),
                response=torch.tensor(rs,dtype=torch.long),question=torch.full((len(ys),),q['original_index'],dtype=torch.long)))
            files[str(path.relative_to(out))]=sha256(path)
        original.frozen.atomic_json(marker,dict(id=q['id'],split=q['split'],attempted=attempted,
            retained=retained,statuses=statuses,readouts=readouts,files=files,seconds=time.perf_counter()-started))
        print(f'Extracted {i+1}/{len(m["questions"])} questions; {retained}/{attempted} answers',flush=True)
    for name in names(m['layers']):
        for split in ('train','val','test'):
            parts=[torch.load(out/name/'sampled'/f'{i:05d}.pt',weights_only=True)
                   for i,q in enumerate(m['questions']) if q['split']==split]
            if not parts or not sum(len(p['y']) for p in parts):raise ValueError(f'Empty {split}')
            original.frozen.save_tensor(out/name/f'{split}.pt',{key:torch.cat([p[key] for p in parts]) for key in parts[0]})
    original.frozen.atomic_json(out/'extraction.json',dict(files={str(path.relative_to(out)):sha256(path)
        for name in names(m['layers']) for path in (out/name).glob('*.pt')}))


def fit(out,m):
    records={};reference={}
    for name in names(m['layers']):
        data=[torch.load(out/name/f'{split}.pt',weights_only=True) for split in ('train','val')]
        for split,d in zip(('train','val'),data):
            if split not in reference:reference[split]={key:d[key] for key in ('y','question','response','position')}
            if any(not torch.equal(d[key],value) for key,value in reference[split].items()):
                raise ValueError('Unmatched examples across layer arms')
        print(f'Ridge fitting {name}: dimension {data[0]["x"].shape[1]}',flush=True)
        records[name]=fit_head(out/name/'heads',*data)
    selected={kind:min((f'{kind}-block{layer+1}' for layer in m['layers']),
        key=lambda name:records[name]['validation_brier']) for kind in ('jspace','hidden')}
    selection=dict(best_single=selected,validation={name:dict(alpha=r['alpha'],brier=r['validation_brier'])
        for name,r in records.items()},concatenation_blocks=[layer+1 for layer in m['layers']],
        scope='Geometric pattern and pursuit fixed in advance; best individual layers selected on validation only.')
    original.frozen.atomic_json(out/'selection.json',selection)
    chosen=['jspace_concat','hidden_concat','random_concat','hidden_two_thirds',selected['jspace'],selected['hidden']]
    results={};predictions={};reference=None
    for name in chosen:
        data=torch.load(out/name/'test.pt',weights_only=True)
        if reference is None:reference={key:data[key] for key in ('y','question','response','position')}
        if any(not torch.equal(data[key],value) for key,value in reference.items()):raise ValueError('Unpaired test data')
        predictions[name]=prediction(records[name],data)
        np.save(out/name/'predictions.npy',predictions[name])
        results[name]=metrics(data['y'].numpy(),predictions[name])
    comparisons={f'{a}_minus_{b}':paired(predictions[a],predictions[b],reference['y'].numpy(),reference['question'].numpy())
        for a,b in [('jspace_concat','hidden_concat'),('jspace_concat','random_concat'),
                    ('jspace_concat',selected['jspace']),('jspace_concat',selected['hidden']),
                    ('jspace_concat','hidden_two_thirds')]}
    counts={split:dict(attempted=0,retained=0,prefixes=0,questions=0) for split in ('train','val','test')}
    diagnostics={str(layer+1):[] for layer in m['layers']}
    seconds=0.;review=['<!doctype html><meta charset="utf-8"><title>Geometric J-space readouts</title>',
        '<h1>Layer-specific concept readouts</h1><p>First five test questions; first retained answer. '
        'Readouts are approximate reconstruction coefficients, not explanations of correctness.</p>']
    shown=0
    for i,q in enumerate(m['questions']):
        item=json.loads((out/f'question-{i:05d}.json').read_text());seconds+=item['seconds']
        for key in ('attempted','retained'):counts[q['split']][key]+=item[key]
        counts[q['split']]['questions']+=1
        for r in item['readouts']:
            diagnostics[str(r['block'])].extend(zip(r['residual_energy_fraction'],r['active_distinct_atoms']))
            if r['block']==m['layers'][0]+1:counts[q['split']]['prefixes']+=len(r['prefix_lengths'])
        if q['split']=='test' and item['readouts'] and shown<5:
            ri=item['readouts'][0]['response_index']
            path=Path(m['trajectory_source'])/'trajectories'/f'{q["original_index"]:05d}.json'
            row=json.loads(path.read_text())['responses'][ri]
            review.append('<details><summary>'+html.escape(q['id'])+'</summary><pre style="white-space:pre-wrap">'+
                html.escape(q['question']+'\n'+row['response']+'\n'+json.dumps(
                    [r for r in item['readouts'] if r['response_index']==ri],indent=2,ensure_ascii=False))+'</pre></details>')
            shown+=1
    diagnostics={layer:dict(residual_energy_fraction_mean=float(np.mean(values,axis=0)[0]),
        active_distinct_atoms_mean=float(np.mean(values,axis=0)[1])) for layer,values in diagnostics.items()}
    scope=('Frozen base actor, saved MATH answers, inspected questions and retained verifier labels. '
           'Layer-specific official lenses plus our sparse reconstruction; final transport is identity. '
           'Ridge return regression, not LSTD/PPO. Paired question intervals exploratory/unadjusted. '
           'Concatenated features are not claimed cheap without measuring total cost.')
    summary=dict(results=results,paired=comparisons,counts=counts,selection=selection,
        dimension=m['dimension']*len(m['layers']),config=m['config'],scope=scope,
        diagnostics=diagnostics,extraction_shard_seconds=seconds,
        fitting={name:{k:v for k,v in record.items() if k not in ('state','mean','scale')}
                 for name,record in records.items()})
    original.frozen.atomic_json(out/'summary.json',summary)
    lines=[scope,'Blocks (final first): '+str([layer+1 for layer in m['layers']]),
        f'Concatenated dimension: {summary["dimension"]}',f'Counts: {counts}',
        'representation               Brier       accuracy']
    lines += [f'{name:28} {r["brier"]:.5f}     {r["accuracy"]:.3f}' for name,r in results.items()]
    lines += [f'{name}: {result}' for name,result in comparisons.items()]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    (out/'responses.html').write_text('\n'.join(review))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path,nargs='?',help='Completed original J-space study')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--geometric-base',type=int,default=2)
    parser.add_argument('--sparsity',type=int,default=25)
    parser.add_argument('--pursuit',choices=['matching','gradient'],default='gradient')
    parser.add_argument('--train-questions',type=int,default=400)
    parser.add_argument('--val-questions',type=int,default=100)
    parser.add_argument('--answers',type=int,default=4)
    parser.add_argument('--calibration-prompts',type=int,default=100)
    parser.add_argument('--dim-batch',type=int,default=2)
    parser.add_argument('--stage',choices=['lens','extract','fit'])
    args=parser.parse_args();out=args.out.resolve()
    if args.stage:
        m=json.loads((out/'manifest.json').read_text())
        {'lens':fit_lenses,'extract':extract,'fit':fit}[args.stage](out,m)
        return
    if args.geometric_base<2 or min(args.sparsity,args.train_questions,args.val_questions,args.answers,args.calibration_prompts,args.dim_batch)<1:
        parser.error('Positive budgets and geometric base >= 2 required')
    config=dict(geometric_base=args.geometric_base,sparsity=3 if args.smoke else args.sparsity,
        pursuit=args.pursuit,question_limits=dict(train=4 if args.smoke else args.train_questions,
        val=3 if args.smoke else args.val_questions,test=3 if args.smoke else 0),
        answers=2 if args.smoke else args.answers,calibration_prompts=2 if args.smoke else args.calibration_prompts,
        dim_batch=args.dim_batch,smoke=args.smoke)
    if out.exists():
        m=json.loads((out/'manifest.json').read_text())
        if m['config']!=config or (args.source and str(args.source.resolve())!=m['source']):
            raise ValueError('Resume settings changed')
        validate(m)
    else:
        if args.source is None:parser.error('Source required for a new study')
        m=initialize(args.source.resolve(),out,config,original.installed_lens())
    started=time.perf_counter();state=dict(state='running',slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    try:
        for stage in ('lens','extract','fit'):
            marker=out/('lens-metadata.json' if stage=='lens' else 'extraction.json' if stage=='extract' else 'summary.json')
            if marker.exists() and stage!='fit':
                record=json.loads(marker.read_text())
                files={'lens.pt':record['sha256']} if stage=='lens' else record['files']
                if any(sha256(out/path)!=digest for path,digest in files.items()):raise ValueError('Cached artifacts changed')
                continue
            state['stage']=stage;original.frozen.atomic_json(out/'status.json',state)
            with (out/f'{stage}.log').open('a') as log:
                subprocess.run([sys.executable,__file__,'--out',str(out),'--stage',stage],stdout=log,stderr=subprocess.STDOUT,check=True)
        state['state']='complete'
    except BaseException as exc:
        state.update(state='failed',error=str(exc));raise
    finally:
        state['invocation_seconds']=time.perf_counter()-started
        original.frozen.atomic_json(out/'status.json',state)
        with (out/'invocations.jsonl').open('a') as handle:handle.write(json.dumps(state)+'\n')


if __name__=='__main__':main()
