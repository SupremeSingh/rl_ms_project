"""Official J-lens + our sparse reconstruction: matched offline MATH value probes.

No generation, rescoring, LSTD or policy updates. Source data are read-only.
Calibration uses reserved TRAIN questions and their first saved continuation,
selected without rewards. Validation/test questions never fit the lens.
"""
import argparse
from collections import Counter
from importlib import metadata
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
import frozen_critics as frozen
from compare_critic_layers import positions
from math_rl.critic_layers import sampled_layer_states, layer_spec
from math_rl.critic_probe import PREFIX_SAMPLING
from math_rl.jspace import (JLENS_COMMIT, REPRESENTATIONS, reconstruct,
                           representations, rotated_dictionary, vocabulary_directions)
from math_rl.provenance import sha256, snapshot, write_json
from math_rl.reference_audit import reference_decision

ROOT = frozen.ROOT
MODEL = ROOT / 'models/qwen-math'


def installed_lens():
    """Reject an unpinned/nonofficial install before spending GPU time."""
    dist = metadata.distribution('jlens')
    direct = json.loads(dist.read_text('direct_url.json') or '{}')
    if (direct.get('vcs_info', {}).get('commit_id') != JLENS_COMMIT
            or direct.get('url', '').removesuffix('.git') != 'https://github.com/anthropics/jacobian-lens'):
        raise ValueError('Install the pinned official library with scripts/setup_jlens.sh')
    return {name: metadata.version(name) for name in
            ('jlens', 'torch', 'transformers', 'numpy', 'huggingface-hub', 'tokenizers')}


def eligible_questions(questions):
    ids, wording, selected, audit = set(), set(), [], []
    for i, q in enumerate(questions):
        text = ' '.join(q['question'].split())
        if q['id'] in ids or text in wording:
            raise ValueError('Duplicate question IDs/text: split isolation is not guaranteed')
        ids.add(q['id']); wording.add(text)
        if q['split'] not in ('train', 'val', 'test') or q.get('level') not in (4, 5):
            raise ValueError('Expected saved MATH level 4/5 splits')
        if 'reference_solution' not in q:
            raise ValueError('Complete original reference solutions required')
        decision = reference_decision(q['question'], q['reference_solution'], q['ground_truth'])
        audit.append(dict(id=q['id'], split=q['split'], **decision))
        if decision['eligible']:
            selected.append(dict(q, original_index=i))
    return selected, audit


def select_questions(questions, config):
    eligible, audit = eligible_questions(questions)
    train = sorted((q for q in eligible if q['split']=='train'),
                   key=lambda q: hashlib.sha256(('jlens-calibration-814:'+q['id']).encode()).hexdigest())
    n = config['calibration_prompts']
    if len(train) <= n:
        raise ValueError('Not enough training questions after reference filtering')
    calibration = train[:n]
    reserved = {q['id'] for q in calibration}
    selected = []
    for split in ('train', 'val', 'test'):
        pool = [q for q in eligible if q['split']==split and q['id'] not in reserved]
        if config['smoke']:
            pool = pool[:{'train':12, 'val':8, 'test':8}[split]]
        if not pool:
            raise ValueError(f'Empty {split}')
        selected.extend(pool)
    return selected, calibration, audit


def initialize(source, out, config, environment):
    if (source/'data/questions.json').exists(): source = source/'data'
    source = source.resolve()
    saved = json.loads((source/'manifest.json').read_text())
    if sha256(source/'questions.json') != saved['questions_sha256']:
        raise ValueError('Source question hash mismatch')
    if saved['config'].get('multistage'):
        raise ValueError('Use the ordinary MATH fitting run, not planning data')
    questions, calibration, audit = select_questions(
        json.loads((source/'questions.json').read_text())['questions'], config)
    model_hashes = {}
    for path, digest in saved['files'].items():
        if '/models/qwen-math/' in path:
            local = MODEL/Path(path).name
            if sha256(local) != digest: raise ValueError('Generating base model has changed')
            model_hashes[local.name] = digest
    if not any(name.endswith('.safetensors') for name in model_hashes):
        raise ValueError('Missing source model weight provenance')
    # Also lock tokenizer/config files not recorded in older manifests.
    for p in MODEL.iterdir():
        if p.is_file() and p.suffix in ('.json', '.txt', '.model', '.safetensors'):
            model_hashes[p.name] = sha256(p)
    sources = {str(source/name):sha256(source/name) for name in ('manifest.json', 'questions.json')}
    for q in questions+calibration:
        path = source/'trajectories'/f'{q["original_index"]:05d}.json'
        sources[str(path)] = sha256(path)
    manifest = dict(protocol='jspace-probes-v1', source=str(source), config=config,
        questions=questions, calibration=calibration, source_hashes=sources,
        model_hashes=model_hashes, environment=environment, jlens_commit=JLENS_COMMIT,
        provenance=snapshot(ROOT), prefix_sampling=PREFIX_SAMPLING,
        calibration_scope='Reserved training questions + first saved continuation, reward-blind; math-domain lens, not generic-web replication.',
        reconstruction='Our nonnegative matching pursuit, at most k atoms; not official sparse decomposition or an exact projection.',
        selection_scope='Same inspected splits; strict reference eligibility; saved verifier labels unchanged.')
    out.mkdir(parents=True, exist_ok=False)
    for name in REPRESENTATIONS:
        folder = out/name
        for sub in ('heads', 'sampled', 'trajectories'): (folder/sub).mkdir(parents=True)
        for i,q in enumerate(questions):
            (folder/'trajectories'/f'{i:05d}.json').symlink_to(source/'trajectories'/f'{q["original_index"]:05d}.json')
    write_json(out/'reference-audit.json', audit)
    frozen.atomic_json(out/'manifest.json', manifest)
    return manifest


def validate(manifest, environment):
    if environment != manifest['environment']: raise ValueError('Environment changed; use a new output')
    if snapshot(ROOT)['files'] != manifest['provenance']['files']:
        raise ValueError('Code/config changed; use a new output')
    for path,digest in manifest['source_hashes'].items():
        if sha256(path) != digest: raise ValueError(f'Source changed: {path}')
    for name,digest in manifest['model_hashes'].items():
        if sha256(MODEL/name) != digest: raise ValueError('Base model changed')


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError('Run extraction on the cluster GPU')
    # No Flash Attention dependency; float32 for derivative accuracy and parity.
    hf = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
        torch_dtype=torch.float32, attn_implementation='eager').cuda().eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    return hf, tok


def fit_lens(out, m):
    import jlens
    hf, tok = load_model()
    adapter = jlens.from_hf(hf, tok, force_bos=False)
    source_layer = layer_spec(adapter.n_layers)['two_thirds']-1
    prompts = []
    for q in m['calibration']:
        data = json.loads((Path(m['source'])/'trajectories'/f'{q["original_index"]:05d}.json').read_text())
        if data['question_id'] != q['id']: raise ValueError('Calibration question mismatch')
        # First attempt, never selected by success or verifier status.
        ids = q['prompt_token_ids']+data['responses'][0]['token_ids']
        prompts.append(tok.decode(ids[:128], skip_special_tokens=False))
    frozen.atomic_json(out/'calibration.json', dict(ids=[q['id'] for q in m['calibration']], prompts=prompts))
    jlens.configure_logging()
    started = time.perf_counter()
    lens = jlens.fit(adapter, prompts=prompts, source_layers=[source_layer],
        target_layer=adapter.n_layers-1, dim_batch=m['config']['dim_batch'],
        max_seq_len=128, skip_first=16, checkpoint_path=str(out/'lens-checkpoint.pt'))
    if lens.n_prompts != len(prompts):
        raise ValueError('Official fitter skipped calibration prompts; inspect lens.log before proceeding')
    if not torch.isfinite(lens.jacobians[source_layer]).all(): raise ValueError('Nonfinite lens')
    temp = out/'lens.tmp.pt'; lens.save(str(temp), dtype=torch.float32); temp.replace(out/'lens.pt')
    frozen.atomic_json(out/'lens-metadata.json', dict(sha256=sha256(out/'lens.pt'),
        source_layer_zero_based=source_layer, completed_blocks=source_layer+1,
        target_layer_zero_based=adapter.n_layers-1, dimension=adapter.d_model,
        n_prompts=lens.n_prompts, seconds_this_invocation=time.perf_counter()-started,
        convention='Block outputs before final RMSNorm; current+future target sum, source-position mean; skip first16 and last position.'))


def extract(out, m):
    import jlens
    info = json.loads((out/'lens-metadata.json').read_text())
    if sha256(out/'lens.pt') != info['sha256']: raise ValueError('Lens changed')
    hf, tok = load_model()
    lens = jlens.JacobianLens.load(str(out/'lens.pt'))
    layer = info['source_layer_zero_based']
    if layer != layer_spec(len(hf.model.layers))['two_thirds']-1: raise ValueError('Layer mismatch')
    J = lens.jacobians[layer].cuda()
    if hf.lm_head.bias is not None: raise ValueError('Expected Qwen unbiased unembedding')
    with torch.no_grad():
        dictionary = vocabulary_directions(hf.lm_head.weight, hf.model.norm.weight, J)
        control = rotated_dictionary(dictionary)
    checked = False
    for i,q in enumerate(m['questions']):
        done = out/f'question-{i:05d}.json'
        if done.exists():
            previous=json.loads(done.read_text())
            if any(sha256(out/path)!=digest for path,digest in previous['files'].items()):
                raise ValueError('Cached feature shard changed')
            continue
        rows = json.loads((out/'final/trajectories'/f'{i:05d}.json').read_text())
        if rows['question_id'] != q['id']: raise ValueError('Question/trajectory mismatch')
        xs={name:[] for name in REPRESENTATIONS}; ys=[];ps=[];rs=[];reviews=[]
        statuses=Counter(); attempted=retained=0
        started=time.perf_counter()
        for ri,row in enumerate(rows['responses']):
            if m['config']['smoke'] and ri>=2: break
            attempted+=1; statuses[row['verifier_status']]+=1
            if not frozen.usable_response(row): continue
            if row['score'] not in (0,1) or row['score'] != (row['verifier_status']=='correct'):
                raise ValueError('Inconsistent saved reward')
            if 'action_mask' in row or 'context_token_ids' in row: raise ValueError('Staged responses unsupported')
            retained+=1
            length=len(row['token_ids']);prefix=positions(q['original_index'],ri,length)
            tokens=torch.tensor([q['prompt_token_ids']+row['token_ids']],device='cuda')
            captured=sampled_layer_states(hf.model,tokens,len(q['prompt_token_ids']),length,prefix)
            h=captured['two_thirds'].cuda().float();final=captured['final'].cuda().float()
            if not checked:
                # The sampled pre-action vector must not depend on future tokens.
                cut=prefix[1]; short=tokens[:,:len(q['prompt_token_ids'])+cut+1]
                causal=sampled_layer_states(hf.model,short,len(q['prompt_token_ids']),cut+1,[cut])
                for name in ('two_thirds','final'):
                    torch.testing.assert_close(captured[name][1],causal[name][0],rtol=1e-4,atol=1e-4)
                checked=True
            j, ids, coeff = reconstruct(h,dictionary,m['config']['sparsity'])
            rand,_,_=reconstruct(h,control,m['config']['sparsity'])
            features=representations(h,final,j,rand)
            for name in REPRESENTATIONS: xs[name].append(features[name].cpu())
            ys += [float(row['score'])]*len(prefix);ps+=prefix;rs += [ri]*len(prefix)
            # All sampled sparse coordinates remain inspectable, including token identities.
            reviews.append(dict(response_index=ri, prefix_lengths=prefix,
                token_ids=ids.cpu().tolist(), coefficients=coeff.cpu().tolist(),
                tokens=[[tok.decode([t]) for t in line] for line in ids.cpu().tolist()],
                residual_energy_fraction=((h-j).square().sum(1)/h.square().sum(1).clamp_min(1e-12)).cpu().tolist(),
                random_residual_energy_fraction=((h-rand).square().sum(1)/h.square().sum(1).clamp_min(1e-12)).cpu().tolist()))
        hashes={}
        for name in REPRESENTATIONS:
            path=out/name/'sampled'/f'{i:05d}.pt'
            dim=info['dimension']*(2 if name.endswith('_final') else 1)
            frozen.save_tensor(path,dict(x=torch.cat(xs[name]) if xs[name] else torch.empty(0,dim),
                y=torch.tensor(ys),position=torch.tensor(ps,dtype=torch.long),response=torch.tensor(rs,dtype=torch.long),
                question=torch.full((len(ys),),q['original_index'],dtype=torch.long)))
            hashes[str(path.relative_to(out))]=sha256(path)
        frozen.atomic_json(done,dict(id=q['id'],split=q['split'],attempted=attempted,retained=retained,
            statuses=dict(statuses),seconds=time.perf_counter()-started,files=hashes,readouts=reviews))
        print(f'Extracted {i+1}/{len(m["questions"])} questions; {retained}/{attempted} answers retained',flush=True)
    for name in REPRESENTATIONS:
        for split in ('train','val','test'):
            parts=[torch.load(out/name/'sampled'/f'{i:05d}.pt',weights_only=True) for i,q in enumerate(m['questions']) if q['split']==split]
            if not parts or not sum(len(p['y']) for p in parts): raise ValueError(f'Empty retained {split}')
            frozen.save_tensor(out/name/f'{split}.pt',{k:torch.cat([p[k] for p in parts]) for k in parts[0]})
    frozen.atomic_json(out/'extraction.json',dict(complete=True,
        files={str(p.relative_to(out)):sha256(p) for name in REPRESENTATIONS for p in (out/name).glob('*.pt')}))


def summarize(out, m):
    data=torch.load(out/'final/test.pt',weights_only=True)
    y=data['y'].numpy(); q=data['question'].numpy(); kinds=m['config']['heads']+['ridge_value']
    summaries={name:json.loads((out/name/'summary.json').read_text()) for name in REPRESENTATIONS}
    errors={};results={}
    lines=['Offline J-space probe comparison; frozen base actor, saved MATH answers.',
           'Official jlens.fit; our approximate nonnegative matching pursuit. No LSTD/PPO training.',
           'representation/head              Brier    accuracy   cross entropy']
    for name in REPRESENTATIONS:
        other=torch.load(out/name/'test.pt',weights_only=True)
        for key in ('y','question','position','response'):
            if not torch.equal(data[key],other[key]):raise ValueError('Unpaired examples')
        for kind in kinds:
            rows=[summaries[name]['ridge']] if kind=='ridge_value' else [r for r in summaries[name]['heads'] if r['kind']==kind]
            predpaths=[out/name/'heads/ridge_value-predictions.npy'] if kind=='ridge_value' else [out/name/'heads'/f'{kind}-{s}-predictions.npy' for s in m['config']['seeds']]
            predictions=[np.load(path) for path in predpaths]
            errors[name,kind]=np.mean([(p-y)**2 for p in predictions],0)
            result={metric:float(np.mean([r['test'][metric] for r in rows])) for metric in ('brier','accuracy','cross_entropy')}
            results[name+'/'+kind]=result
            lines.append(f'{name+"/"+kind:33} {result["brier"]:.5f}  {result["accuracy"]:.3f}      {result["cross_entropy"]:.5f}')
    pairs=[('jspace','two_thirds'),('jspace_final','final'),('jspace_final','two_thirds_final'),
           ('jspace','random'),('jspace_final','random_final'),('residual','two_thirds')]
    comparisons={}
    for kind in kinds:
        for left,right in pairs:
            delta=errors[left,kind]-errors[right,kind]
            dq=np.array([delta[q==qid].mean() for qid in np.unique(q)])
            rng=np.random.default_rng(912)
            interval=np.quantile([rng.choice(dq,len(dq),replace=True).mean() for _ in range(2000)],[.025,.975]).tolist()
            comparisons[f'{left}_minus_{right}/{kind}']=dict(mean=float(dq.mean()),interval_95=interval)
    counts=Counter(); statuses=Counter(); seconds=0
    for i in range(len(m['questions'])):
        record=json.loads((out/f'question-{i:05d}.json').read_text())
        counts['attempted']+=record['attempted'];counts['retained']+=record['retained']
        statuses.update(record['statuses']);seconds+=record['seconds']
    scope='Inspected questions; conditional on retained verifier labels. Paired question intervals are exploratory/unadjusted. Head seeds share data. No causal, PPO or cheaper-inference claim.'
    lines += ['',f'Answers: {dict(counts)}; extraction shard seconds: {seconds:.1f}',
              'Paired Brier differences (negative favors first):']+[f'{k}: {v}' for k,v in comparisons.items()]+[scope,
              'Per-probe summary.json includes calibration, AUROC, prefix metrics, parameters and fitting costs.',
              'question-*.json includes concept tokens/weights; original full responses are linked under final/trajectories/.']
    frozen.atomic_json(out/'summary.json',dict(results=results,paired=comparisons,answers=dict(counts),
        statuses=dict(statuses),extraction_seconds=seconds,scope=scope,smoke=m['config']['smoke']))
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    review=['<!doctype html><meta charset="utf-8"><title>J-space response review</title>',
            '<style>body{max-width:1000px;margin:30px auto;font:16px system-ui}pre{white-space:pre-wrap}details{margin:20px 0}</style>',
            '<h1>J-space response review</h1><p>First five retained test questions, first two retained answers each. '
            'Selection is outcome-independent. Concept coefficients are our approximate reconstruction, not correctness judgments.</p>']
    shown=0
    for i,question in enumerate(m['questions']):
        if question['split']!='test':continue
        record=json.loads((out/f'question-{i:05d}.json').read_text())
        if not record['readouts']:continue
        rows=json.loads((out/'final/trajectories'/f'{i:05d}.json').read_text())['responses']
        for item in record['readouts'][:2]:
            row=rows[item['response_index']]
            review.append('<details><summary>'+html.escape(question['id']+f" / answer {item['response_index']}")+'</summary><pre>'+html.escape(
                'QUESTION: '+question['question']+'\nREFERENCE: '+str(question['ground_truth'])+
                '\nSAVED STATUS: '+row['verifier_status']+'\nRESPONSE:\n'+row['response'])+'</pre>')
            review.append('<pre>'+html.escape(json.dumps(item,indent=2,ensure_ascii=False))+'</pre></details>')
        shown+=1
        if shown==5:break
    (out/'responses.html').write_text('\n'.join(review))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path,nargs='?');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--calibration-prompts',type=int)
    p.add_argument('--sparsity',type=int,default=16);p.add_argument('--dim-batch',type=int,default=2)
    p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--epochs',type=int)
    p.add_argument('--heads',nargs='+',choices=['linear_logit','mlp2','resnet10'],default=['linear_logit','mlp2'])
    p.add_argument('--stage',choices=['lens','extract','fit']);p.add_argument('--representation',choices=REPRESENTATIONS)
    args=p.parse_args();out=args.out.resolve();environment=installed_lens()
    if args.stage:
        m=json.loads((out/'manifest.json').read_text())
        if args.stage=='lens':fit_lens(out,m)
        elif args.stage=='extract':extract(out,m)
        else:
            if args.representation is None:p.error('--representation required')
            frozen.fit(out/args.representation,m['config'],m['questions'])
        return
    config=dict(smoke=args.smoke,calibration_prompts=(2 if args.smoke else 100) if args.calibration_prompts is None else args.calibration_prompts,
        sparsity=args.sparsity,dim_batch=args.dim_batch,seeds=args.seeds or ([42] if args.smoke else [42,43,44]),
        epochs=(2 if args.smoke else 200) if args.epochs is None else args.epochs,heads=args.heads,
        learning_rates=[1e-3] if args.smoke else [1e-2,1e-3,1e-4],patience=30)
    config['min_epochs']=min(30,config['epochs'])
    config['response_limit']=2 if args.smoke else None
    if min(config['calibration_prompts'],args.sparsity,args.dim_batch,config['epochs'])<1:p.error('Positive budgets required')
    if len(set(config['seeds']))!=len(config['seeds']):p.error('Duplicate seeds')
    if len(set(args.heads))!=len(args.heads):p.error('Duplicate heads')
    if out.exists():
        m=json.loads((out/'manifest.json').read_text())
        if m['config']!=config:raise ValueError('Resume settings changed')
        if args.source is not None:
            source=args.source/'data' if (args.source/'data/questions.json').exists() else args.source
            if str(source.resolve())!=m['source']:raise ValueError('Resume source changed')
        validate(m,environment)
    else:
        if args.source is None:p.error('Source required for a new study')
        m=initialize(args.source,out,config,environment)
    started=time.perf_counter()
    state={'state':'running','allocated_gpus':os.environ.get('SLURM_GPUS_ON_NODE'),
           'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    try:
        for stage in ('lens','extract'):
            marker=out/('lens-metadata.json' if stage=='lens' else 'extraction.json')
            if marker.exists():
                record=json.loads(marker.read_text())
                hashes={'lens.pt':record['sha256']} if stage=='lens' else record['files']
                if any(sha256(out/path)!=digest for path,digest in hashes.items()):raise ValueError('Cached artifacts changed')
                continue
            state.update(stage=stage);frozen.atomic_json(out/'status.json',state)
            with (out/f'{stage}.log').open('a') as log:
                subprocess.run([sys.executable,__file__,'--out',str(out),'--stage',stage],stdout=log,stderr=subprocess.STDOUT,check=True)
        for name in REPRESENTATIONS:
            state.update(stage='fit',representation=name);frozen.atomic_json(out/'status.json',state)
            with (out/f'fit-{name}.log').open('a') as log:
                subprocess.run([sys.executable,__file__,'--out',str(out),'--stage','fit','--representation',name],stdout=log,stderr=subprocess.STDOUT,check=True)
        summarize(out,m);state.update(state='complete')
    except BaseException as exc:
        state.update(state='failed',error=str(exc));raise
    finally:
        state['invocation_seconds']=time.perf_counter()-started
        frozen.atomic_json(out/'status.json',state)
        with (out/'invocations.jsonl').open('a') as handle:
            handle.write(json.dumps(state)+'\n')


if __name__=='__main__':main()
