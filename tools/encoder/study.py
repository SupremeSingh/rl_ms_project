"""Numina offline encoder study: generation -> MNTP -> SimCSE -> features -> Ridge.

This implements a bounded Numina-specific LLM2Vec-style recipe, not the original
paper's corpus/hyperparameter replication. No VERL or online actor updates.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from importlib.metadata import version

import numpy as np
import torch
from slice_critic.config import completion_prompt, select_layers
from slice_critic.storage import model_identity, sha256, write_json, save_tensor, load_tensor
from slice_critic.features import isotropic_normalization
from slice_critic.bottleneck import Bottleneck, loss, encode
from slice_critic.linear import fit_system
from . import model as enc

ROOT=Path(__file__).resolve().parents[2]
ENCODERS=('decoder','bidir-unadapted','llm2vec')
MODES=('token','chunk')
BUDGETS={'smoke':dict(train=20,val=10,test=10,answers=2,cap=512,adapt_steps=2,epochs=2),
         'full':dict(train=500,val=100,test=200,answers=4,cap=2048,adapt_steps=200,epochs=30)}


def read(p): return json.loads(Path(p).read_text())
def seed(*parts): return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8],16)%2**31
def cache_key(*parts): return hashlib.sha256(json.dumps(parts).encode()).hexdigest()

def runtime():
    return {str(p.relative_to(ROOT)):sha256(p) for pattern in
            ('tools/encoder/*.py','scripts/*encoder*.sh','scripts/*encoder*.py',
             'packages/slice-critic/src/**/*.py','src/math_rl/ppo_reward.py','src/math_rl/verifier_batch.py',
             'configs/container.sha256','configs/ppo-verifier-environment.txt')
            for p in ROOT.glob(pattern)}


def dependencies():
    return {name:version(name) for name in ('torch','numpy','transformers','datasets')}


def prepare(source,out,mode,all_prefixes=False):
    from math_rl.reproduction import validate_questions
    if out.exists(): raise ValueError('Use a new output directory')
    source=source.resolve(); old=read(source/'manifest.json')
    for name,digest in old['data_hashes'].items():
        if sha256(source/'data'/name)!=digest: raise ValueError('Source data changed')
    questions=read(source/'data/questions.json')['questions']; validate_questions(questions)
    identity=model_identity(ROOT/'models/qwen-math')
    if identity!=old['model_hashes']: raise ValueError('Original frozen actor changed')
    budget=BUDGETS[mode]; chosen=[]
    for split in ('train','val','test'):
        rows=sorted((q for q in questions if q['split']==split),key=lambda q:seed('encoder-v1',q['id']))
        if len(rows)<budget[split]: raise ValueError('Insufficient question families')
        chosen+=rows[:budget[split]]
    out.mkdir(parents=True)
    write_json(out/'questions.json',chosen)
    write_json(out/'manifest.json',dict(protocol='numina-encoder-v1',mode=mode,budget=budget,
        all_prefixes=all_prefixes,prefixes_per_answer=8,adapt_length=512,rank=8,
        adaptation_lr=1e-4,contrastive_questions=4,epsilon=.01,seeds=[42,43,44],
        layers=select_layers(28),raw_layer=18,model_files=identity,runtime=runtime(),dependencies=dependencies(),
        questions_sha256=sha256(out/'questions.json'),source_manifest_sha256=sha256(source/'manifest.json'),
        scope='Offline inspected Numina; fixed frozen actor; sampled token/chunk prefixes unless all-prefixes enabled. '
              'LLM2Vec-style MNTP plus dropout contrastive adaptation on training questions only. No PPO.'))


def validate(out):
    m=read(out/'manifest.json')
    if (m.get('protocol')!='numina-encoder-v1' or m.get('mode') not in BUDGETS
            or m['budget']!=BUDGETS[m['mode']] or m['layers']!=select_layers(28)
            or m['raw_layer']!=18 or m['epsilon']!=.01 or m['rank']!=8
            or m['seeds']!=[42,43,44] or m['prefixes_per_answer']!=8):
        raise ValueError('Unsupported/changed encoder protocol')
    if m['runtime']!=runtime() or m['questions_sha256']!=sha256(out/'questions.json'):
        raise ValueError('Study inputs/code changed; prepare a new study')
    if m['dependencies']!=dependencies():raise ValueError('Study dependency versions changed')
    if m['model_files']!=model_identity(ROOT/'models/qwen-math'): raise ValueError('Base weights changed')
    return m


def load_model(dtype=torch.bfloat16):
    from transformers import AutoModelForCausalLM
    torch.manual_seed(42)
    model=AutoModelForCausalLM.from_pretrained(ROOT/'models/qwen-math',local_files_only=True,
        trust_remote_code=False,torch_dtype=dtype,attn_implementation='eager').to('cuda')
    if (len(model.model.layers),model.config.hidden_size)!=(28,1536): raise ValueError('Expected original Qwen Math 1.5B')
    return model


def generate(out,m):
    from slice_critic.backend import TransformersBackend
    from slice_critic.chunks import token_chunks
    from math_rl.ppo_reward import compute_score
    backend=TransformersBackend(ROOT/'models/qwen-math')
    folder=out/'answers';folder.mkdir(exist_ok=True)
    for q in read(out/'questions.json'):
        path=folder/f'{cache_key(q["id"])}.json'
        if path.exists():
            record=read(path)
            if record['question_id']!=q['id'] or record['manifest_sha256']!=sha256(out/'manifest.json'):raise ValueError('Stale answer cache')
            continue
        prompt=backend.prompt_ids(q['question'])
        if not 0<len(prompt)<=512:raise ValueError('Prompt exceeds fixed budget')
        rows=backend.generate(prompt,m['budget']['answers'],m['budget']['cap'],seed(q['id'],'answers'))
        if len(rows)!=m['budget']['answers']:raise ValueError('Incomplete answer generation')
        labels=compute_score(['numina_numeric']*len(rows),[r['response'] for r in rows],[q['ground_truth']]*len(rows))
        for i,(row,label) in enumerate(zip(rows,labels,strict=True)):
            row['label']=label
            if label['score'] not in (0,1):raise ValueError('Invalid binary outcome')
            units={}
            for unit,available in [('token',list(range(len(row['token_ids'])))),
                                   ('chunk',list(token_chunks(backend.tokenizer,row['token_ids'],'math',32).starts))]:
                if not m['all_prefixes'] and len(available)>m['prefixes_per_answer']:
                    rng=np.random.default_rng(seed(q['id'],i,unit))
                    available=sorted(rng.choice(available,m['prefixes_per_answer'],replace=False).tolist())
                units[unit]=available
            row['positions']=units
        write_json(path,dict(question_id=q['id'],manifest_sha256=sha256(out/'manifest.json'),prompt_ids=prompt,responses=rows))
        print('Generated',q['split'],q['id'],flush=True)
    files={p.name:sha256(p) for p in folder.glob('*.json')}
    write_json(out/'generation.json',dict(files=files,manifest_sha256=sha256(out/'manifest.json')))


def records(out):
    gate=read(out/'generation.json')
    if gate['manifest_sha256']!=sha256(out/'manifest.json'):raise ValueError('Stale generation')
    for name,digest in gate['files'].items():
        if sha256(out/'answers'/name)!=digest:raise ValueError('Generated answers changed')
    for q in read(out/'questions.json'):
        r=read(out/'answers'/f'{cache_key(q["id"])}.json')
        for i,row in enumerate(r['responses']):yield q,i,r['prompt_ids'],row


def adaptation_corpus(out,m):
    # One prefix per training question. Contrastive batches never treat two
    # answers to the same question as negatives. Outcome labels are not read.
    corpus=[];seen=set()
    for q,i,prompt,row in records(out):
        if q['split']!='train' or q['id'] in seen:continue
        seen.add(q['id'])
        body=row['token_ids']; position=max(1,len(body)//2)
        corpus.append((q['id'],(prompt+body[:position])[:m['adapt_length']]))
    return corpus


def adapt(out,m,stage):
    target=out/f'{stage}.pt'
    if target.exists():raise ValueError('Adapter already exists; no silent retraining')
    model=load_model();probe=enc.attention_check(model,'cuda')
    if stage=='mntp':enc.install_adapters(model,m['rank'])
    else:
        record=load_tensor(out/'mntp.pt')
        if record['manifest_sha256']!=sha256(out/'manifest.json'):raise ValueError('Stale MNTP adapter')
        enc.restore_adapter(model,out/'mntp.pt')
    model.train();model.config.use_cache=False
    # Only attention dropout supplies SimCSE positive-view augmentation.
    for layer in model.model.layers:layer.self_attn.attention_dropout=.1 if stage=='contrastive' else 0.
    model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=m['adaptation_lr'],weight_decay=0.)
    corpus=adaptation_corpus(out,m)
    if len(corpus)<m['contrastive_questions']:raise ValueError('Too few training questions for contrastive negatives')
    rng=np.random.default_rng(42);generator=torch.Generator().manual_seed(42);history=[]
    started=time.perf_counter()
    for step in range(m['budget']['adapt_steps']):
        optimizer.zero_grad()
        if stage=='mntp':
            _,tokens=corpus[int(rng.integers(len(corpus)))];ids=torch.tensor([tokens],device='cuda')
            objective=enc.masked_next_loss(model,ids,model.config.eos_token_id,generator)
        else:
            selected=rng.choice(len(corpus),m['contrastive_questions'],replace=False)
            ids=[torch.tensor([corpus[i][1]],device='cuda') for i in selected]
            left=torch.cat([enc.embedding(model,x) for x in ids]);right=torch.cat([enc.embedding(model,x) for x in ids])
            objective=enc.contrastive_loss(left,right)
        if not torch.isfinite(objective):raise ValueError('Nonfinite adaptation objective')
        objective.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step();history.append(float(objective.detach()))
        print(stage,step+1,history[-1],flush=True)
    save_tensor(target,dict(weights=enc.adapter_state(model),rank=m['rank'],history=history,
        stage=stage,parent_sha256=sha256(out/'mntp.pt') if stage=='contrastive' else None,
        manifest_sha256=sha256(out/'manifest.json'),seconds=time.perf_counter()-started,attention_check=probe))


def extract(out,m,kind):
    model=load_model(torch.float32).eval().requires_grad_(False)
    if kind=='llm2vec':
        record=enc.restore_adapter(model,out/'contrastive.pt');model.eval().requires_grad_(False)
        if record['manifest_sha256']!=sha256(out/'manifest.json') or record['parent_sha256']!=sha256(out/'mntp.pt'):
            raise ValueError('Adapter provenance mismatch')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    probe=enc.attention_check(model,'cuda')
    folder=out/'features'/kind;folder.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();states_total=0
    torch.cuda.reset_peak_memory_stats()
    for q,i,prompt,row in records(out):
        target=folder/f'{cache_key(q["id"],i)}.pt'
        positions=sorted(set(row['positions']['token']+row['positions']['chunk']))
        identity=dict(manifest_sha256=sha256(out/'manifest.json'),generation_sha256=sha256(out/'generation.json'),
                      adapter_sha256=sha256(out/'contrastive.pt') if kind=='llm2vec' else None)
        if target.exists():
            cached=load_tensor(target)
            if cached['identity']!=identity or cached['positions']!=positions:raise ValueError('Stale feature cache')
            states_total+=len(positions);continue
        if kind=='decoder':
            features=enc.causal_features(model,prompt,row['token_ids'],positions,m['layers'],m['raw_layer'],'cuda')
        else:
            tensors=[]
            for position in positions:
                # Separate prefixes: no future actions can enter bidirectional attention.
                tensors.append(enc.prefix_features(model,prompt+row['token_ids'][:position],m['layers'],m['raw_layer'],True,'cuda'))
            features={k:torch.stack([t[k] for t in tensors]) for k in ('concat','raw','rms')}
        save_tensor(target,dict(question_id=q['id'],answer=i,split=q['split'],y=float(row['label']['score']),
            positions=positions,units=row['positions'],identity=identity,**features))
        states_total+=len(positions)
    write_json(folder/'complete.json',dict(files={p.name:sha256(p) for p in folder.glob('*.pt')},
        attention_check=probe,adapter_sha256=sha256(out/'contrastive.pt') if kind=='llm2vec' else None,
        manifest_sha256=sha256(out/'manifest.json'),seconds=time.perf_counter()-started,states=states_total,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        timing_scope='Current extraction invocation; cache hits may reuse earlier work. Shared token/chunk union, not isolated per-mode timers.'))


def collection(out,kind,unit,split):
    folder=out/'features'/kind;gate=read(folder/'complete.json')
    if gate['manifest_sha256']!=sha256(out/'manifest.json'):raise ValueError('Stale features')
    if kind=='llm2vec' and gate['adapter_sha256']!=sha256(out/'contrastive.pt'):raise ValueError('Encoder changed')
    xs={k:[] for k in ('concat','raw','rms')};ys=[];questions=[]
    for name,digest in gate['files'].items():
        if sha256(folder/name)!=digest:raise ValueError('Features changed')
        r=load_tensor(folder/name)
        if r['split']!=split:continue
        # Held-out evaluation states are shared across both training-unit variants.
        selected=list(range(len(r['positions']))) if split!='train' else [i for i,p in enumerate(r['positions']) if p in r['units'][unit]]
        for k in xs:xs[k].append(r[k][selected])
        ys.extend([r['y']]*len(selected));questions.extend([r['question_id']]*len(selected))
    if not ys:raise ValueError('Empty feature collection')
    return dict(**{k:torch.cat(v) for k,v in xs.items()},y=torch.tensor(ys),questions=questions)


def equal_question(values,questions):
    groups={}
    for value,q in zip(values,questions,strict=True):groups.setdefault(q,[]).append(float(value))
    return {q:float(np.mean(v)) for q,v in groups.items()}


def fit(out,m,kind):
    folder=out/'fits'/kind;folder.mkdir(parents=True,exist_ok=True)
    for unit in MODES:
        train,val=[collection(out,kind,unit,s) for s in ('train','val')]
        if set(train['y'].tolist())!={0.,1.} or set(val['y'].tolist())!={0.,1.}:raise ValueError('Train/val need both outcome classes')
        for representation in ('raw','rms','slice'):
            for random_seed in (m['seeds'] if representation=='slice' else [0]):
                path=folder/f'{unit}-{representation}-{random_seed}.pt'
                if path.exists():raise ValueError('Fit already exists; use a new study')
                params={};history=[]
                if representation=='slice':
                    torch.manual_seed(random_seed);mean,scale=isotropic_normalization(train['concat'])
                    net=Bottleneck(7680,768).cuda();optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.001)
                    best=float('inf');best_state=None
                    for epoch in range(m['budget']['epochs']):
                        net.train()
                        for ids in torch.randperm(len(train['y'])).split(256):
                            x=((train['concat'][ids]-mean)/scale).cuda(); y=train['y'][ids].cuda()
                            objective=loss(net,x,y,.01)
                            if not torch.isfinite(objective):raise ValueError('Nonfinite SLICE objective')
                            optimizer.zero_grad();objective.backward();optimizer.step()
                        pred=encode(net,val['concat'],mean,scale,'cuda');net.eval()
                        with torch.no_grad():p=net.readout(pred.cuda()).flatten().sigmoid().cpu()
                        score=np.mean(list(equal_question((p-val['y']).square(),val['questions']).values()))
                        history.append(float(score))
                        if score<best:
                            best=score;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                    if best_state is None:raise ValueError('No finite validation-selected compressor')
                    net.load_state_dict(best_state)
                    x=encode(net,train['concat'],mean,scale,'cuda')
                    params=dict(compressor=best_state,mean=mean,scale=scale)
                else:x=train[representation]
                design=torch.cat([x.double(),torch.ones(len(x),1,dtype=torch.float64)],1)
                w,residual=fit_system(design.T@design,design.T@train['y'].double(),m['epsilon'])
                save_tensor(path,dict(weights=w,params=params,validation_history=history,residual=residual,
                    manifest_sha256=sha256(out/'manifest.json'),features_sha256=sha256(out/'features'/kind/'complete.json')))


def evaluate(out,m):
    results={};predictions={}
    for kind in ENCODERS:
        for unit in MODES:
            test=collection(out,kind,unit,'test')
            for representation in ('raw','rms','slice'):
                rows=[];seed_metrics=[]
                for random_seed in (m['seeds'] if representation=='slice' else [0]):
                    r=load_tensor(out/'fits'/kind/f'{unit}-{representation}-{random_seed}.pt')
                    if r['manifest_sha256']!=sha256(out/'manifest.json') or r['features_sha256']!=sha256(out/'features'/kind/'complete.json'):raise ValueError('Stale fitted model')
                    if representation=='slice':
                        net=Bottleneck(7680,768);net.load_state_dict(r['params']['compressor'])
                        x=encode(net,test['concat'],r['params']['mean'],r['params']['scale'],'cpu')
                    else:x=test[representation]
                    design=torch.cat([x.double(),torch.ones(len(x),1,dtype=torch.float64)],1)
                    p=design@r['weights'];y=test['y']
                    brier=equal_question((p-y).square(),test['questions']);rows.append(brier)
                    seed_metrics.append(dict(seed=random_seed,question_brier=np.mean(list(brier.values())),
                        accuracy=np.mean(list(equal_question(((p>=.5)==y.bool()).float(),test['questions']).values())),
                        outside_unit_interval=float(((p<0)|(p>1)).double().mean())))
                name=f'{kind}/{unit}/{representation}'
                results[name]=dict(raw_brier=float(np.mean([v['question_brier'] for v in seed_metrics])),
                    accuracy=float(np.mean([v['accuracy'] for v in seed_metrics])),seeds=seed_metrics,questions=len(rows[0]),states=len(y))
                predictions[name]={q:float(np.mean([row[q] for row in rows])) for q in rows[0]}
    comparisons={}
    for unit in MODES:
        base=predictions[f'decoder/{unit}/slice']
        for kind in ENCODERS[1:]:
            other=predictions[f'{kind}/{unit}/slice']
            if set(other)!=set(base):raise ValueError('Unequal test questions')
            difference=np.array([other[q]-base[q] for q in sorted(base)])
            rng=np.random.default_rng(765)
            bootstrap=[rng.choice(difference,len(difference),replace=True).mean() for _ in range(2000)]
            comparisons[f'{kind}-minus-decoder/{unit}/slice']=dict(mean=float(difference.mean()),
                descriptive_question_bootstrap_95=np.quantile(bootstrap,[.025,.975]).tolist(),
                scope='Exploratory, unadjusted; conditional on one actor and adaptation seed; compressor seeds averaged')
    record=dict(results=results,comparisons=comparisons,manifest_sha256=sha256(out/'manifest.json'),scope=m['scope'],
        extraction={kind:read(out/'features'/kind/'complete.json') for kind in ENCODERS},
        evaluation_states='Identical token/chunk union on validation/test for every arm; only training-state selection differs.')
    write_json(out/'summary.json',record)
    lines=['Offline Numina eventual-correctness prediction; equal-question raw Brier; unclipped Ridge.']
    lines += [f'{k:40} Brier={v["raw_brier"]:.6f} accuracy={100*v["accuracy"]:.2f}%' for k,v in results.items()]
    lines += [json.dumps(comparisons),m['scope'],'No PPO, MC ground-truth-value or total-cost superiority claim.']
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    write_json(out/'verification.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),summary_sha256=sha256(out/'summary.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['prepare','generate','mntp','contrastive','extract','fit','evaluate'])
    p.add_argument('--out',required=True,type=Path);p.add_argument('--source',type=Path)
    p.add_argument('--mode',choices=list(BUDGETS),default='smoke');p.add_argument('--task',type=int,default=0)
    p.add_argument('--all-prefixes',action='store_true',help='Every token/chunk start; potentially very expensive for bidirectional encoders')
    a=p.parse_args();out=a.out.resolve();started=time.perf_counter()
    if a.stage=='prepare':
        if a.source is None:p.error('--source required')
        prepare(a.source,out,a.mode,a.all_prefixes);return
    m=validate(out)
    try:
        if a.stage=='generate':generate(out,m)
        elif a.stage in ('mntp','contrastive'):adapt(out,m,a.stage)
        elif a.stage in ('extract','fit'):
            if not 0<=a.task<len(ENCODERS):raise ValueError('Invalid encoder task')
            (extract if a.stage=='extract' else fit)(out,m,ENCODERS[a.task])
        else:evaluate(out,m)
        write_json(out/'status'/f'{a.stage}-{a.task}.json',dict(state='complete',seconds=time.perf_counter()-started))
    except Exception as exc:
        write_json(out/'status'/f'{a.stage}-{a.task}.json',dict(state='failed',error=str(exc)));raise


if __name__=='__main__':main()
