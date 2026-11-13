"""Expanded offline comparison. Separate protocol; no PPO or actor optimizer."""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'packages/slice-critic/src'),str(Path(__file__).parent)]
import torch
import study
from slice_critic.storage import sha256,write_json,model_identity
from slice_critic.features import rms_coordinates,isotropic_normalization
from slice_critic.bottleneck import Bottleneck,loss,encode
from slice_critic.cumulative import raw_design
from slice_critic.linear import fit_system
from math_rl.value_audit import stable_seed
from math_rl.encoder_features import attention_probe,require_bidirectional

PROTOCOL='encoder-pooled-v2'
LAYERS=[27,26,24,20,12]
ARMS=['A','B','C','D']

def read(p):return json.loads(Path(p).read_text())
def identity():
    files=[Path(__file__),Path(__file__).with_name('distributed_check.py'),Path(__file__).with_name('submit_compare.py'),Path(__file__).with_name('run_compare.sh')]
    files+=list((ROOT/'src/math_rl').glob('*.py'))+list((ROOT/'packages/slice-critic/src/slice_critic').rglob('*.py'))
    files += [Path(__file__).with_name('study.py'),Path(__file__).with_name('official_train.py'),ROOT/'requirements-encoder.txt',ROOT/'scripts/container_exec.sh',ROOT/'scripts/cluster_env.sh']
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(files))}

def seal(out,name,payload):
    path=out/name
    if path.exists():raise FileExistsError(path)
    write_json(path,payload)
    write_json(path.with_suffix(path.suffix+'.sha256'),dict(sha256=sha256(path)))

def checked(out,name):
    p=out/name
    if sha256(p)!=read(p.with_suffix(p.suffix+'.sha256'))['sha256']:raise ValueError(f'Changed artifact: {p}')
    return read(p)

def provenance(path):
    for name in ('run.json','identity.json'):
        p=path.parent/name
        if p.exists() and read(p).get('questions_sha256')==sha256(path):return read(p)
    raise ValueError(f'No matching saved run provenance for {path}')

def inventory():
    candidates=[]
    for root in ('outputs','data'):
        for p in (ROOT/root).glob('**/questions.json'):
            try:
                q=read(p)
                if isinstance(q,list) and len(q)>=13000:
                    provenance(p);candidates.append(str(p))
            except (ValueError,OSError):continue
    print(json.dumps(dict(verified_offline_question_files=candidates),indent=2))

def prepare(out,args):
    source=args.source.resolve();old=read(source/'manifest.json')
    gate=read(source/'verification.json')
    if not gate.get('passed') or old.get('protocol')!='fixed-policy-critics-v1':raise ValueError('Completed fixed-policy source required')
    for name,h in gate['files'].items():
        if sha256(source/name)!=h:raise ValueError('Source changed')
    questions=args.questions
    if questions is None:
        candidates=[]
        for root in ('outputs','data'):
            for p in (ROOT/root).glob('**/questions.json'):
                try:
                    provenance(p)
                    if isinstance(read(p),list) and len(read(p))>=13000:candidates.append(p)
                except (ValueError,OSError):continue
        if len(candidates)!=1:raise ValueError('Specify --questions using inventory; need exactly one verified 13,000-question offline corpus')
        questions=candidates[0]
    questions=questions.resolve();qs=read(questions);meta=provenance(questions)
    if meta['questions_sha256']!=sha256(questions):raise ValueError('Question provenance mismatch')
    if model_identity(Path(old['actor_path']))!=old['actor_identity']:raise ValueError('Frozen actor changed')
    if model_identity(ROOT/'models/qwen-math')!=old['base_identity']:raise ValueError('Base model changed')
    forbidden={}
    for p in (args.numina.resolve()/'data/questions.json',args.numina.resolve()/'online/questions.json'):
        if p.exists():
            obj=read(p);
            from slice_critic.text import family_key
            forbidden.update({family_key(q['question']):q['split'] for q in (obj['questions'] if isinstance(obj,dict) else obj)})
    limits=dict(train=8,val=4,test=4) if args.mode=='smoke' else dict(train=7800,val=2600,test=2600)
    selected=[];families=set()
    from slice_critic.text import family_key
    for split,n in limits.items():
        group=sorted([q for q in qs if q['split']==split],key=lambda q:stable_seed('pooled-v2',q['id']))
        if len(group)<n:raise ValueError(f'Need {n} existing {split} questions; no repartitioning allowed')
        for q in group[:n]:
            family=family_key(q['question'])
            if family in families or (split=='train' and forbidden.get(family) in ('val','test')):raise ValueError('Question-family overlap or historical held-out question in training')
            families.add(family);selected.append(dict(id=q['id'],question=q['question'],ground_truth=q['reference'],split=split))
    official=args.official.resolve()
    if subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()!=study.OFFICIAL or subprocess.check_output(['git','-C',str(official),'status','--porcelain'],text=True).strip():raise ValueError('Clean pinned official checkout required')
    out.mkdir(parents=True,exist_ok=False)
    m=dict(protocol=PROTOCOL,mode=args.mode,runtime=identity(),questions_path=str(questions),questions_hash=sha256(questions),source=str(source),source_hash=sha256(source/'manifest.json'),actor_path=old['actor_path'],actor_identity=old['actor_identity'],model=str(ROOT/'models/qwen-math'),base_identity=old['base_identity'],official=str(official),seed=84621,reference_seed=84622,response_cap=2048,max_model_len=2560,sampling=dict(temperature=1.,top_p=1.,top_k=-1),shards=8,answers=8,adapt_seeds=[42] if args.mode=='smoke' else [42,43,44],compressor_seeds=[42] if args.mode=='smoke' else [42,43,44,45,46],epochs=2 if args.mode=='smoke' else 120,checkpoint_every=1 if args.mode=='smoke' else 20,adapt_steps=2 if args.mode=='smoke' else 1000,mc_questions=4 if args.mode=='smoke' else 1000,mc_samples=4 if args.mode=='smoke' else 128,mc_offsets=[0,128,256,512],stride=32,scope='Offline one-frozen-actor value comparison. Reused/inspected Numina families, not PPO or fresh benchmark.')
    seal(out,'manifest.json',m);seal(out,'questions.json',selected)

def validate(out):
    m=checked(out,'manifest.json')
    if m['protocol']!=PROTOCOL or m['runtime']!=identity():raise ValueError('Protocol/runtime changed')
    if sha256(m['questions_path'])!=m['questions_hash']:raise ValueError('Question source changed')
    if model_identity(m['actor_path'])!=m['actor_identity'] or model_identity(m['model'])!=m['base_identity']:raise ValueError('Actor/base weights changed')
    if sha256(Path(m['source'])/'manifest.json')!=m['source_hash']:raise ValueError('Source manifest changed')
    if subprocess.check_output(['git','-C',m['official'],'rev-parse','HEAD'],text=True).strip()!=study.OFFICIAL or subprocess.check_output(['git','-C',m['official'],'status','--porcelain'],text=True).strip():raise ValueError('Official checkout changed')
    return m

def items(out,m):
    result=[dict(q,key=f'{stable_seed(q["id"]):010d}-{i}',answer_index=i) for q in checked(out,'questions.json') for i in range(m['answers'])]
    if len({r['key'] for r in result})!=len(result):raise ValueError('Rollout key collision')
    return result

def answer(out,r):return checked(out,f'answers/{r["key"]}.json')

def generate(out,m,task):
    from math_rl.value_audit_gpu import Generator,grade
    from slice_critic.config import completion_prompt
    g=Generator(Path(m['actor_path']),m)
    for q in checked(out,'questions.json')[task::m['shards']]:
        prompt=g.tokenizer.encode(completion_prompt(q['question']),add_special_tokens=False)
        if not 0<len(prompt)<=512:raise ValueError('Prompt exceeds sealed limit')
        samples=grade(g,q['ground_truth'],g.generate(prompt,m['answers'],m['response_cap'],stable_seed(m['seed'],q['id'])),[])
        for i,s in enumerate(samples):seal(out,f'answers/{stable_seed(q["id"]):010d}-{i}.json',dict(s,prompt_ids=prompt,question_id=q['id'],split=q['split']))
        print('Generated',q['id'],flush=True)
    seal(out,f'generation-{task}.json',dict(passed=True))

def positions(data,m):return list(range(0,len(data['token_ids']),m['stride']))

def profile(out,m):
    import numpy as np
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(m['model'],local_files_only=True)
    lengths=[];texts=[];seen=set();counts=defaultdict(int)
    # Four absolute, reward-independent prefix offsets per training answer.
    for row in items(out,m):
        if row['split']=='test':continue
        d=answer(out,row)
        lengths.extend(len(d['prompt_ids'])+p for p in positions(d,m))
        if row['split']!='train':continue
        for p in [0,128,256,512,1024,1536,2016]:
            if p>=len(d['token_ids']):continue
            ids=d['prompt_ids']+d['token_ids'][:p]
            text=' '.join(tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False).splitlines()).strip()
            if not text or '!@#$%^&*()' in text or text in seen:continue
            # Keep full prompt+prefix, never silently truncate/repack SimCSE examples.
            if len(tok.encode(text,add_special_tokens=True))>m['max_model_len']:raise ValueError('Adaptation example exceeds context')
            if counts[row['id']]>=16:continue
            counts[row['id']]+=1;seen.add(text);texts.append(text)
    if len(texts)<8:raise ValueError('Too few training-only prefix texts')
    (out/'adaptation.txt').write_text('\n'.join(texts)+'\n')
    with (out/'adaptation.txt').open() as f:loaded=[s.strip() for s in f]
    if loaded!=texts:raise ValueError('Line-file round trip changed adaptation text')
    seal(out,'profile.json',dict(percentiles=np.quantile(lengths,[.5,.9,.95,.99,1]).tolist(),coverage={str(n):float(np.mean(np.array(lengths)<=n)) for n in (512,1024,2048,2560,4096)},context=m['max_model_len'],reason='Full sealed generation context; all fitting prefixes covered without truncation',training_texts=len(texts),adaptation_sha256=sha256(out/'adaptation.txt')))

def backbones(m):return ['causal']+[f'adapted-{s}' for s in m['adapt_seeds']]
def combinations(m):
    return [(b,a,s) for b in backbones(m) for a in ARMS for s in (m['compressor_seeds'] if a in ('C','D') else [0])]

def configs(out,m,seed):
    work=out/f'adapt-{seed}'
    a,b=study.trainer_configs(dict(model=m['model'],adaptation_steps=m['adapt_steps']),work)
    for c in (a,b):c.update(seed=seed,max_seq_length=checked(out,'profile.json')['context'],gradient_accumulation_steps=1)
    a['save_steps']=b['save_steps']=min(250,m['adapt_steps'])
    a.update(train_file=str(out/'adaptation.txt'),validation_file=str(out/'adaptation.txt'),per_device_train_batch_size=2)
    b.update(dataset_file_path=str(out/'adaptation.txt'),per_device_train_batch_size=8)
    return a,b

def adapt(out,m,task):
    seed=m['adapt_seeds'][task];work=out/f'adapt-{seed}';work.mkdir(exist_ok=False)
    for stage,c in zip(('mntp','simcse'),configs(out,m,seed)):
        checked(out,'distributed-check.json')
        write_json(work/f'{stage}.json',c)
        cmd=[sys.executable]
        if stage=='simcse':cmd+=['-m','torch.distributed.run','--standalone','--nproc_per_node=4']
        cmd += [str(Path(__file__).with_name('official_train.py')),str(Path(m['official'])/'experiments'/f'run_{stage}.py'),str(work/f'{stage}.json')]
        env=dict(os.environ,TOKENIZERS_PARALLELISM='false')
        if stage=='mntp':env['CUDA_VISIBLE_DEVICES']=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0]
        with (work/f'{stage}.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
        from llm2vec import LLM2Vec
        adapter=work/stage if stage=='mntp' else work/stage/f'checkpoint-{m["adapt_steps"]}'
        wrapper=LLM2Vec.from_pretrained(m['model'] if stage=='mntp' else str(work/'mntp-model'),peft_model_name_or_path=str(adapter),merge_peft=True,enable_bidirectional=True,skip_instruction=False,pooling_mode='mean',max_length=checked(out,'profile.json')['context'],torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2')
        wrapper.save(str(work/('mntp-model' if stage=='mntp' else 'adapted-model')));del wrapper
    seal(out,f'adapt-{seed}/complete.json',dict(model_identity=model_identity(work/'adapted-model')))

@torch.inference_mode()
def pooled(model,seq):
    if model.training or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen eval backbone required')
    if not seq or len(seq)>model.config.max_position_embeddings:raise ValueError('Invalid or overlong prefix')
    device=next(model.parameters()).device;captured={};hooks=[]
    for l in sorted(set(LAYERS+[18])):
        def hook(module,args,output,l=l):
            h=output[0] if isinstance(output,tuple) else output
            if l==27:h=model.norm(h)
            captured[l]=rms_coordinates(h[0].float().mean(0),model.config.rms_norm_eps).cpu()
        hooks.append(model.layers[l].register_forward_hook(hook))
    try:
        ids=torch.tensor([seq],device=device)
        model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
    finally:
        for h in hooks:h.remove()
    return dict(A=captured[27],B=captured[18],multi=torch.cat([captured[l] for l in LAYERS]))

def load(out,m,b):
    from llm2vec import LLM2Vec
    path=m['model'] if b=='causal' else str(out/f'adapt-{b.split("-")[1]}'/'adapted-model')
    if b!='causal':
        if model_identity(path)!=checked(out,f'adapt-{b.split("-")[1]}/complete.json')['model_identity']:raise ValueError('Adapted weights changed')
    model=LLM2Vec.from_pretrained(path,enable_bidirectional=b!='causal',torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').model.cuda().eval().requires_grad_(False)
    if len(model.layers)!=28 or model.config.hidden_size!=1536:raise ValueError('Expected 28-block, 1536-wide Qwen backbone')
    sensitivity=attention_probe(model,[15,21,18,32])
    if b!='causal':require_bidirectional(sensitivity,0.)
    elif sensitivity>1e-5:raise ValueError('Causal control is not causal')
    return model

def extract(out,m,task):
    b=backbones(m)[task//m['shards']];shard=task%m['shards'];model=load(out,m,b)
    for row in items(out,m)[shard::m['shards']]:
        d=answer(out,row);ps=positions(d,m);features=[pooled(model,d['prompt_ids']+d['token_ids'][:p]) for p in ps]
        path=out/'features'/b/f'{row["key"]}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(path)
        torch.save(dict(x={k:torch.stack([r[k] for r in features]) for k in features[0]},positions=ps,y=torch.full((len(ps),),float(d['score'])),question=row['id'],split=row['split']),path)
        write_json(path.with_suffix('.json'),dict(sha256=sha256(path),answer_sha256=sha256(out/f'answers/{row["key"]}.json')))
    seal(out,f'extract-{b}-{shard}.json',dict(passed=True))

def feature(out,b,row):
    p=out/'features'/b/f'{row["key"]}.pt';meta=read(p.with_suffix('.json'))
    if sha256(p)!=meta['sha256'] or sha256(out/f'answers/{row["key"]}.json')!=meta['answer_sha256']:raise ValueError('Feature provenance changed')
    return torch.load(p,map_location='cpu',weights_only=True)

def xvalue(r,a):return r['x'][a if a in ('A','B') else 'multi']
def group_metric(values):return sum(sum(v)/len(v) for v in values.values())/len(values)

def fit_head(out,m,b,a,compressor,mean,scale):
    width=1536 if a!='D' else 768
    aa=torch.zeros(width+1,width+1,dtype=torch.float64);bb=torch.zeros(width+1,dtype=torch.float64)
    for row in items(out,m):
        if row['split']!='train':continue
        r=feature(out,b,row);x=xvalue(r,a)
        if compressor is not None:x=encode(compressor,x,mean,scale,'cuda')
        phi=raw_design(x);aa+=phi.T@phi;bb+=phi.T@r['y'].double()
    w,residual=fit_system(aa,bb,.01)
    return dict(weights=w,residual=residual,epsilon=.01,folds=1)

def restored(record):
    if 'compressor' not in record:return None
    model=Bottleneck(len(record['mean']),record['width']);model.load_state_dict(record['compressor']);return model.eval()

def predict(r,a,record,model=None):
    x=xvalue(r,a)
    if 'compressor' in record:
        if model is None:model=restored(record)
        x=encode(model,x,record['mean'],record['scale'],'cpu')
    return raw_design(x)@record['weights']

def score(out,m,b,a,record,split):
    grouped=defaultdict(list);model=restored(record)
    for row in items(out,m):
        if row['split']!=split:continue
        r=feature(out,b,row);p=predict(r,a,record,model);grouped[row['id']].extend((p-r['y']).square().tolist())
    return group_metric(grouped),{q:sum(v)/len(v) for q,v in grouped.items()}

def fit(out,m,task):
    b,a,seed=combinations(m)[task];folder=out/'fits'/f'{b}-{a}-{seed}';folder.mkdir(parents=True,exist_ok=False)
    if a in ('A','B'):
        record=fit_head(out,m,b,a,None,None,None);record['validation_brier']=score(out,m,b,a,record,'val')[0]
    else:
        torch.manual_seed(seed);xs=[];ys=[]
        for row in items(out,m):
            if row['split']!='train':continue
            r=feature(out,b,row);g=torch.Generator().manual_seed(stable_seed('prefixes',row['key']));ids=torch.randperm(len(r['y']),generator=g)[:8]
            xs.append(xvalue(r,a)[ids]);ys.append(r['y'][ids])
        x=torch.cat(xs);y=torch.cat(ys)
        if set(y.tolist())!={0.,1.}:raise ValueError('Need both outcome classes in fitting data')
        mean,scale=isotropic_normalization(x);width=1536 if a=='C' else 768
        model=Bottleneck(x.shape[1],width).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
        best=float('inf');history=[]
        for epoch in range(1,m['epochs']+1):
            model.train()
            for ids in torch.randperm(len(x)).split(2048):
                objective=loss(model,((x[ids]-mean)/scale).cuda(),y[ids].cuda(),.01)
                if not torch.isfinite(objective):raise ValueError('Nonfinite compressor loss')
                opt.zero_grad();objective.backward();opt.step()
            if epoch%m['checkpoint_every'] and epoch!=m['epochs']:continue
            model.eval();head=fit_head(out,m,b,a,model,mean,scale)
            candidate=dict(head,compressor={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},mean=mean,scale=scale,width=width,epoch=epoch)
            val=score(out,m,b,a,candidate,'val')[0];history.append(dict(epoch=epoch,validation_brier=val))
            print(b,a,seed,history[-1],flush=True)
            if val<best:best=val;record=candidate
        record.update(validation_brier=best,history=history)
    torch.save(record,folder/'model.pt');seal(out,f'fits/{folder.name}/complete.json',dict(sha256=sha256(folder/'model.pt')))

def mc(out,m,task):
    from math_rl.value_audit_gpu import Generator,grade
    g=Generator(Path(m['actor_path']),m)
    tests=[q for q in checked(out,'questions.json') if q['split']=='test'][:m['mc_questions']]
    rows=[]
    for q in tests[task::m['shards']]:
        key=f'{stable_seed(q["id"]):010d}-0';d=answer(out,dict(key=key))
        for p in m['mc_offsets']:
            if p>=len(d['token_ids']):continue
            samples=[]
            for offset in range(0,m['mc_samples'],16):
                n=min(16,m['mc_samples']-offset)
                seed=stable_seed(m['reference_seed'],q['id'],p,offset)
                batch=grade(g,q['ground_truth'],g.generate(d['prompt_ids']+d['token_ids'][:p],n,m['response_cap']-p,seed),d['token_ids'][:p])
                seal(out,f'references/{key}-{p}-{offset}.json',dict(seed=seed,samples=batch))
                samples+=batch
            rows.append(dict(question=q['id'],key=key,position=p,samples=len(samples),successes=sum(s['score'] for s in samples)))
            print('MC',q['id'],p,flush=True)
    seal(out,f'mc-{task}.json',rows)

def primary_intervals(m,byquestion,draws=2000):
    """Paired question/fit-seed/adaptation-seed bootstrap; one fixed actor."""
    import numpy as np
    rng=np.random.default_rng(89172);result={}
    for arm in ARMS:
        fit_seeds=m['compressor_seeds'] if arm in ('C','D') else [0]
        for metric in ('brier','mc'):
            first=byquestion[f'causal-{arm}-{fit_seeds[0]}'][metric];qs=sorted(first)
            def values(name):
                q=byquestion[name][metric]
                return np.array([q[k] if metric=='brier' else q[k][0] for k in qs])
            delta=np.array([[values(f'adapted-{a}-{arm}-{s}')-values(f'causal-{arm}-{s}') for s in fit_seeds] for a in m['adapt_seeds']])
            samples=[delta[np.ix_(rng.integers(delta.shape[0],size=delta.shape[0]),rng.integers(delta.shape[1],size=delta.shape[1]),rng.integers(len(qs),size=len(qs)))].mean() for _ in range(draws)]
            result[f'{arm}-{metric}']=dict(mean=float(delta.mean()),interval_95=np.quantile(samples,[.025,.975]).tolist(),family_adjusted_interval=np.quantile(samples,[.00625,.99375]).tolist(),scope='Paired hierarchical bootstrap over adaptation seeds, compressor seeds and questions; Bonferroni family of four readouts per metric. Conditional on one actor, rollout bank and finite MC references.')
    return result


def evaluate(out,m):
    import numpy as np
    refs=[r for task in range(m['shards']) for r in checked(out,f'mc-{task}.json')];result={};byquestion={}
    for b,a,seed in combinations(m):
        name=f'{b}-{a}-{seed}';path=out/'fits'/name/'model.pt'
        if sha256(path)!=checked(out,f'fits/{name}/complete.json')['sha256']:raise ValueError('Fit changed')
        record=torch.load(path,map_location='cpu',weights_only=True)
        brier,q=score(out,m,b,a,record,'test');errs=defaultdict(list);model=restored(record)
        for ref in refs:
            r=feature(out,b,dict(key=ref['key']));i=r['positions'].index(ref['position']);v=float(predict(r,a,record,model)[i]);target=ref['successes']/ref['samples'];errs[ref['question']].append((abs(v-target),(v-target)**2))
        eq={q:np.mean(v,axis=0).tolist() for q,v in errs.items()}
        result[name]=dict(brier=brier,mc_mae=sum(v[0] for v in eq.values())/len(eq),mc_rmse=math.sqrt(sum(v[1] for v in eq.values())/len(eq)),validation_brier=record['validation_brier'])
        byquestion[name]=dict(brier=q,mc=eq)
    # Average fitting seeds within a representation before paired question bootstrap.
    contrasts={};rng=np.random.default_rng(7361)
    pairs=[(f'adapted-{s}-{a}',f'causal-{a}') for s in m['adapt_seeds'] for a in ARMS]
    pairs += [(f'{b}-{a}',f'{b}-{d}') for b in backbones(m) for a,d in [('A','B'),('C','A'),('C','D')]]
    for first,second in pairs:
        def grouped(prefix,metric='brier'):
            names=[n for n in byquestion if n.rsplit('-',1)[0]==prefix]
            keys=set.intersection(*(set(byquestion[n][metric]) for n in names))
            return {q:np.mean([byquestion[n][metric][q] if metric=='brier' else byquestion[n][metric][q][0] for n in names]) for q in keys}
        left,right=grouped(first),grouped(second);keys=sorted(left.keys()&right.keys());delta=np.array([left[q]-right[q] for q in keys]);draws=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(2000)]
        leftmc,rightmc=grouped(first,'mc'),grouped(second,'mc');mcids=sorted(leftmc.keys()&rightmc.keys());md=np.array([leftmc[q]-rightmc[q] for q in mcids]);mc_draws=[rng.choice(md,len(md),replace=True).mean() for _ in range(2000)]
        contrasts[first+'_minus_'+second]=dict(mean=float(delta.mean()),interval=np.quantile(draws,[.025,.975]).tolist(),mc_mae_difference=float(md.mean()),mc_mae_interval=np.quantile(mc_draws,[.025,.975]).tolist(),scope='Exploratory question-clustered, conditional on actor/rollouts; adaptation seeds reported separately')
    seal(out,'summary.json',dict(results=result,contrasts=contrasts,primary=primary_intervals(m,byquestion),by_question=byquestion,mc_states=len(refs),mc_continuations=sum(r['samples'] for r in refs),reference_counts=refs,finite_mc_note='128-sample references are noisy; intervals conditional on the shared sampled reference bank, not exact values. Short trajectories have fewer states.',scope=m['scope']))
    (out/'report.txt').write_text(json.dumps(dict(results=result,contrasts=contrasts,primary=primary_intervals(m,byquestion),mc_states=len(refs),scope=m['scope']),indent=2)+'\n')
    seal(out,'verification.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),summary_sha256=sha256(out/'summary.json')))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=['inventory','prepare','generate','profile','adapt','extract','fit','mc','evaluate']);p.add_argument('--out',type=Path);p.add_argument('--task',type=int,default=0);p.add_argument('--mode',choices=['smoke','main'],default='smoke');p.add_argument('--source',type=Path,default=Path('outputs/fixed-policy-critics-v1'));p.add_argument('--questions',type=Path);p.add_argument('--numina',type=Path,default=Path('outputs/numina-no-folds-v1'));p.add_argument('--official',type=Path,default=Path('third_party/llm2vec'))
    args=p.parse_args();torch.set_num_threads(4)
    if args.stage=='inventory':inventory();return
    if args.out is None:p.error('--out required')
    out=args.out.resolve()
    if args.stage=='prepare':prepare(out,args);return
    m=validate(out)
    if args.stage in ('profile','evaluate'):globals()[args.stage](out,m)
    else:globals()[args.stage](out,m,args.task)

if __name__=='__main__':main()
