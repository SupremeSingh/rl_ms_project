"""Offline transfer: frozen Qwen; fresh answers; source/target/pooled/combined SLICE."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from slice_critic.config import TrainingConfig, select_layers, completion_prompt
from slice_critic.data import load_rows, prepare_questions
from slice_critic.text import family_key
from slice_critic.numbers import parse_number
from slice_critic.references import reference_decision
from slice_critic.storage import write_json, read_json, save_tensor, load_tensor, sha256, model_identity
from slice_critic.backend import resolve_model, TransformersBackend
from slice_critic.training import VerificationClient, _generate, _sampled, _read_record
from slice_critic.bottleneck import Bottleneck
from slice_critic.features import isotropic_normalization

ROOT=Path(__file__).resolve().parents[2]
ARMS=('full_transfer','feature_transfer','target_slice','pooled_slice','combined_768','wide_pooled','rms_two_thirds')
SEEDS=(42,43,44)
EPSILONS=(.01,1.,100.)


def code_hashes():
    paths=[Path(__file__),Path(__file__).with_name('run.sh'),Path(__file__).with_name('submit.py'),*sorted((ROOT/'packages/slice-critic/src/slice_critic').glob('*.py'))]
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def verify(out):
    m=read_json(out/'manifest.json')
    if m['code']!=code_hashes(): raise ValueError('Code changed; use a new study directory')
    if m['questions_sha256']!=sha256(out/'questions.json'): raise ValueError('Questions changed')
    return m


def normalize_rows(rows,domain):
    result=[];excluded=Counter()
    for r in rows:
        if domain=='gsm8k':
            q=r['question'];text=r['answer']
            number=parse_number(text.rsplit('####',1)[-1]) if text.count('####')==1 else None
        else:
            if str(r.get('source','')).strip().lower() in ('gsm8k','math'):
                excluded['held_out_source_tag']+=1;continue
            q=r['problem'];d=reference_decision(q,r['solution'])
            number=d['numeric_reference'] if d['eligible'] else None
        if number is None: excluded['unsupported_reference']+=1;continue
        result.append(dict(problem=q,answer=str(number)))
    return result,dict(excluded)


def exclude(rows,blocked):
    return [r for r in rows if family_key(r['problem']) not in blocked]


def assign(rows,train,val,test):
    if len(rows)!=train+val+test: raise ValueError('Split budget mismatch')
    for i,q in enumerate(rows):q['split']='train' if i<train else 'val' if i<train+val else 'test'
    return rows


def prepare(a):
    if a.out.exists():raise ValueError('Use a new output directory')
    c=TrainingConfig(dataset='AI-MO/NuminaMath-CoT',model_path=a.model,max_new_tokens=a.tokens,
        responses_per_question=a.answers,prefixes_per_answer=8,epochs=a.epochs,
        verifier_python=a.verifier_python)
    c.validate()
    # Preflight the isolated pinned grader on the CPU node, before downloads/GPU jobs.
    grader=VerificationClient(c.verifier_python);grader.check()
    from transformers import AutoTokenizer
    model,rev=resolve_model(c)
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    class Backend:
        max_context=4096
        def prompt_ids(self,q):return tok.encode(completion_prompt(q),add_special_tokens=False)
    backend=Backend();gc=replace(c,dataset='openai/gsm8k',dataset_config='main')
    tr,ti=load_rows(gc);te,ei=load_rows(replace(gc,dataset_split='test',dataset_revision=ti['revision']))
    tr,tx=normalize_rows(tr,'gsm8k');te,ex=normalize_rows(te,'gsm8k')
    # All native target questions blocked, including ones not sampled for this study.
    blocked={family_key(r['problem']) for r in tr+te}
    tr=exclude(tr,{family_key(r['problem']) for r in te})
    gt,ga=prepare_questions(tr,replace(c,max_questions=a.train+a.val),backend)
    ge,gb=prepare_questions(te,replace(c,max_questions=a.test),backend)
    gsm=assign(gt,a.train,a.val,0)+assign(ge,0,0,a.test)
    nr,ni=load_rows(c);nr,nx=normalize_rows(nr,'numina');before=len(nr);nr=exclude(nr,blocked)
    num,na=prepare_questions(nr,replace(c,max_questions=a.train+a.val+a.test),backend)
    assign(num,a.train,a.val,a.test)
    qs=[dict(q,domain=d,id=d+'-'+q['id']) for d,rows in [('numina',num),('gsm8k',gsm)] for q in rows]
    families=[q['family'] for q in qs]
    if len(families)!=len(set(families)):raise ValueError('Family overlap')
    a.out.mkdir(parents=True);write_json(a.out/'questions.json',qs)
    write_json(a.out/'manifest.json',dict(protocol='slice-transfer-v1',code=code_hashes(),
        questions_sha256=sha256(a.out/'questions.json'),model_path=str(model),model_revision=rev,
        model_hashes=model_identity(model),verifier=grader.provenance,config=c.__dict__,shards=a.shards,layers=select_layers(28)+[18],
        train_questions=a.train,val_questions=a.val,test_questions=a.test,seeds=list(SEEDS),ridge_epsilons=list(EPSILONS),
        datasets=dict(numina=ni,gsm_train=ti,gsm_test=ei),eligibility=dict(numina=nx,
        cross_domain_removed=before-len(nr),numina_selected=na,gsm_train=tx,gsm_test=ex,gsm_selected=ga,gsm_test_selected=gb),
        scope='Offline sampled-prefix Ridge; numeric eligible subsets; fresh frozen-Qwen answers. Template exclusion is not semantic/pretraining decontamination. No RL.'))
    print('Prepared',len(qs),'questions;',len(qs)*c.responses_per_question,'answers',flush=True)


def generate(out,task):
    m=verify(out)
    if not 0<=task<m['shards']:raise ValueError('Invalid shard')
    c=TrainingConfig(**m['config']);grader=VerificationClient(c.verifier_python);grader.check()
    if grader.provenance!=m['verifier']:raise ValueError('Verifier identity changed after preparation')
    backend=TransformersBackend(m['model_path'])
    if backend.model_files!=m['model_hashes']:raise ValueError('Model changed')
    qs=[q for i,q in enumerate(read_json(out/'questions.json')) if i%m['shards']==task]
    folder=out/'shards'/str(task);_generate(c,folder,qs,backend,grader);counts=Counter()
    for q in qs:
        for r in _read_record(folder/'trajectories'/f'{q["id"]}.json')['responses']:
            counts[(q['domain'],q['split'],r['verifier_status'])]+=1
    for d in ('numina','gsm8k'):
        for s in ('train','val','test'):
            chosen=[q for q in qs if q['domain']==d and q['split']==s]
            if chosen:save_tensor(folder/f'{d}-{s}.pt',_sampled(folder,chosen,c,backend,m['layers'],s))
    write_json(folder/'complete.json',dict(manifest_sha256=sha256(out/'manifest.json'),
        label_statuses={'/'.join(k):v for k,v in counts.items()},caches={p.name:sha256(p) for p in sorted(folder.glob('*.pt'))}))


def pack(out):
    m=verify(out);mh=sha256(out/'manifest.json');records={}
    for t in range(m['shards']):
        folder=out/'shards'/str(t);done=read_json(folder/'complete.json')
        if done['manifest_sha256']!=mh:raise ValueError('Shard identity mismatch')
        for n,h in done['caches'].items():
            if sha256(folder/n)!=h:raise ValueError('Changed shard cache')
    for d in ('numina','gsm8k'):
        for s in ('train','val','test'):
            pieces=[load_tensor(out/'shards'/str(t)/f'{d}-{s}.pt') for t in range(m['shards']) if (out/'shards'/str(t)/f'{d}-{s}.pt').exists()]
            if not pieces:raise ValueError('Empty split')
            data={k:torch.cat([v[k] for v in pieces]) if k in ('x','y') else sum([v[k] for v in pieces],[]) for k in pieces[0]}
            if not torch.isfinite(data['x']).all() or set(data['y'].tolist())!={0.,1.}:raise ValueError(f'{d}/{s} requires both outcomes; enlarge sample')
            path=out/'packed'/f'{d}-{s}.pt';save_tensor(path,data);records[path.name]=sha256(path)
    write_json(out/'packed/complete.json',dict(manifest_sha256=mh,caches=records))


def data_split(out,d,s):
    done=read_json(out/'packed/complete.json');p=out/'packed'/f'{d}-{s}.pt'
    if done['manifest_sha256']!=sha256(out/'manifest.json') or sha256(p)!=done['caches'][p.name]:raise ValueError('Packed identity mismatch')
    return load_tensor(p)


class Combined(nn.Module):
    def __init__(self):
        super().__init__();self.projection=nn.Linear(1536,768);self.readout=nn.Linear(768,1)
    def features(self,x):return self.projection(x)
    def forward(self,x):
        z=self.features(x);return self.readout(z).flatten(),z


def fit_features(tr,va,width,seed,epochs,device,combined=False):
    if set(tr['y'].tolist())!={0.,1.} or set(va['y'].tolist())!={0.,1.}:raise ValueError('Both classes required')
    torch.manual_seed(seed);mean,scale=isotropic_normalization(tr['x'])
    model=(Combined() if combined else Bottleneck(tr['x'].shape[1],width)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.001);g=torch.Generator().manual_seed(seed)
    best=float('inf');saved=None;history=[]
    for epoch in range(epochs):
        model.train()
        for ids in torch.randperm(len(tr['y']),generator=g).split(2048):
            logits,z=model(((tr['x'][ids]-mean)/scale).to(device));objective=F.binary_cross_entropy_with_logits(logits,tr['y'][ids].to(device))
            if not combined:objective=objective+.01*z.mean()
            if not torch.isfinite(objective):raise ValueError('Nonfinite objective')
            opt.zero_grad();objective.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        with torch.no_grad():pred=torch.cat([model(((x-mean)/scale).to(device))[0].sigmoid().cpu() for x in va['x'].split(2048)])
        score=float((pred-va['y']).square().mean());history.append(score)
        print(f'Feature epoch {epoch+1}/{epochs}: validation Brier {score:.6f}',flush=True)
        if score<best:best=score;saved={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    return dict(state=saved,mean=mean,scale=scale,width=width,input_dim=tr['x'].shape[1],combined=combined,history=history)


@torch.no_grad()
def transform(r,x,device='cpu'):
    model=Combined() if r['combined'] else Bottleneck(r['input_dim'],r['width'])
    model.load_state_dict(r['state']);model.to(device).eval()
    return torch.cat([model.features(((v-r['mean'])/r['scale']).to(device)).cpu() for v in x.split(2048)])


def question_brier(pred,y,groups):
    errors=(pred.double()-y.double()).square().numpy();sums={};counts={}
    for q,e in zip(groups,errors):sums[q]=sums.get(q,0.)+e;counts[q]=counts.get(q,0)+1
    return np.array([sums[q]/counts[q] for q in sorted(sums)])


def ridge_fit(x,y,vx,vy,vg):
    mean,scale=isotropic_normalization(x)
    def design(z):return torch.cat([((z-mean)/scale).double(),torch.ones(len(z),1,dtype=torch.float64)],1)
    t=design(x);v=design(vx);a=t.T@t;b=t.T@y.double();best=None
    for eps in EPSILONS:
        system=a+eps*torch.eye(len(a),dtype=a.dtype);w=torch.linalg.solve(system,b)
        score=float(question_brier(v@w,vy,vg).mean())
        if best is None or score<best['validation_brier']:best=dict(w=w,mean=mean,scale=scale,epsilon=eps,validation_brier=score,condition=float(torch.linalg.cond(system)))
    return best


def ridge_predict(h,x):return ((x-h['mean'])/h['scale']).double()@h['w'][:-1]+h['w'][-1]


def pool(a,b,budget,seed):
    result={k:[] for k in a}
    for i,d in enumerate((a,b)):
        qs=sorted(set(d['question']));count=budget//2 if i==0 else budget-budget//2
        if len(qs)<count:raise ValueError('Insufficient pool questions')
        chosen=set(np.random.default_rng(seed+i).choice(qs,count,replace=False));ids=[j for j,q in enumerate(d['question']) if q in chosen]
        for k in result:result[k].append(d[k][ids] if k in ('x','y') else [d[k][j] for j in ids])
    return {k:torch.cat(v) if k in ('x','y') else sum(v,[]) for k,v in result.items()}


def fit(out,task,device):
    m=verify(out);seed=SEEDS[task];folder=out/'fits'/str(seed)
    if (folder/'locked.json').exists():raise ValueError('Fit already locked')
    data={d:{s:data_split(out,d,s) for s in ('train','val')} for d in ('numina','gsm8k')}
    for domain in data:
        for s in data[domain]:data[domain][s]['x']=data[domain][s]['x'][:,:7680]
    pooled={s:pool(data['numina'][s],data['gsm8k'][s],m['train_questions'] if s=='train' else m['val_questions'],seed) for s in ('train','val')}
    reps={}
    for name,domain,width in [('source','numina',768),('target','gsm8k',768),('pooled','pool',768),('wide','pool',1536)]:
        d=pooled if domain=='pool' else data[domain]
        print('Fitting',name,'seed',seed,'width',width,flush=True)
        reps[name]=fit_features(d['train'],d['val'],width,seed,m['config']['epochs'],device)
    target=data['gsm8k'];joined={s:dict(target[s],x=torch.cat([transform(reps[n],target[s]['x'],device) for n in ('source','target')],1)) for s in ('train','val')}
    reps['combined']=fit_features(joined['train'],joined['val'],768,seed,m['config']['epochs'],device,True)
    heads={}
    for arm in ARMS:
        domain='numina' if arm=='full_transfer' else 'gsm8k';d=data[domain];xs=[]
        for s in ('train','val'):
            if arm=='rms_two_thirds':x=data_split(out,domain,s)['x'][:,7680:]
            elif arm=='combined_768':x=transform(reps['combined'],joined[s]['x'],device)
            else:
                name={'full_transfer':'source','feature_transfer':'source','target_slice':'target','pooled_slice':'pooled','wide_pooled':'wide'}[arm]
                x=transform(reps[name],d[s]['x'],device)
            xs.append(x)
        heads[arm]=ridge_fit(xs[0],d['train']['y'],xs[1],d['val']['y'],d['val']['question'])
    save_tensor(folder/'models.pt',dict(representations=reps,heads=heads))
    write_json(folder/'locked.json',dict(seed=seed,manifest_sha256=sha256(out/'manifest.json'),model_sha256=sha256(folder/'models.pt'),packed_sha256=sha256(out/'packed/complete.json'),scope='No test tensors loaded. Source-only validation for full-transfer critic. Target validation for adapted heads.'))


def evaluate(out):
    m=verify(out);test=data_split(out,'gsm8k','test');source_test=data_split(out,'numina','test')
    errors={a:[] for a in ARMS};rows={};source_context={}
    for seed in SEEDS:
        f=out/'fits'/str(seed);locked=read_json(f/'locked.json')
        if (locked['manifest_sha256']!=sha256(out/'manifest.json') or locked['model_sha256']!=sha256(f/'models.pt')
                or locked['packed_sha256']!=sha256(out/'packed/complete.json')):raise ValueError('Locked model or packed data changed')
        state=load_tensor(f/'models.pt');reps=state['representations'];x=test['x'][:,:7680]
        source_z=transform(reps['source'],source_test['x'][:,:7680])
        source_pred=ridge_predict(state['heads']['full_transfer'],source_z)
        source_context[str(seed)]=float(question_brier(source_pred,source_test['y'],source_test['question']).mean())
        z={n:transform(r,x) for n,r in reps.items() if n!='combined'}
        z['combined']=transform(reps['combined'],torch.cat([z['source'],z['target']],1))
        for a in ARMS:
            features=test['x'][:,7680:] if a=='rms_two_thirds' else z[{'full_transfer':'source','feature_transfer':'source','target_slice':'target','pooled_slice':'pooled','wide_pooled':'wide','combined_768':'combined'}[a]]
            pred=ridge_predict(state['heads'][a],features);qe=question_brier(pred,test['y'],test['question']);errors[a].append(qe)
            rows[f'{a}/{seed}']=dict(question_brier=float(qe.mean()),raw_brier=float((pred-test['y']).square().mean()),clipped_brier=float((pred.clamp(0,1)-test['y']).square().mean()),accuracy=float(((pred>=.5)==test['y'].bool()).float().mean()),epsilon=state['heads'][a]['epsilon'],condition=state['heads'][a]['condition'])
            save_tensor(out/'predictions'/f'{a}-{seed}.pt',dict(prediction=pred,y=test['y'],question=test['question']))
    comparisons={};rng=np.random.default_rng(42)
    for a,b in [('full_transfer','feature_transfer'),('feature_transfer','target_slice'),('pooled_slice','target_slice'),('combined_768','pooled_slice'),('combined_768','wide_pooled'),('target_slice','rms_two_thirds')]:
        delta=np.array(errors[a])-np.array(errors[b]);draws=np.array([delta[rng.integers(3,size=3)][:,rng.integers(delta.shape[1],size=delta.shape[1])].mean() for _ in range(4000)])
        comparisons[a+'_minus_'+b]=dict(mean=float(delta.mean()),descriptive_95_interval=np.quantile(draws,[.025,.975]).tolist())
    summary={a:float(np.mean(errors[a])) for a in ARMS}
    write_json(out/'summary.json',dict(question_mean_brier=summary,per_seed=rows,comparisons=comparisons,prefixes=len(test['y']),questions=len(set(test['question'])),source_in_domain_question_brier=source_context,scope=m['scope'],limitations='Three compressor seeds; six exploratory unadjusted intervals. Retained verifiable answers only; prefixes not independent. Target labels used for adapted arms, not zero-shot critic. Pooled question budgets matched; ensemble uses two specialists plus combiner. No PPO or cost-efficiency claim.'))
    (out/'report.txt').write_text('\n'.join(['Offline Numina -> GSM8K transfer; lower question-mean raw Brier is better.',*[f'{a:22} {v:.6f}' for a,v in summary.items()],*[f'{k}: {v}' for k,v in comparisons.items()],'Inspect summary.json, predictions and shard label_statuses. Not an RL comparison.'])+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','generate','pack','fit','evaluate'],required=True)
    p.add_argument('--task',type=int,default=0);p.add_argument('--model');p.add_argument('--verifier-python',default=str(Path('.venv-ppo-verifier/bin/python').absolute()));p.add_argument('--shards',type=int,default=4)
    for name,default in [('train',2000),('val',500),('test',1000),('answers',4),('tokens',1024),('epochs',30)]:p.add_argument('--'+name,type=int,default=default)
    a=p.parse_args();a.out=a.out.resolve();torch.set_num_threads(8)
    if min(a.train,a.val,a.test)<20 or min(a.shards,a.answers,a.tokens,a.epochs)<1:p.error('Question counts >=20 and positive budgets required')
    start=time.monotonic()
    try:
        if a.stage=='prepare':prepare(a)
        elif a.stage=='generate':generate(a.out,a.task)
        elif a.stage=='pack':pack(a.out)
        elif a.stage=='fit':
            if not 0<=a.task<3:raise ValueError('Fit task 0..2')
            fit(a.out,a.task,'cuda')
        else:evaluate(a.out)
        write_json(a.out/'status'/f'{a.stage}-{a.task}.json',dict(state='complete',seconds=time.monotonic()-start))
    except Exception as e:
        if a.stage!='prepare' and a.out.exists():write_json(a.out/'status'/f'{a.stage}-{a.task}.json',dict(state='failed',error=str(e)))
        raise
if __name__=='__main__':main()
