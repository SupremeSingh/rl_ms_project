"""Matched transfer clarification; validation selection precedes explicit test evaluation."""
import argparse
from dataclasses import replace
from pathlib import Path
import time
import numpy as np
import torch
import study as base
from slice_critic.config import TrainingConfig
from slice_critic.storage import read_json,write_json,save_tensor,load_tensor,sha256,model_identity
from slice_critic.data import load_rows,prepare_questions
from slice_critic.references import reference_decision
from slice_critic.text import family_key
from slice_critic.backend import TransformersBackend
from slice_critic.training import VerificationClient,_generate,_sampled
from slice_critic.features import isotropic_normalization

ROOT=Path(__file__).resolve().parents[2]
SEEDS=(42,43,44,45,46)
ARMS=('rms_two_thirds','transferred_slice','target_slice')
GRID=(.01,1.,10.,100.,1000.,10000.)
SUBJECTS=('algebra','counting_and_probability','geometry','intermediate_algebra','number_theory','prealgebra','precalculus')


def hashes():
    files=[Path(__file__),Path(__file__).with_name('submit_clarify.py'),Path(__file__).with_name('run_clarify.sh'),Path(base.__file__),*sorted((ROOT/'packages/slice-critic/src/slice_critic').glob('*.py'))]
    return {str(p.relative_to(ROOT)):sha256(p) for p in files}


def verify(out):
    m=read_json(out/'manifest.json')
    if m['code']!=hashes() or m['questions_sha256']!=sha256(out/'questions.json'):raise ValueError('Study inputs/code changed')
    source=Path(m['source'])
    if m['source_manifest_sha256']!=sha256(source/'manifest.json'):raise ValueError('Source manifest changed')
    return m


def math_rows(rows):
    good=[];excluded=[]
    for i,r in enumerate(rows):
        d=reference_decision(r['problem'],r['solution'],r.get('answer'))
        if d['eligible']:good.append(dict(problem=r['problem'],answer=d['numeric_reference']))
        else:excluded.append(dict(index=i,reasons=d['reasons']))
    return good,excluded


def prepare(a):
    if a.out.exists():raise ValueError('Use a new output directory')
    source=a.source.absolute();old=read_json(source/'manifest.json')
    if old['protocol']!='slice-transfer-v1' or read_json(source/'status/evaluate-0.json')['state']!='complete':raise ValueError('Completed transfer source required')
    c=TrainingConfig(**old['config']);grader=VerificationClient(c.verifier_python);grader.check()
    if grader.provenance!=old['verifier']:raise ValueError('Verifier changed')
    # Keep the saved model path spelling; never resolve cache aliases on the login host.
    if model_identity(old['model_path'])!=old['model_hashes']:raise ValueError('Model changed')
    from transformers import AutoTokenizer
    from slice_critic.config import completion_prompt
    tok=AutoTokenizer.from_pretrained(old['model_path'],local_files_only=True)
    class Backend:
        max_context=4096
        def prompt_ids(self,q):return tok.encode(completion_prompt(q),add_special_tokens=False)
    backend=Backend();native=[];identities=[];revision=None
    for subject in SUBJECTS:
        rows,identity=load_rows(replace(c,dataset='EleutherAI/hendrycks_math',dataset_config=subject,dataset_revision=revision,dataset_split='train'))
        revision=identity['revision'];identities.append(identity);native.extend(rows)
    test,identity=load_rows(replace(c,dataset='HuggingFaceH4/MATH-500',dataset_config=None,dataset_revision=None,dataset_split='test'))
    identities.append(identity)
    # Block every native MATH train/MATH-500 family from source representation fitting.
    math_families={family_key(r['problem']) for r in native+list(test)}
    oldq=read_json(source/'questions.json')
    if sha256(source/'questions.json')!=old['questions_sha256']:raise ValueError('Source questions changed')
    all_gsm={q['family'] for q in oldq if q['domain']=='gsm8k'}
    train_rows,train_ex=math_rows(native);test_rows,test_ex=math_rows(test)
    test_families={family_key(r['problem']) for r in test}
    train_rows=base.exclude(train_rows,test_families|all_gsm)
    tr,audit=prepare_questions(train_rows,replace(c,max_questions=a.train+a.val),backend)
    base.assign(tr,a.train,a.val,0)
    # Select all eligible numeric MATH-500 questions in full mode, not a fabricated 500 denominator.
    test_rows=base.exclude(test_rows,all_gsm)
    _,test_audit=prepare_questions(test_rows,replace(c,max_questions=20),backend)
    count=min(20,test_audit['eligible']) if a.smoke else test_audit['eligible']
    te,_=prepare_questions(test_rows,replace(c,max_questions=count),backend);base.assign(te,0,0,count)
    qs=[dict(q,domain='math',id='math-'+q['id']) for q in tr+te]
    a.out.mkdir(parents=True);write_json(a.out/'questions.json',qs)
    write_json(a.out/'manifest.json',dict(protocol='slice-transfer-clarify-v1',source=str(source),source_manifest_sha256=sha256(source/'manifest.json'),
        code=hashes(),questions_sha256=sha256(a.out/'questions.json'),config=c.__dict__,model_path=old['model_path'],model_hashes=old['model_hashes'],
        verifier=old['verifier'],layers=old['layers'],shards=a.shards,seeds=list(SEEDS),smoke=a.smoke,
        epochs=2 if a.smoke else 30,budgets=[20,40] if a.smoke else [100,500,2000],val_questions=a.val,
        math_blocked_families=sorted(math_families),math_datasets=identities,math_train_audit=audit,
        math_reference_exclusions=dict(train=train_ex,test=test_ex),math_test_audit=test_audit,math_test_questions=count,
        scope='Offline correctness prediction. GSM8K inspected/reused; MATH benchmark historically inspected. Numeric subset, not all MATH-500. Number-template exclusions are not semantic/pretraining decontamination.'))


def generate(out,task):
    m=verify(out)
    if not 0<=task<m['shards']:raise ValueError('Invalid shard')
    c=TrainingConfig(**m['config']);g=VerificationClient(c.verifier_python);g.check()
    if g.provenance!=m['verifier']:raise ValueError('Verifier changed')
    backend=TransformersBackend(m['model_path'])
    if backend.model_files!=m['model_hashes']:raise ValueError('Model changed')
    qs=[q for i,q in enumerate(read_json(out/'questions.json')) if i%m['shards']==task];folder=out/'shards'/str(task)
    _generate(c,folder,qs,backend,g)
    from collections import Counter
    counts=Counter()
    for q in qs:
        for r in base._read_record(folder/'trajectories'/f'{q["id"]}.json')['responses']:counts[q['split']+'/'+r['verifier_status']]+=1
    for s in ('train','val','test'):
        chosen=[q for q in qs if q['split']==s]
        if chosen:save_tensor(folder/f'math-{s}.pt',_sampled(folder,chosen,c,backend,m['layers'],s))
    write_json(folder/'complete.json',dict(manifest=sha256(out/'manifest.json'),statuses=dict(counts),caches={p.name:sha256(p) for p in folder.glob('*.pt')}))


def subset(d,questions):
    ids=[i for i,q in enumerate(d['question']) if q in questions]
    if not ids:raise ValueError('No retained prefixes in subset')
    return {k:v[ids] if k in ('x','y') else [v[i] for i in ids] for k,v in d.items()}


def ordering(d):return sorted(set(d['question']))


def pack(out):
    m=verify(out);records={};counts={};source=Path(m['source'])
    # Source and GSM cache reads verify their original checksums, not their obsolete code hashes.
    oldq=read_json(source/'questions.json');blocked=set(m['math_blocked_families'])
    source_ids={q['id'] for q in oldq if q['domain']=='numina' and q['family'] not in blocked}
    for domain in ('numina','gsm8k','math'):
        splits=('train','val') if domain=='numina' else ('train','val','test')
        for s in splits:
            if domain!='math':
                d=base.data_split(source,domain,s)
                if domain=='numina':d=subset(d,source_ids)
            else:
                pieces=[]
                for t in range(m['shards']):
                    f=out/'shards'/str(t);done=read_json(f/'complete.json')
                    if done['manifest']!=sha256(out/'manifest.json'):raise ValueError('Shard changed')
                    name=f'math-{s}.pt'
                    if name in done['caches']:
                        if sha256(f/name)!=done['caches'][name]:raise ValueError('Cache changed')
                        pieces.append(load_tensor(f/name))
                if not pieces:raise ValueError('Empty math split')
                d={k:torch.cat([p[k] for p in pieces]) if k in ('x','y') else sum([p[k] for p in pieces],[]) for k in pieces[0]}
            if set(d['y'].tolist())!={0.,1.}:raise ValueError(f'{domain}/{s} needs both classes')
            if not torch.isfinite(d['x']).all() or d['x'].shape[1]!=9216:raise ValueError('Invalid feature cache')
            p=out/'packed'/f'{domain}-{s}.pt';save_tensor(p,d);records[p.name]=sha256(p)
            counts[domain+'/'+s]=dict(questions=len(set(d['question'])),prefixes=len(d['y']),correct_fraction=float(d['y'].mean()))
    write_json(out/'packed/complete.json',dict(manifest=sha256(out/'manifest.json'),files=records,coverage=counts))


def data(out,domain,split):
    done=read_json(out/'packed/complete.json');p=out/'packed'/f'{domain}-{split}.pt'
    if done['manifest']!=sha256(out/'manifest.json') or done['files'][p.name]!=sha256(p):raise ValueError('Packed data changed')
    return load_tensor(p)


def ridge_grid(x,y,vx,vy,groups):
    mean,scale=isotropic_normalization(x)
    def design(v):return torch.cat([((v-mean)/scale).double(),torch.ones(len(v),1,dtype=torch.float64)],1)
    t=design(x);v=design(vx);a=t.T@t;b=t.T@y.double()
    eig,u=torch.linalg.eigh(a);eig=eig.clamp_min(0);ub=u.T@b
    heads={};scores={};grid=list(GRID)
    while True:
        for eps in grid:
            if eps in heads:continue
            w=u@(ub/(eig+eps));pred=v@w
            score=float(base.question_brier(pred,vy,groups).mean())
            heads[eps]=dict(w=w,mean=mean,scale=scale,epsilon=eps,condition=float((eig[-1]+eps)/(eig[0]+eps)),epsilon_per_prefix=eps/len(x))
            scores[eps]=score
        best=min(scores,key=scores.get)
        if best!=max(grid):break
        if max(grid)>=1e10:raise ValueError('Regularization still selects ceiling 1e10; test evaluation blocked')
        grid.append(max(grid)*10)
    return heads,best,scores


def fit(out,task,device):
    m=verify(out);seed=SEEDS[task];folder=out/'fits'/str(seed)
    if (folder/'locked.json').exists():raise ValueError('Already locked')
    src={s:data(out,'numina',s) for s in ('train','val')}
    maxbudget=max(m['budgets']);n=min(maxbudget,len(ordering(src['train'])));nv=min(m['val_questions'],len(ordering(src['val'])))
    if n<40 or nv<20:raise ValueError('Too few source families after overlap exclusions')
    sr={s:subset(src[s],set(ordering(src[s])[:n if s=='train' else nv])) for s in src}
    sr={s:dict(d,x=d['x'][:,:7680]) for s,d in sr.items()}
    source=base.fit_features(sr['train'],sr['val'],768,seed,m['epochs'],device)
    models={'source':source};heads={};selection={};sizes={}
    for domain in ('gsm8k','math'):
        tr=data(out,domain,'train');va=data(out,domain,'val');order=ordering(tr)
        if len(order)<maxbudget:raise ValueError(f'{domain} has fewer retained training families than largest head budget')
        rep_tr=subset(tr,set(order[:n]));rep_va=subset(va,set(ordering(va)[:nv]))
        target=base.fit_features(dict(rep_tr,x=rep_tr['x'][:,:7680]),dict(rep_va,x=rep_va['x'][:,:7680]),768,seed,m['epochs'],device);models[domain]=target
        for budget in m['budgets']:
            small=subset(tr,set(order[:budget]));sizes[domain+'/'+str(budget)]=len(set(small['question']))
            for arm in ARMS:
                if arm=='rms_two_thirds':x,vx=small['x'][:,7680:],va['x'][:,7680:]
                else:
                    r=source if arm=='transferred_slice' else target
                    x=base.transform(r,small['x'][:,:7680],device);vx=base.transform(r,va['x'][:,:7680],device)
                candidates,best,scores=ridge_grid(x,small['y'],vx,va['y'],va['question'])
                key=f'{domain}/{budget}/{arm}';heads[key]=candidates
                selection[key]=dict(epsilon=best,validation_question_brier=scores[best],grid=scores)
    save_tensor(folder/'models.pt',dict(representations=models,heads=heads))
    write_json(folder/'locked.json',dict(manifest=sha256(out/'manifest.json'),packed=sha256(out/'packed/complete.json'),models=sha256(folder/'models.pt'),selection=selection,
        source_training_questions=n,representation_validation_questions=nv,head_questions=sizes,
        note='Target-trained SLICE always uses the matched fixed representation budget, even on small Ridge learning-curve budgets. Those curves measure head adaptation, not total label use. No test tensors loaded.'))


def validate(out):
    m=verify(out);records={}
    for seed in SEEDS:
        f=out/'fits'/str(seed);r=read_json(f/'locked.json')
        if (r['manifest']!=sha256(out/'manifest.json') or r['packed']!=sha256(out/'packed/complete.json') or r['models']!=sha256(f/'models.pt')):raise ValueError('Locked fit changed')
        records[str(seed)]=r
    write_json(out/'validation.json',dict(manifest=sha256(out/'manifest.json'),fits=records))
    lines=['Validation only; no test predictions. All seeds locked.']
    lines += [f'{seed} {k}: epsilon={v["epsilon"]}, validation Brier={v["validation_question_brier"]:.6f}' for seed,r in records.items() for k,v in r['selection'].items()]
    (out/'validation-report.txt').write_text('\n'.join(lines)+'\n')


def score(pred,y,groups):
    p=pred.clamp(0,1);ece=0.
    for i in range(10):
        mask=(p>=i/10)&(p<(i+1)/10) if i<9 else (p>=.9)&(p<=1)
        if mask.any():ece+=float(mask.double().mean()*abs(p[mask].mean()-y[mask].double().mean()))
    return dict(question_brier=float(base.question_brier(pred,y,groups).mean()),question_clipped_brier=float(base.question_brier(p,y,groups).mean()),
                accuracy=float(((pred>=.5)==y.bool()).double().mean()),ece_10_bins=ece)


def evaluate(out):
    m=verify(out);validate(out);allrows={};errors={};baselines={}
    for domain in ('gsm8k','math'):
        test=data(out,domain,'test');train=data(out,domain,'train');groups=test['question']
        # A training-derived constant predictor; not the test outcome mean.
        for budget in m['budgets']:
            prior=float(subset(train,set(ordering(train)[:budget]))['y'].mean())
            baselines[f'{domain}/{budget}']=score(torch.full_like(test['y'],prior).double(),test['y'],groups)
        for seed in SEEDS:
            f=out/'fits'/str(seed);lock=read_json(f/'locked.json');state=load_tensor(f/'models.pt')
            zs={a:test['x'][:,7680:] if a=='rms_two_thirds' else base.transform(state['representations']['source' if a=='transferred_slice' else domain],test['x'][:,:7680]) for a in ARMS}
            for budget in m['budgets']:
                for arm in ARMS:
                    key=f'{domain}/{budget}/{arm}';chosen=lock['selection'][key]['epsilon']
                    for eps,h in state['heads'][key].items():
                        pred=base.ridge_predict(h,zs[arm]);row=score(pred,test['y'],groups)
                        allrows[f'{key}/{seed}/{eps}']=dict(row,selected=eps==chosen,condition=h['condition'])
                        if eps==chosen:errors.setdefault(key,[]).append(base.question_brier(pred,test['y'],groups))
                    pred=base.ridge_predict(state['heads'][key][chosen],zs[arm])
                    save_tensor(out/'predictions'/f'{domain}-{budget}-{arm}-{seed}.pt',dict(prediction=pred,y=test['y'],question=groups))
    paired={};rng=np.random.default_rng(2026)
    for domain in ('gsm8k','math'):
        for budget in m['budgets']:
            for a,b in [('transferred_slice','rms_two_thirds'),('transferred_slice','target_slice')]:
                delta=np.array(errors[f'{domain}/{budget}/{a}'])-np.array(errors[f'{domain}/{budget}/{b}'])
                draws=np.array([delta[rng.integers(5,size=5)][:,rng.integers(delta.shape[1],size=delta.shape[1])].mean() for _ in range(4000)])
                paired[f'{domain}/{budget}/{a}_minus_{b}']=dict(mean=float(delta.mean()),interval_95=np.quantile(draws,[.025,.975]).tolist())
    means={k:float(np.mean(v)) for k,v in errors.items()}
    write_json(out/'summary.json',dict(selected_question_brier=means,fixed_epsilon_metrics=allrows,constant_baselines=baselines,paired=paired,
        coverage=read_json(out/'packed/complete.json')['coverage'],scope=m['scope'],
        limitations='Five compressor/shuffle seeds, same answers. Twelve unadjusted exploratory intervals; fixed-grid test rows are diagnostics, not selection. Numeric eligible MATH-500 subset only. No PPO. Target SLICE has representation-training labels beyond small head budgets.'))
    (out/'report.txt').write_text('\n'.join(['Offline transfer clarification; selected equal-question raw Brier, lower is better.',*[f'{k}: {v:.6f}' for k,v in means.items()],*[f'{k}: {v}' for k,v in paired.items()],'Fixed epsilon diagnostics, clipped Brier, calibration and coverage in summary.json. No test-based reselection.'])+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['prepare','generate','pack','fit','validate','evaluate']);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--task',type=int,default=0);p.add_argument('--shards',type=int,default=4);p.add_argument('--smoke',action='store_true');p.add_argument('--train',type=int,default=2000);p.add_argument('--val',type=int,default=500)
    a=p.parse_args();a.out=a.out.absolute();torch.set_num_threads(8);start=time.monotonic()
    try:
        if a.stage=='prepare':
            if a.source is None:p.error('--source required')
            if a.smoke:a.train,a.val=40,20
            prepare(a)
        elif a.stage=='generate':generate(a.out,a.task)
        elif a.stage=='pack':pack(a.out)
        elif a.stage=='fit':
            if not 0<=a.task<5:raise ValueError('Fit task 0..4')
            fit(a.out,a.task,'cuda')
        elif a.stage=='validate':validate(a.out)
        else:evaluate(a.out)
        write_json(a.out/'status'/f'{a.stage}-{a.task}.json',dict(state='complete',seconds=time.monotonic()-start))
    except Exception as e:
        if a.stage!='prepare' and a.out.exists():write_json(a.out/'status'/f'{a.stage}-{a.task}.json',dict(state='failed',error=str(e)))
        raise
if __name__=='__main__':main()
