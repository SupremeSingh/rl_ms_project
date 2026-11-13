"""Locked offline Numina comparison; invokes official LLM2Vec trainers unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'packages/slice-critic/src')]
import torch
from slice_critic.storage import sha256,write_json,model_identity
from math_rl.encoder_features import block_features,prefix_inputs,attention_probe,require_bidirectional

OFFICIAL='6bbd52528bee4936786ff0e9eb8a569698b1c731'
ARMS=('causal','bidir','adapted')
LAYERS=[27,26,24,20,12]


def read(p):return json.loads(Path(p).read_text())


def identity():
    paths=[Path(__file__),Path(__file__).with_name('official_train.py'),ROOT/'src/math_rl/encoder_features.py']
    paths+=list((ROOT/'packages/slice-critic/src').rglob('*.py'))
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}


def select_question_ids(rows,limit,outcomes=None):
    ids=sorted({r['id'] for r in rows},
               key=lambda q:hashlib.sha256(str(q).encode()).hexdigest())
    if len(ids)<limit:raise ValueError('Insufficient source questions')
    selected=ids[:limit]
    if outcomes is not None:
        labels=set().union(*(outcomes(q) for q in selected))
        if labels!={0,1}:
            # Smoke-only repair: retain whole questions and all their answers.
            for q in ids[limit:]:
                if outcomes(q)-labels:
                    for i in reversed(range(len(selected))):
                        trial=selected[:];trial[i]=q
                        combined=set().union(*(outcomes(x) for x in trial))
                        if combined>=labels|outcomes(q):
                            selected=trial;labels=combined;break
                    if labels=={0,1}:break
            if labels!={0,1}:
                raise ValueError('Smoke split needs correct and incorrect nonempty saved answers')
    return selected


def prepare(source,numina,sanity,out,official,smoke):
    # Detect optional-container import collisions before reserving adaptation time.
    subprocess.run([sys.executable,str(Path(__file__).with_name('official_train.py')),'--check'],check=True)
    gate=read(sanity/'report.json')
    if not gate.get('passed') or gate.get('execution')!='unpadded-prefix':
        raise ValueError('Passed unpadded encoder sanity required')
    commit=subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()
    if commit!=OFFICIAL or subprocess.check_output(['git','-C',str(official),'status','--porcelain'],text=True).strip():
        raise ValueError('Require clean pinned official checkout')
    verification=read(source/'verification.json')
    if not verification.get('passed'):raise ValueError('Passed fixed-policy source required')
    for name,digest in verification['files'].items():
        if sha256(source/name)!=digest:raise ValueError('Source verification artifact changed')
    from math_rl.fixed_policy import items,answer
    m=read(source/'manifest.json')
    all_rows=items(source,m)
    questions=read(numina/'data/questions.json')['questions']
    allowed={q['id']:q for q in questions}
    source_rows=[]
    limits={'train':8,'val':4,'test':4} if smoke else {'train':500,'val':100,'test':50}
    for split,limit in limits.items():
        candidates=[r for r in all_rows if r['split']==split]
        cache={}
        def saved(row):
            if row['key'] not in cache:cache[row['key']]=answer(source,row)
            return cache[row['key']]
        def outcomes(q):
            return {saved(r)['score'] for r in candidates
                    if r['id']==q and saved(r)['token_ids']}
        ids=select_question_ids(candidates,limit,outcomes if smoke and split!='test' else None)
        for row in all_rows:
            if row['split']!=split or row['id'] not in ids:continue
            q=allowed.get(row['id'])
            if q is None or q['split']!=split or str(q['ground_truth'])!=str(row['ground_truth']):
                raise ValueError('Saved answer source differs from existing Numina split')
            data=saved(row)
            if data['score'] not in (0,1) or data['score']!=int(data['verifier_status']=='correct'):
                raise ValueError('Invalid saved verifier outcome')
            source_rows.append(dict(key=row['key'],question_id=row['id'],split=split,
                prompt_ids=data['prompt_ids'],token_ids=data['token_ids'],score=data['score'],
                source_file=str(source/'answers'/f'{row["key"]}.json'),
                source_sha256=sha256(source/'answers'/f'{row["key"]}.json')))
    for split in ('train','val'):
        if {r['score'] for r in source_rows if r['split']==split and r['token_ids']}!={0,1}:
            raise ValueError(f'{split} selection needs correct and incorrect nonempty saved answers')
    model=ROOT/'models/qwen-math'
    if model_identity(model)!=m['base_identity']:raise ValueError('Base Qwen differs from saved study')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'rows.json',source_rows)
    record=dict(protocol='encoder-offline-v1',smoke=smoke,source=str(source),numina=str(numina),
        official=str(official),official_commit=OFFICIAL,model=str(model),model_identity=model_identity(model),
        source_manifest_sha256=sha256(source/'manifest.json'),sanity_sha256=sha256(sanity/'report.json'),
        rows_sha256=sha256(out/'rows.json'),runtime=identity(),arms=list(ARMS),
        seeds=[42] if smoke else [42,43,44],adaptation_steps=2 if smoke else 500,
        compressor_epochs=2 if smoke else 30,epsilon=.01,chunk_fallback_tokens=32,
        question_selection='SHA256 order; smoke train/val replace final question if needed for both outcome classes; test unchanged',
        scope='Offline supervised prefix correctness, not MC value MAE or PPO. Same inspected Numina splits; reused fixed-policy sampled answers.')
    write_json(out/'manifest.json',record)


def validate(out):
    m=read(out/'manifest.json')
    if m['runtime']!=identity() or m['rows_sha256']!=sha256(out/'rows.json'):
        raise ValueError('Code or locked rows changed')
    if model_identity(Path(m['model']))!=m['model_identity']:raise ValueError('Base weights changed')
    for r in read(out/'rows.json'):
        if sha256(r['source_file'])!=r['source_sha256']:raise ValueError('Source answer changed')
    official=Path(m['official'])
    if subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()!=OFFICIAL:
        raise ValueError('Official revision changed')
    if subprocess.check_output(['git','-C',str(official),'status','--porcelain'],text=True).strip():
        raise ValueError('Official code modified')
    return m


def load(m,arm):
    from llm2vec import LLM2Vec
    base=m['model'] if arm!='adapted' else str(Path(m['out'])/'adapted-model')
    wrapper=LLM2Vec.from_pretrained(base,enable_bidirectional=arm!='causal',
        torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2')
    model=wrapper.model.cuda().eval().requires_grad_(False)
    return model,wrapper.tokenizer


def corpus(rows,tokenizer,length=128):
    lines=[]
    for r in rows:
        if r['split']!='train':continue
        ids=r['prompt_ids']+r['token_ids']
        for start in range(0,len(ids)-length+1,length):
            text=' '.join(tokenizer.decode(ids[start:start+length],skip_special_tokens=True).splitlines()).strip()
            if not text or '!@#$%^&*()' in text:continue
            # Official SimCSE loader strips lines; preserve exact fixed lengths after that operation.
            if len(tokenizer.encode(text,add_special_tokens=True))==length:lines.append(text)
    if len(set(lines))<8:raise ValueError('Too few distinct equal-length training windows')
    return list(dict.fromkeys(lines))


def write_corpus(path,lines,tokenizer,length=128):
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Read exactly as the official Wiki1M loader does, including universal newlines.
    with path.open(encoding='utf-8') as f:loaded=[line.strip() for line in f]
    if loaded!=lines or not loaded or any(not line for line in loaded):
        raise ValueError('Adaptation text changed during line-file round trip or contains blank lines')
    if any(len(tokenizer.encode(line,add_special_tokens=True))!=length for line in loaded):
        raise ValueError('Adaptation file must contain only fixed-length tokenized lines')
    print(f'Adaptation corpus validated: {len(loaded)} nonempty {length}-token lines',flush=True)


def trainer_configs(m,out):
    steps=m['adaptation_steps']
    common=dict(do_train=True,do_eval=False,per_device_train_batch_size=2,
        gradient_accumulation_steps=4,gradient_checkpointing=True,bf16=True,
        torch_dtype='bfloat16',attn_implementation='flash_attention_2',
        max_steps=steps,stop_after_n_steps=steps,save_steps=steps,logging_steps=1,
        save_only_model=True,report_to=[],seed=42,learning_rate=3e-5,lora_r=16,
        overwrite_output_dir=False,dataloader_drop_last=True)
    mntp=dict(common,model_name_or_path=m['model'],train_file=str(out/'adaptation.txt'),
        validation_file=str(out/'adaptation.txt'),max_seq_length=128,line_by_line=False,
        mask_token_type='blank',data_collator_type='default',mlm_probability=.2,
        output_dir=str(out/'mntp'))
    sim=dict(common,model_name_or_path=str(out/'mntp-model'),bidirectional=True,
        dataset_name='Wiki1M',dataset_file_path=str(out/'adaptation.txt'),
        max_seq_length=128,pooling_mode='mean',simcse_dropout=.3,loss_scale=20,
        remove_unused_columns=False,output_dir=str(out/'simcse'))
    return mntp,sim


def adapt(out,m):
    from transformers import AutoTokenizer
    from llm2vec import LLM2Vec
    if (out/'adaptation.json').exists():raise ValueError('Adaptation already exists; no silent retraining')
    tokenizer=AutoTokenizer.from_pretrained(m['model'],local_files_only=True)
    lines=corpus(read(out/'rows.json'),tokenizer)
    write_corpus(out/'adaptation.txt',lines,tokenizer)
    configs=trainer_configs(m,out);times={}
    env=dict(os.environ,TOKENIZERS_PARALLELISM='false',WANDB_DISABLED='true')
    for stage,config,script in zip(('mntp','simcse'),configs,('run_mntp.py','run_simcse.py')):
        write_json(out/f'{stage}.json',config)
        started=time.perf_counter()
        with (out/f'{stage}.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('official_train.py')),
                            str(Path(m['official'])/'experiments'/script),str(out/f'{stage}.json')],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        adapter=out/stage if stage=='mntp' else out/stage/f'checkpoint-{m["adaptation_steps"]}'
        wrapper=LLM2Vec.from_pretrained(m['model'] if stage=='mntp' else str(out/'mntp-model'),
            peft_model_name_or_path=str(adapter),merge_peft=True,enable_bidirectional=True,
            torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2')
        dest=out/('mntp-model' if stage=='mntp' else 'adapted-model')
        wrapper.save(str(dest));del wrapper
        times[stage]=time.perf_counter()-started
    write_json(out/'adaptation.json',dict(seconds=times,model_identity=model_identity(out/'adapted-model'),
        training_windows=len(lines),text_sha256=sha256(out/'adaptation.txt'),
        training_question_ids=sorted({r['question_id'] for r in read(out/'rows.json') if r['split']=='train'}),
        official_commit=OFFICIAL))


def extract(out,m,arm):
    from slice_critic.chunks import token_chunks
    model,tokenizer=load(dict(m,out=str(out)),arm)
    ids=tokenizer.encode('Question: nine divided by three? Solution: divide nine by three.',add_special_tokens=False)
    sensitivity=attention_probe(model,ids)
    if arm=='causal':
        if sensitivity>1e-5:raise ValueError('Causal control is not causal')
    else:require_bidirectional(sensitivity,0.)
    # Gate the transformed feature API itself on each new backbone, including adapted weights.
    probe=block_features(model,[ids],LAYERS,tokenizer.pad_token_id)
    torch.testing.assert_close(probe,block_features(model,[ids],LAYERS,tokenizer.pad_token_id),rtol=0,atol=0)
    if arm=='adapted':
        if model_identity(out/'adapted-model')!=read(out/'adaptation.json')['model_identity']:
            raise ValueError('Adapted model changed')
    folder=out/arm;folder.mkdir(exist_ok=False)
    records=[];started=time.perf_counter()
    for r in read(out/'rows.json'):
        plan=token_chunks(tokenizer,r['token_ids'],'math',32)
        seq=prefix_inputs(r['prompt_ids'],r['token_ids'],list(plan.starts))
        x=block_features(model,seq,LAYERS,tokenizer.pad_token_id,batch_size=1)
        torch.save(dict(x=x,y=torch.full((len(x),),float(r['score'])),question=r['question_id'],
                        positions=list(plan.starts),split=r['split']),folder/f'{r["key"]}.pt')
        records.append(dict(file=f'{r["key"]}.pt',sha256=sha256(folder/f'{r["key"]}.pt')))
        print(arm,r['key'],r['split'],len(x),'chunks',flush=True)
    torch.cuda.synchronize()
    write_json(folder/'features.json',dict(files=records,seconds=time.perf_counter()-started,
        model_identity=model_identity(out/'adapted-model') if arm=='adapted' else m['model_identity'],
        bidirectional_sensitivity=sensitivity,manifest_sha256=sha256(out/'manifest.json')))


def feature_rows(out,arm):
    folder=out/arm;index=read(folder/'features.json')
    if index['manifest_sha256']!=sha256(out/'manifest.json'):raise ValueError('Stale features')
    result=[]
    for r in index['files']:
        if sha256(folder/r['file'])!=r['sha256']:raise ValueError('Changed features')
        result.append(torch.load(folder/r['file'],weights_only=True))
    return result


def sampled(rows,split):
    xs=[];ys=[]
    generator=torch.Generator().manual_seed(93821)
    for r in rows:
        if r['split']!=split:continue
        ids=torch.randperm(len(r['x']),generator=generator)[:8].sort().values
        xs.append(r['x'][ids]);ys.append(r['y'][ids])
    return dict(x=torch.cat(xs),y=torch.cat(ys))


def fit(out,m):
    from slice_critic.config import TrainingConfig
    from slice_critic.training import _fit_compressor,_model
    from slice_critic.bottleneck import encode
    from slice_critic.cumulative import raw_design
    from slice_critic.linear import fit_system
    for arm in ARMS:
        rows=feature_rows(out,arm)
        for seed in m['seeds']:
            folder=out/arm/f'fit-{seed}';folder.mkdir(exist_ok=False)
            config=TrainingConfig(dataset='locked Numina saved answers',seed=seed,
                                  epochs=m['compressor_epochs'])
            started=time.perf_counter()
            fitted=_fit_compressor(folder,sampled(rows,'train'),sampled(rows,'val'),config)
            compressor=_model(fitted)
            a=torch.zeros(769,769,dtype=torch.float64);b=torch.zeros(769,dtype=torch.float64)
            n=0
            for r in rows:
                if r['split']!='train':continue
                phi=raw_design(encode(compressor,r['x'],fitted['mean'],fitted['scale'],'cpu'))
                a+=phi.T@phi;b+=phi.T@r['y'].double();n+=len(phi)
            w,residual=fit_system(a,b,.01)
            torch.save(dict(a=a,b=b,weights=w,epsilon=.01,folds=1,states=n),folder/'ridge.pt')
            write_json(folder/'fit.json',dict(seconds=time.perf_counter()-started,residual=residual,
                compressor_sha256=sha256(folder/'compressor.pt'),head_sha256=sha256(folder/'ridge.pt')))


def evaluate(out,m):
    from slice_critic.training import _model
    from slice_critic.bottleneck import encode
    from slice_critic.cumulative import raw_design
    results={};question_errors={}
    for arm in ARMS:
        rows=feature_rows(out,arm);seed_results=[];question_errors[arm]=[]
        for seed in m['seeds']:
            folder=out/arm/f'fit-{seed}';audit=read(folder/'fit.json')
            if any(sha256(folder/file)!=audit[key] for file,key in [('compressor.pt','compressor_sha256'),('ridge.pt','head_sha256')]):raise ValueError('Fit changed')
            fitted=torch.load(folder/'compressor.pt',weights_only=True)
            model=_model(fitted);w=torch.load(folder/'ridge.pt',weights_only=True)['weights']
            grouped={};predictions=[]
            for r in rows:
                if r['split']!='test':continue
                z=encode(model,r['x'],fitted['mean'],fitted['scale'],'cpu')
                pred=raw_design(z)@w;error=(pred-r['y']).square()
                grouped.setdefault(r['question'],[]).extend(error.tolist())
                predictions.append(dict(question=r['question'],positions=r['positions'],prediction=pred.tolist(),label=r['y'].tolist()))
            q={q:sum(v)/len(v) for q,v in grouped.items()}
            seed_results.append(sum(q.values())/len(q));question_errors[arm].append(q)
            write_json(folder/'test-predictions.json',predictions)
        results[arm]=dict(question_mean_raw_brier=sum(seed_results)/len(seed_results),by_seed=seed_results,
            extraction_seconds=read(out/arm/'features.json')['seconds'])
    import numpy as np
    contrasts={};rng=np.random.default_rng(8913)
    for arm in ('bidir','adapted'):
        ids=sorted(question_errors['causal'][0])
        delta=np.array([[q1[q]-q0[q] for q in ids] for q1,q0 in zip(question_errors[arm],question_errors['causal'])])
        draws=[delta[np.ix_(rng.integers(len(delta),size=len(delta)),rng.integers(len(ids),size=len(ids)))].mean() for _ in range(2000)]
        contrasts[arm+'_minus_causal']=dict(mean=float(delta.mean()),interval_95=np.quantile(draws,[.025,.975]).tolist(),
            scope='Exploratory unadjusted paired compressor-seed/question bootstrap; one actor and one encoder adaptation seed')
    write_json(out/'summary.json',dict(results=results,contrasts=contrasts,question_errors=question_errors,scope=m['scope']))
    text=['Offline Numina encoder comparison; lower equal-question raw Brier is better.','Not PPO performance or MC-reference value MAE.']
    text += [f'{arm}: {r}' for arm,r in results.items()]
    text += [f'{name}: {r}' for name,r in contrasts.items()]
    text += ['Same SLICE training code/loss/normalization; same chunk states and labels; fresh heads.','Adaptation and extraction costs are additional. No test-based reselection.']
    (out/'report.txt').write_text('\n'.join(text)+'\n')
    write_json(out/'verification.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),
        files={f:sha256(out/f) for f in ['summary.json','report.txt']},scope='Integration/completeness, not a winner gate'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['prepare','adapt','extract','fit','evaluate'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--numina',type=Path)
    p.add_argument('--sanity',type=Path);p.add_argument('--official',type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--arm',choices=ARMS)
    a=p.parse_args();out=a.out.resolve();torch.set_num_threads(4)
    if a.stage=='prepare':
        if any(getattr(a,k) is None for k in ['source','numina','sanity','official']):p.error('prepare requires all source paths')
        prepare(a.source.resolve(),a.numina.resolve(),a.sanity.resolve(),out,a.official.resolve(),a.smoke);return
    m=validate(out)
    if a.stage=='extract':
        if a.arm is None:p.error('extract needs --arm')
        extract(out,m,a.arm)
    else:globals()[a.stage](out,m)


if __name__=='__main__':main()
