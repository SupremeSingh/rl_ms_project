"""Locked test-only evaluation of saved actors; no training or source writes."""
import argparse
import json
from pathlib import Path
import time
import os
import subprocess
import numpy as np
from .provenance import sha256, snapshot, write_json
from .runner import ROOT
from slice_critic.storage import model_identity

ARMS=['slice-token','rms-layer-token','slice-chunk','rms-layer-chunk','ppo','grpo']
PRIMARY=[('slice-chunk','slice-token'),('slice-chunk','ppo')]
def read(p): return json.loads(Path(p).read_text())

def prepare(source,out):
    source=source.resolve(); out=out.resolve()
    if out==source or source in out.parents: raise ValueError('Use a separate output directory')
    parent=read(source/'manifest.json'); verify=read(source/'main/verification.json')
    from .ppo_reward import verifier_command
    verifier=json.loads(subprocess.check_output(verifier_command()+['--provenance'],text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))))
    if not verify['passed'] or verify['manifest_sha256']!=sha256(source/'manifest.json'): raise ValueError('Source verification failed')
    if parent['seeds']!=[1101,1102,1103] or parent['steps']!=120: raise ValueError('Unexpected source protocol')
    cases=[]; common=None
    for seed in parent['seeds']:
        child=source/f'seed{seed}'; m=read(child/'manifest.json')
        if sha256(child/'manifest.json')!=parent['children'][str(seed)]: raise ValueError('Changed child')
        test=child/'data/test.parquet'; questions=child/'data/questions.json'
        for p in [test,questions]:
            if sha256(p)!=m['data_hashes'][str(p.relative_to(child))]: raise ValueError('Changed data')
        rows=[q for q in read(questions)['questions'] if q['split']=='test']
        if len(rows)!=2000 or len({q['id'] for q in rows})!=2000: raise ValueError('Need 2000 distinct test questions')
        identity=[(q['id'],q['question'],q['ground_truth']) for q in rows]
        if common is None: common=identity
        if common!=identity: raise ValueError('Test questions differ')
        for arm in ARMS:
            case=child/f'{arm}-seed{seed}'; actor=case/'global_step_120/actor/huggingface'
            result=read(case/'result.json'); cfg=read(case/'run.json')['config']
            if read(case/'run.json')['verifier']!=verifier:raise ValueError('Verifier differs from original training')
            if result['updates']!=120 or result['answers']!=7680 or result['seed']!=seed: raise ValueError('Incomplete actor')
            if cfg['data']['max_response_length']!=2048 or cfg['actor_rollout_ref']['rollout']['val_kwargs']['do_sample']: raise ValueError('Different evaluation recipe')
            cases.append(dict(arm=arm,seed=seed,path=str(actor),identity=model_identity(actor)))
    cases.append(dict(arm='base',seed=0,path=str(ROOT/'models/qwen-math'),identity=model_identity(ROOT/'models/qwen-math')))
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'manifest.json',dict(protocol='saved-actor-test-v1',source=str(source),source_manifest_sha256=sha256(source/'manifest.json'),
        seeds=parent['seeds'],cases=cases,questions=rows,primary=PRIMARY,runtime=snapshot(ROOT)['files'],
        verifier=verifier,response_cap=2048,max_model_len=2560,scope='Greedy evaluation on reused inspected test questions; no selection or training.'))

def validate(out):
    m=read(out/'manifest.json')
    if m['runtime']!=snapshot(ROOT)['files']: raise ValueError('Runtime changed; start a new follow-up')
    if sha256(Path(m['source'])/'manifest.json')!=m['source_manifest_sha256']: raise ValueError('Source changed')
    return m

def evaluate(out,m,task,smoke):
    import torch
    from vllm import LLM, SamplingParams
    from .value_audit_gpu import load_tokenizer
    from .ppo_reward import compute_score,verifier_command
    from slice_critic.config import completion_prompt
    case=m['cases'][task]; actor=Path(case['path'])
    verifier=json.loads(subprocess.check_output(verifier_command()+['--provenance'],text=True,env=dict(os.environ,PYTHONPATH=str(ROOT/'src'))))
    if verifier!=m['verifier']:raise ValueError('Verifier changed after preparation')
    if model_identity(actor)!=case['identity']: raise ValueError('Actor weights changed')
    phase='smoke' if smoke else 'test'; folder=out/phase; folder.mkdir(exist_ok=True)
    target=folder/f'{task}.json'
    if target.exists(): raise FileExistsError('Existing result retained; do not silently overwrite')
    tokenizer=load_tokenizer(actor)
    rows=m['questions'][:8] if smoke else m['questions']
    prompts=[tokenizer.encode(completion_prompt(q['question']),add_special_tokens=False) for q in rows]
    if any(not 0<len(p)<=512 for p in prompts): raise ValueError('Test prompt exceeds cap; no silent exclusions')
    started=time.perf_counter()
    model=LLM(model=str(actor),tokenizer=str(ROOT/'models/qwen-math'),dtype='bfloat16',tensor_parallel_size=1,
        gpu_memory_utilization=.8,enforce_eager=True,max_model_len=2560,max_num_seqs=16,seed=case['seed'],trust_remote_code=False)
    initialization=time.perf_counter()-started; started=time.perf_counter(); answers=[]
    torch.cuda.synchronize()
    for start in range(0,len(rows),16):
        batch=rows[start:start+16]
        generated=model.generate([dict(prompt_token_ids=p) for p in prompts[start:start+16]],
            SamplingParams(n=1,temperature=0,top_p=1,top_k=-1,max_tokens=2048,detokenize=False,ignore_eos=False,
                           stop_token_ids=[tokenizer.eos_token_id]),use_tqdm=False)
        if len(generated)!=len(batch): raise ValueError('Missing answers')
        decoded=[];tokens=[];capped=[]
        for q,g in zip(batch,generated,strict=True):
            sample=g.outputs[0]; ids=list(sample.token_ids)
            if not ids or len(ids)>2048: raise ValueError('Empty/out-of-budget response')
            tokens.append(ids);capped.append(sample.finish_reason=='length');decoded.append(tokenizer.decode(ids,skip_special_tokens=True))
        graded=compute_score(['numina_numeric']*len(batch),decoded,[q['ground_truth'] for q in batch])
        for q,ids,text,cap,score in zip(batch,tokens,decoded,capped,graded,strict=True):
            answers.append(dict(score,id=q['id'],question=q['question'],token_ids=ids,response=text,capped=cap))
    torch.cuda.synchronize(); seconds=time.perf_counter()-started
    write_json(target,dict(manifest_sha256=sha256(out/'manifest.json'),case=case,answers=answers,
        accuracy=sum(r['score'] for r in answers)/len(answers),initialization_seconds=initialization,
        evaluation_seconds=seconds,evaluation_gpu_hours=seconds/3600,
        mean_response_tokens=sum(len(r['token_ids']) for r in answers)/len(answers),
        truncation_rate=sum(r['capped'] for r in answers)/len(answers),
        gpu=torch.cuda.get_device_name(),scope='One-GPU generation plus verification; setup separately timed, no utilization cost claim.'))

def gate(out,m):
    for i,c in enumerate(m['cases']):
        r=read(out/'smoke'/f'{i}.json')
        if r['manifest_sha256']!=sha256(out/'manifest.json') or r['case']!=c or [a['id'] for a in r['answers']]!=[q['id'] for q in m['questions'][:8]]: raise ValueError('Smoke failed')
        if any(a['score'] not in (0,1) or a['score']!=int(a['verifier_status']=='correct') for a in r['answers']):raise ValueError('Smoke verifier failed')
    write_json(out/'smoke.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json')))

def paired_accuracy(a,b,draws=10000):
    d=a-b; rng=np.random.default_rng(38291); boot=[]
    for _ in range(draws):
        si=rng.integers(len(d),size=len(d)); qi=rng.integers(d.shape[1],size=d.shape[1])
        boot.append(d[np.ix_(si,qi)].mean()*100)
    return dict(mean_pp=float(d.mean()*100),seed_differences_pp=(d.mean(1)*100).tolist(),
                adjusted_interval_pp=np.quantile(boot,[.0125,.9875]).tolist(),scope='Paired seed/question bootstrap; two Bonferroni primary contrasts; only three seeds.')

def report(out,m):
    results=[read(out/'test'/f'{i}.json') for i in range(len(m['cases']))]
    expected=[q['id'] for q in m['questions']]; arrays={}; metrics={}
    for c,r in zip(m['cases'],results,strict=True):
        if r['manifest_sha256']!=sha256(out/'manifest.json') or r['case']!=c or [a['id'] for a in r['answers']]!=expected: raise ValueError('Mismatched test result')
        if any(a['score'] not in (0,1) or a['score']!=int(a['verifier_status']=='correct') for a in r['answers']): raise ValueError('Invalid scoring')
    for arm in ARMS:
        from collections import Counter
        rows=[r for r in results if r['case']['arm']==arm]
        arrays[arm]=np.array([[a['score'] for a in r['answers']] for r in rows])
        acc=arrays[arm].mean(1)*100
        metrics[arm]=dict(accuracy_mean_percent=float(acc.mean()),accuracy_seed_sd_pp=float(acc.std(ddof=1)),accuracy_by_seed_percent=dict(zip(map(str,m['seeds']),acc.tolist())),
            mean_response_tokens=float(np.mean([r['mean_response_tokens'] for r in rows])),truncation_rate=float(np.mean([r['truncation_rate'] for r in rows])),
            evaluation_gpu_hours_mean=float(np.mean([r['evaluation_gpu_hours'] for r in rows])),
            verifier_status_counts=dict(Counter(a['verifier_status'] for r in rows for a in r['answers'])))
    contrasts={a+'_minus_'+b:paired_accuracy(arrays[a],arrays[b]) for a,b in PRIMARY}
    base=results[-1]['accuracy']*100
    for v in metrics.values(): v['gain_over_base_pp']=v['accuracy_mean_percent']-base
    write_json(out/'summary.json',dict(metrics=metrics,primary=contrasts,base_accuracy_percent=base,
        scope=m['scope'],test_questions=2000,actor_runs=18,uncertainty='Three seeds; no universal superiority, equivalence or optimization stability claim.'))
    lines=['Final saved-actor TEST evaluation: 3 seeds, 2000 shared questions, greedy cap 2048.',f'Fresh base-model reference: {base:.2f}%',
           'method                  test mean    seed SD    eval GPUh']
    lines += [f'{a:<24}{v["accuracy_mean_percent"]:.2f}%       {v["accuracy_seed_sd_pp"]:.2f} pp     {v["evaluation_gpu_hours_mean"]:.3f}' for a,v in metrics.items()]
    lines += [f'{k}: {v}' for k,v in contrasts.items()]+[m['scope'],'Inference timing is not training timing.']
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for i,a in enumerate(ARMS):
        v=metrics[a]; ax.errorbar(i,v['accuracy_mean_percent'],yerr=v['accuracy_seed_sd_pp'],fmt='o',capsize=5,color='#234E70')
        ax.scatter([i-.12,i,i+.12],list(v['accuracy_by_seed_percent'].values()),marker='x',color='#D17A22',zorder=3)
    ax.axhline(base,ls='--',color='gray',label=f'Base {base:.2f}%'); ax.set(xticks=range(6),xticklabels=ARMS,ylabel='Greedy test accuracy (%)',title='Final checkpoint: mean ± seed SD; crosses show three actor seeds')
    ax.tick_params(axis='x',rotation=20);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'test-accuracy.png',dpi=180);fig.savefig(out/'test-accuracy.pdf');plt.close(fig)
    write_json(out/'verification.json',dict(passed=True,manifest_sha256=sha256(out/'manifest.json'),files={p:sha256(out/p) for p in ['summary.json','report.txt','test-accuracy.png','test-accuracy.pdf']}))

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--stage',choices=['prepare','smoke','gate','test','report'],required=True)
    p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--task',type=int,default=0)
    a=p.parse_args();out=a.out.resolve()
    if a.stage=='prepare': prepare(a.source,out);return
    m=validate(out)
    if a.stage in ['test','report']:
        g=read(out/'smoke.json')
        if not g['passed'] or g['manifest_sha256']!=sha256(out/'manifest.json'): raise ValueError('Smoke gate missing')
    if a.stage=='gate':gate(out,m)
    elif a.stage=='report':report(out,m)
    else:evaluate(out,m,a.task,a.stage=='smoke')

if __name__=='__main__':main()
