"""Read-only accounting of existing transfer runs. Never generate, fit, or rescore."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess


def read(path):return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def gpu_count(tres):
    values=dict(part.split('=',1) for part in tres.split(',') if '=' in part)
    if 'gres/gpu' in values:return int(values['gres/gpu'])
    typed=[int(v) for k,v in values.items() if k.startswith('gres/gpu:')]
    return sum(typed) if typed else 0


def parse_sacct(text):
    rows={}
    for line in text.splitlines():
        if not line.strip():continue
        fields=[f.strip() for f in line.split('|')]
        if len(fields)<8:raise ValueError('Unexpected sacct schema')
        raw,job,state,seconds,cpus,tres,start,end=fields[:8]
        if not re.fullmatch(r'\d+(?:_\d+)?',job):
            unallocated_range=(re.fullmatch(r'\d+_\[[\d,%-]+\]',job) and state.startswith('CANCELLED') and float(seconds)==0)
            if not unallocated_range:continue  # No batch/extern steps or active array ranges.
        if raw in rows:raise ValueError(f'Duplicate allocation record {raw}; inspect sacct manually')
        elapsed=float(seconds);ncpus=int(cpus or 0);gpus=gpu_count(tres)
        rows[raw]=dict(job_id=job,job_id_raw=raw,state=state,elapsed_seconds=elapsed,allocated_gpus=gpus,
            allocated_gpu_hours=elapsed*gpus/3600,allocated_cpu_hours=elapsed*ncpus/3600,start=start,end=end)
    return list(rows.values())


def job_ledger(studies,sacct_file=None):
    owners={};program=[];missing=[]
    for study in studies:
        side=study.with_name(study.name+'-submission');tsv=side/'jobs.tsv'
        if tsv.exists():
            for line in tsv.read_text().splitlines():
                if not line.strip():continue
                stage,job=line.split('\t');job=job.strip()
                if not job.isdigit():raise ValueError(f'Invalid confirmed job ID in {tsv}')
                if job in owners:raise ValueError(f'Job {job} appears in multiple ledgers; refusing double count')
                owners[job]=dict(study=str(study),stage=stage)
        else:missing.append(str(tsv))
        for path in sorted((study/'status').glob('*.json')):
            record=read(path)
            if record.get('state')=='complete' and 'seconds' in record:
                program.append(dict(study=str(study),stage=path.stem.rsplit('-',1)[0],task=path.stem,
                    python_stage_seconds=record['seconds']))
    provenance={};allocations=[]
    if owners:
        if sacct_file is not None:
            output=Path(sacct_file).read_text();provenance=dict(kind='supplied sacct export',path=str(sacct_file),sha256=digest(sacct_file))
        else:
            cmd=['sacct','-X','-n','-P','-j',','.join(sorted(owners)),
                 '--format=JobIDRaw,JobID%80,State%30,ElapsedRaw,AllocCPUS,AllocTRES%250,Start,End']
            try:
                r=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=60);output=r.stdout
                provenance=dict(kind='live Slurm allocation accounting',command=cmd)
            except (OSError,subprocess.SubprocessError) as exc:
                output='';provenance=dict(kind='unavailable',error=str(exc))
        for row in parse_sacct(output):
            parent=row['job_id'].split('_')[0]
            if parent in owners:allocations.append(dict(row,**owners[parent]))
    returned={r['job_id'].split('_')[0] for r in allocations}
    missing_jobs=sorted(set(owners)-returned)
    grouped={}
    for row in allocations:
        key=row['study']+' / '+row['stage'];g=grouped.setdefault(key,dict(allocations=0,gpu_hours=0.,cpu_hours=0.))
        g['allocations']+=1;g['gpu_hours']+=row['allocated_gpu_hours'];g['cpu_hours']+=row['allocated_cpu_hours']
    warnings=[r['job_id'] for r in allocations if r['stage'] in ('generate','fit') and r['elapsed_seconds']>0 and r['allocated_gpus']==0]
    return dict(allocations=allocations,resource_warning_job_ids=warnings,stage_totals=grouped,python_stage_timings=program,accounting=provenance,
        missing_job_ledgers=missing,missing_job_ids=missing_jobs,
        total_allocated_gpu_hours=sum(r['allocated_gpu_hours'] for r in allocations),
        total_allocated_cpu_hours=sum(r['allocated_cpu_hours'] for r in allocations),
        note='Allocation hours, not GPU utilization or dollar charges. Parallel tasks are summed; this is not elapsed turnaround time. Only jobs in supplied ledgers are covered; requeue history may be incomplete.')


def trajectory_usage(study):
    manifest=read(study/'manifest.json');questions=read(study/'questions.json')
    if digest(study/'questions.json')!=manifest['questions_sha256']:raise ValueError(f'Question identity changed: {study}')
    known={q['id']:q for q in questions};used={};counts=defaultdict(Counter);incomplete=[]
    for path in sorted((study/'shards').glob('*/trajectories/*.json')):
        wrapper=read(path);payload=wrapper['payload']
        checksum=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
        if checksum!=wrapper['sha256']:raise ValueError(f'Changed trajectory: {path}')
        qid=payload['question_id']
        if qid not in known:raise ValueError(f'Unknown trajectory {qid}')
        if qid in used:raise ValueError(f'Duplicate trajectory {qid}')
        q=known[qid];group=q['domain']+'/'+q['split'];r=Counter(questions=1)
        retained=False
        for answer in payload['responses']:
            tokens=len(answer['token_ids']);status=answer['verifier_status']
            r['answers']+=1;r['response_tokens_including_eos']+=tokens
            r['prompt_tokens_times_answers']+=len(q['prompt_token_ids']);r['capped_answers']+=int(answer['capped'])
            r['status/'+status]+=1
            if status in ('correct','incorrect') and tokens:
                retained=True;r['retained_answers']+=1;r['retained_encoder_input_tokens']+=len(q['prompt_token_ids'])+tokens;r['sampled_prefixes']+=min(tokens,manifest['config']['prefixes_per_answer'])
        r['retained_question']=int(retained)
        if len(payload['responses'])!=manifest['config']['responses_per_question']:incomplete.append(qid)
        counts[group].update(r);used[qid]=dict(group=group,family=q['family'],**r)
    absent=sorted(set(known)-set(used))
    if absent or incomplete:raise ValueError(f'Incomplete saved generation in {study}: {len(absent)} missing, {len(incomplete)} partial')
    return dict(groups={k:dict(v) for k,v in counts.items()},by_question=used,
        manifest_sha256=digest(study/'manifest.json'),note='Prompt-token count is a token-volume indicator, not measured prompt FLOPs. Response tokens include stored EOS. Sampled prefixes condition on retained verifiable answers.')


def subset_usage(records,group,budget):
    ids=sorted(q for q,r in records.items() if r['group']==group and r['retained_question'])
    if len(ids)<budget:raise ValueError(f'Only {len(ids)} retained families for {group}; requested {budget}')
    total=Counter()
    for qid in ids[:budget]:
        total.update({k:v for k,v in records[qid].items() if isinstance(v,int)})
    return dict(total)


def build(study,extras=(),sacct_file=None):
    manifest=read(study/'manifest.json');summary=read(study/'summary.json')
    if manifest['protocol']!='slice-transfer-clarify-v1':raise ValueError('Completed clarification study required')
    source=Path(manifest['source']);old=read(source/'manifest.json')
    if digest(source/'manifest.json')!=manifest['source_manifest_sha256']:raise ValueError('Source identity changed')
    for name in ('model_hashes','layers','verifier'):
        if manifest[name]!=old[name]:raise ValueError('Source/model/verifier mismatch')
    studies=[]
    for path in (source,study,*extras):
        path=Path(path).absolute()
        if str(path) not in map(str,studies):studies.append(path)
    ledger=job_ledger(studies,sacct_file)
    source_usage=trajectory_usage(source);new_usage=trajectory_usage(study)
    # Cache-reused GSM8K token volumes belong to original collection, not new generation.
    records={**source_usage['by_question'],**new_usage['by_question']}
    if len(records)!=len(source_usage['by_question'])+len(new_usage['by_question']):raise ValueError('Colliding source/new question IDs')
    for seed in manifest['seeds']:
        lock=read(study/'fits'/str(seed)/'locked.json')
        if (lock['manifest']!=digest(study/'manifest.json') or lock['packed']!=digest(study/'packed/complete.json') or lock['models']!=digest(study/'fits'/str(seed)/'models.pt')):raise ValueError('Locked fit changed')
    curves=[]
    for key,brier in summary['selected_question_brier'].items():
        domain,budget,arm=key.split('/');usage=subset_usage(records,domain+'/train',int(budget))
        curves.append(dict(target=domain,arm=arm,head_questions=int(budget),test_question_brier=brier,
            **usage,warning='Observed subset token/label volume, not an isolated timed acquisition or total label cost. Target SLICE also used its larger representation-training budget; all arms used validation labels.'))
    raw=1537;slice_=769
    resource=dict(slice_features=768,layer_features=1536,slice_head_parameters=slice_,layer_head_parameters=raw,
        slice_float64_A_bytes=8*slice_**2,layer_float64_A_bytes=8*raw**2,
        matrix_storage_ratio=slice_**2/raw**2,nominal_cubic_solve_ratio=slice_**3/raw**3,
        caveat='Algebraic head sizes only, not measured hardware speedups. SLICE adds a dense 7680→768 compressor; both require Qwen inference.')
    return dict(protocol='slice-cost-audit-v1',studies=[str(p) for p in studies],allocation_ledger=ledger,
        original_collection=source_usage['groups'],new_math_collection=new_usage['groups'],head_adaptation_curves=curves,
        representation_training_questions={str(s):read(study/'fits'/str(s)/'locked.json')['source_training_questions'] for s in manifest['seeds']},
        theoretical_head_resources=resource,
        scope='Read-only retrospective audit. Original generation/extraction is counted once. Five-seed research sweeps are actual study cost, not the cost of deploying one selected critic.',
        unmeasured=['Generation and extraction have joint timings, not separate timers.',
          'Fit jobs jointly train source/target compressors and all Ridge grids; no per-method or per-head fit timers exist.',
          'Original source fit jobs also trained pooled/wide/combined arms; no fair allocation to the selected source SLICE alone.',
          'No critic inference throughput, GPU utilization, energy, paid-label prices or dollars were measured.',
          'No causal label-saving ratio, break-even reuse count or equal-quality compute advantage is established. Source feature training is prior investment.',
          'Historical runs, failed submissions and smokes are included only if their ledgers are explicitly supplied; this is not whole-project accounting.'])


def write_report(record,out):
    out.mkdir(parents=True);(out/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    ledger=record['allocation_ledger'];lines=['SLICE transfer: retrospective cost audit',record['scope'],'',
        f'Recorded allocated GPU hours: {ledger["total_allocated_gpu_hours"]:.3f}',
        f'Recorded allocated CPU hours: {ledger["total_allocated_cpu_hours"]:.3f}',
        'GPU figures are allocations, not utilization; missing accounting is not zero cost.','', 'Stage ledger (study / phase; allocated GPUh):']
    lines.extend(f'{k}: {v["gpu_hours"]:.3f}' for k,v in ledger['stage_totals'].items())
    for name in ('original_collection','new_math_collection'):
        lines+=['',name+':']
        for group,r in record[name].items():lines.append(f'{group}: {r["questions"]} questions; {r["answers"]} answers; {r["response_tokens_including_eos"]} response tokens; {r["capped_answers"]} capped')
    lines+=['','Head-adaptation data volume (shared across arms; not total training cost):']
    for row in record['head_adaptation_curves']:
        lines.append(f'{row["target"]}/{row["head_questions"]}/{row["arm"]}: Brier={row["test_question_brier"]:.6f}; subset answers={row["answers"]}; response tokens={row["response_tokens_including_eos"]}')
    if ledger['missing_job_ids'] or ledger['missing_job_ledgers'] or ledger['resource_warning_job_ids'] or ledger['accounting'].get('kind')=='unavailable':
        lines+=['','INCOMPLETE ALLOCATION ACCOUNTING: '+json.dumps({k:ledger[k] for k in ('missing_job_ids','missing_job_ledgers','resource_warning_job_ids','accounting')})]
    lines+=['','Unmeasured:']+['- '+v for v in record['unmeasured']]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')


def plots(record,out):
    import os,tempfile
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'slice-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for domain,ax in zip(('gsm8k','math'),axes):
        for arm in ('rms_two_thirds','transferred_slice','target_slice'):
            rows=sorted([r for r in record['head_adaptation_curves'] if r['target']==domain and r['arm']==arm],key=lambda r:r['head_questions'])
            ax.plot([r['response_tokens_including_eos'] for r in rows],[r['test_question_brier'] for r in rows],'o-',label=arm)
        ax.set_xscale('log');ax.set_xlabel('Response tokens in selected head-training questions');ax.set_title(domain if domain=='gsm8k' else 'MATH-500 numeric subset');ax.grid(alpha=.2)
    axes[0].set_ylabel('Test question-mean Brier (lower is better)');axes[1].legend(fontsize=8)
    fig.suptitle('Prediction versus target head-data volume; prior representation cost excluded')
    fig.text(.5,-.03,'Same collected pool reused; target-trained SLICE used additional representation labels. Points are descriptive means.',ha='center',fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'quality-vs-head-tokens.{ext}',bbox_inches='tight')
    plt.close(fig)
    stages=record['allocation_ledger']['stage_totals']
    if stages:
        labels=[Path(k.split(' / ')[0]).name+' / '+k.split(' / ')[1] for k in stages]
        values=[v['gpu_hours'] for v in stages.values()]
        fig,ax=plt.subplots(figsize=(11,max(4,len(labels)*.35)));ax.barh(labels,values);ax.set_xlabel('Allocated GPU hours (summed over jobs)');ax.invert_yaxis();ax.set_title('Recorded research-run allocations; shared work counted once');fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/f'allocation-ledger.{ext}',bbox_inches='tight')
        plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--plot-report',type=Path)
    p.add_argument('--extra-study',type=Path,action='append',default=[]);p.add_argument('--sacct-file',type=Path);p.add_argument('--plots',action='store_true')
    a=p.parse_args()
    if a.out.exists():p.error('Use a new audit output directory; source results are preserved')
    if a.plot_report:
        a.out.mkdir(parents=True);plots(read(a.plot_report),a.out);print('Saved figures:',a.out);return
    if a.study is None:p.error('--study required unless --plot-report is supplied')
    record=build(a.study.absolute(),a.extra_study,a.sacct_file);write_report(record,a.out)
    if a.plots:plots(record,a.out)
    print((a.out/'report.txt').read_text());print('Saved:',a.out/'summary.json')
if __name__=='__main__':main()
