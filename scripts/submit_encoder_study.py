"""Stdlib scheduler for a separate Numina encoder study; never touches active PPO outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def submit(source,out,mode='smoke',smoke=None,concurrency=3,cpu_on_gpu=False,all_prefixes=False):
    source=source.resolve();out=out.resolve()
    if out.exists() or not (source/'manifest.json').is_file():raise ValueError('Require an existing source and new output directory')
    if not 1<=concurrency<=3:raise ValueError('Concurrency must be 1–3')
    if mode=='full':
        if smoke is None:raise ValueError('Full mode requires --smoke')
        manifest=json.loads((smoke/'manifest.json').read_text());gate=json.loads((smoke/'verification.json').read_text())
        if (not gate['passed'] or manifest['mode']!='smoke'
            or gate['manifest_sha256']!=hashlib.sha256((smoke/'manifest.json').read_bytes()).hexdigest()
            or gate['summary_sha256']!=hashlib.sha256((smoke/'summary.json').read_bytes()).hexdigest()):raise ValueError('Passed unchanged smoke required')
        for name,digest in manifest['runtime'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Code changed since smoke')
    ledger=Path(str(out)+'-submission');ledger.mkdir(parents=True,exist_ok=True)
    (ledger/'.submission').mkdir(exist_ok=False);logs=ledger/'logs';logs.mkdir(exist_ok=True)
    def job(stage,dep='',array=None,gpu=True):
        gpu=gpu or cpu_on_gpu
        args=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=48G',
              '--time=24:00:00' if gpu else '--time=01:00:00',f'--job-name=enc-{stage}',
              f'--output={logs}/{stage}-%A_%a.out','--exclude=compsci-cluster-fitz-15']
        args+=['--partition=compsci-gpu','--gres=gpu:a5000:1'] if gpu else ['--partition=compsci']
        if dep:args += [f'--dependency=afterok:{dep}','--kill-on-invalid-dep=yes']
        if array:args += [f'--array={array}']
        args += ['scripts/run_encoder_study.sh',stage,str(out),mode,str(source),'1' if gpu else '0','1' if all_prefixes else '0']
        sidecar=ledger/f'{stage}.json';sidecar.write_text(json.dumps(dict(state='submitting',command=args)))
        try:r=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,timeout=90)
        except (OSError,subprocess.TimeoutExpired) as exc:
            sidecar.write_text(json.dumps(dict(state='ambiguous_or_failed',error=str(exc))))
            raise RuntimeError('Check Slurm before retrying; submission outcome unknown') from exc
        if r.returncode or not re.fullmatch(r'\d+(;[\w.-]+)?',r.stdout.strip()):
            sidecar.write_text(json.dumps(dict(state='ambiguous_or_failed',stdout=r.stdout,stderr=r.stderr)))
            raise RuntimeError(f'Check Slurm before retrying: {r.stderr}')
        jid=r.stdout.strip().split(';')[0];sidecar.write_text(json.dumps(dict(state='submitted',job_id=jid,command=args)))
        with (ledger/'jobs.tsv').open('a') as f:f.write(f'{stage}\t{jid}\n')
        print(f'Submitted {stage}: {jid}',flush=True);return jid
    dep=job('prepare',gpu=False)
    for stage in ('generate','mntp','contrastive'):dep=job(stage,dep)
    dep=job('extract',dep,f'0-2%{concurrency}')
    dep=job('fit',dep,f'0-2%{concurrency}')
    job('evaluate',dep,gpu=False)
    print(f'Jobs: {ledger}/jobs.tsv\nReport: {out}/report.txt')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['smoke','full'],default='smoke');p.add_argument('--smoke',type=Path)
    p.add_argument('--concurrency',type=int,default=3)
    p.add_argument('--cpu-on-gpu',action='store_true',help='Run short preparation/report jobs on a GPU node too')
    p.add_argument('--all-prefixes',action='store_true',help='Every token/chunk prefix instead of bounded sampling; very expensive')
    a=p.parse_args();submit(a.source,a.out,a.mode,a.smoke,a.concurrency,a.cpu_on_gpu,a.all_prefixes)
