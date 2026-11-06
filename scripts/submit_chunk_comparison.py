"""Stdlib Slurm controller; fail closed after timeouts and record every job ID."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def submit(out,mode,source=None,concurrency=6):
    out=out.absolute();ledger=Path(str(out)+'-submission');ledger.mkdir(parents=True,exist_ok=True)
    if mode=='smoke' and (source is None or out.exists()):raise ValueError('Smoke requires --source and a new output directory')
    if mode=='train':
        gate=json.loads((out/'smoke.json').read_text())
        import hashlib
        if not gate.get('passed') or gate['manifest_sha256']!=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Matching passed smoke is required')
    if not 1<=concurrency<=20:raise ValueError('Concurrency must be between 1 and 20')
    marker=ledger/('.submission-'+mode)
    marker.mkdir(exist_ok=False) # Ambiguous submissions must be inspected before retrying.
    logs=ledger/'logs';logs.mkdir(exist_ok=True)
    def job(stage,dependency='',array=None,gpu=False):
        args=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=128G',
            '--exclude=compsci-cluster-fitz-15',f'--time={"12:00:00" if stage=="run" else "04:00:00" if gpu else "01:00:00"}',
            f'--job-name=chunk-compare-{stage}',f'--output={logs}/{stage}-%A_%a.out']
        args+=['--partition=compsci-gpu','--gres=gpu:a5000:2'] if gpu else ['--partition=compsci']
        if dependency:args+=[f'--dependency=afterok:{dependency}','--kill-on-invalid-dep=yes']
        if array:args+=[f'--array={array}']
        args+=['scripts/run_chunk_comparison.sh',stage,str(out),'1' if gpu else '0',str(source.absolute()) if source else '']
        # Record intent first: a timed-out sbatch may already have accepted the job.
        sidecar=ledger/f'{stage}-submission.json';sidecar.write_text(json.dumps(dict(state='submitting',command=args),indent=2)+'\n')
        r=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
        if r.returncode or not re.fullmatch(r'\d+(;[\w.-]+)?',r.stdout.strip()):
            sidecar.write_text(json.dumps(dict(state='ambiguous_or_failed',stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
            raise RuntimeError(f'{stage} submission failed: {r.stderr}. Check Slurm before retrying; no automatic resubmission.')
        value=r.stdout.strip().split(';')[0]
        sidecar.write_text(json.dumps(dict(state='submitted',job_id=value,command=args),indent=2)+'\n')
        with (ledger/'jobs.tsv').open('a') as f:f.write(f'{stage}\t{value}\n')
        print(f'Submitted {stage}: {value}',flush=True);return value
    if mode=='smoke':
        prep=job('prepare');smoke=job('smoke',prep,f'0-3%{min(4,concurrency)}',True);job('gate',smoke)
    else:
        train=job('run',array=f'0-19%{concurrency}',gpu=True);job('report',train)
    print(f'Jobs: {ledger}/jobs.tsv\nReport: {out}/report.txt')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',required=True,choices=['smoke','train'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--concurrency',type=int,default=6);a=p.parse_args()
    submit(a.out,a.mode,a.source,a.concurrency)
