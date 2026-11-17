"""Explicit smoke/full submissions; test and controlled-cost results stay separate."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import hashlib

ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_gate(out,name):
    g=json.loads((out/name).read_text())
    if not g['passed'] or g['manifest_sha256']!=digest(out/'manifest.json'):raise ValueError('Matching passed gate required')

def submit(out,mode,source=None,smoke=None,concurrency=6):
    out=Path(out).resolve()
    if not 1<=concurrency<=12:raise ValueError('Invalid concurrency')
    if mode=='main':check_gate(out,'smoke.json')
    else:
        if out.exists() or source is None:raise ValueError('Require source and a new output directory')
        if mode=='benchmark':
            if smoke is None:raise ValueError('Benchmark smoke required')
            check_gate(Path(smoke),'verification.json')
    ledger=Path(str(out)+'-submission');ledger.mkdir(parents=True,exist_ok=True)
    (ledger/f'.submission-{mode}').mkdir(exist_ok=False);logs=ledger/'logs';logs.mkdir(exist_ok=True)
    def job(module,stage,dep=None,count=None,gpus=1,hours=2):
        label=f'{mode}-{stage}'
        args=['sbatch','--parsable','--partition=compsci-gpu','--nodes=1','--ntasks=1',
            f'--gres=gpu:a5000:{gpus}','--cpus-per-task=8',f'--mem={128 if gpus==2 else 64}G',
            '--exclude=compsci-cluster-fitz-15,compsci-cluster-fitz-21',f'--time={hours:02d}:00:00',
            f'--job-name=followup-{label}',f'--output={logs}/{label}-%A_%a.out']
        if dep:args += [f'--dependency=afterok:{dep}','--kill-on-invalid-dep=yes']
        if count:args += [f'--array=0-{count-1}%{min(count,concurrency)}']
        args += ['scripts/run_test_followup.sh',module,stage,str(out),str(Path(source).resolve()) if source else '',str(Path(smoke).resolve()) if smoke else '']
        sidecar=ledger/f'{label}.json';sidecar.write_text(json.dumps(dict(state='submitting',command=args),indent=2))
        try:r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=90)
        except (OSError,subprocess.TimeoutExpired) as e:
            sidecar.write_text(json.dumps(dict(state='ambiguous',error=str(e))))
            raise RuntimeError('Check Slurm before retrying; submission outcome uncertain') from e
        if r.returncode or not re.fullmatch(r'\d+(;[\w.-]+)?',r.stdout.strip()):
            sidecar.write_text(json.dumps(dict(state='failed_or_ambiguous',stdout=r.stdout,stderr=r.stderr)))
            raise RuntimeError('Check Slurm before retrying: '+r.stderr)
        job_id=r.stdout.strip().split(';')[0];sidecar.write_text(json.dumps(dict(state='submitted',job_id=job_id,command=args),indent=2))
        with (ledger/'jobs.tsv').open('a') as f:f.write(f'{label}\t{job_id}\n')
        print(f'Submitted {label}: {job_id}',flush=True);return job_id
    if mode=='smoke':
        a=job('test_followup','prepare');b=job('test_followup','smoke',a,count=19)
        job('test_followup','gate',b)
    elif mode=='main':
        a=job('test_followup','test',count=19,hours=4);job('test_followup','report',a)
    else:
        a=job('cost_benchmark','prepare');job('cost_benchmark','run',a,gpus=2,hours=2 if mode=='benchmark-smoke' else 12)
    print(f'Jobs: {ledger}/jobs.tsv')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['smoke','main','benchmark-smoke','benchmark'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--smoke',type=Path);p.add_argument('--concurrency',type=int,default=6)
    a=p.parse_args();submit(a.out,a.mode,a.source,a.smoke,a.concurrency)
