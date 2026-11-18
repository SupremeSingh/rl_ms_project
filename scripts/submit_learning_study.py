"""Submit 15 independent learning-curve actors with a maximum concurrency of six."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import hashlib
ROOT=Path(__file__).resolve().parents[1]

def submit(out,mode,source=None,hours=12,concurrency=6):
    out=Path(out).resolve()
    if mode not in ['smoke','main'] or not 1<=hours<=168 or not 1<=concurrency<=6:raise ValueError('Invalid mode/time/concurrency')
    if mode=='smoke':
        if source is None or out.exists():raise ValueError('New output and source required')
    else:
        m=out/'manifest.json';g=json.loads((out/'smoke.json').read_text())
        if not g['passed'] or g['manifest_sha256']!=hashlib.sha256(m.read_bytes()).hexdigest():raise ValueError('Passed smoke required')
    ledger=Path(str(out)+'-submission');ledger.mkdir(parents=True,exist_ok=True)
    (ledger/f'.submission-{mode}').mkdir(exist_ok=False);logs=ledger/'logs';logs.mkdir(exist_ok=True)
    def job(stage,dep=None,tasks=None,gpus=1,wall=2):
        label=mode+'-'+stage
        cmd=['sbatch','--parsable','--partition=compsci-gpu','--nodes=1','--ntasks=1',f'--gres=gpu:a5000:{gpus}',
             '--cpus-per-task=8',f'--mem={128 if gpus==2 else 64}G',f'--time={wall:02d}:00:00',
             '--exclude=compsci-cluster-fitz-15,compsci-cluster-fitz-21',f'--job-name=curves-{label}',f'--output={logs}/{label}-%A_%a.out']
        if dep:cmd += [f'--dependency=afterok:{dep}','--kill-on-invalid-dep=yes']
        if tasks:cmd += [f'--array=0-{tasks-1}%{min(tasks,concurrency)}']
        cmd += ['scripts/run_learning_study.sh',stage,str(out),str(Path(source).resolve()) if source else '']
        sidecar=ledger/f'{label}.json';sidecar.write_text(json.dumps(dict(state='submitting',command=cmd),indent=2))
        try:r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=90)
        except (OSError,subprocess.TimeoutExpired) as e:
            sidecar.write_text(json.dumps(dict(state='ambiguous',error=str(e))))
            raise RuntimeError('Check Slurm before any retry; submission uncertain') from e
        if r.returncode or not re.fullmatch(r'\d+(;[\w.-]+)?',r.stdout.strip()):
            sidecar.write_text(json.dumps(dict(state='failed_or_ambiguous',stdout=r.stdout,stderr=r.stderr)))
            raise RuntimeError('Check Slurm before retrying: '+r.stderr)
        jid=r.stdout.strip().split(';')[0];sidecar.write_text(json.dumps(dict(state='submitted',job_id=jid,command=cmd)))
        with (ledger/'jobs.tsv').open('a') as f:f.write(f'{label}\t{jid}\n')
        print(f'Submitted {label}: {jid}',flush=True);return jid
    if mode=='smoke':
        a=job('prepare');b=job('smoke',a,tasks=5,gpus=2,wall=4);job('smoke-gate',b)
    else:
        a=job('train',tasks=15,gpus=2,wall=hours);b=job('train-gate',a)
        c=job('test',b,tasks=16,wall=4);job('report',c)
    print(f'Jobs: {ledger}/jobs.tsv')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['smoke','main'],required=True)
    p.add_argument('--source',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--hours-per-actor',type=int,default=12);p.add_argument('--concurrency',type=int,default=6)
    a=p.parse_args();submit(a.out,a.mode,a.source,a.hours_per_actor,a.concurrency)
