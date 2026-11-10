"""GPU-partition-only controller for a frozen-policy critic comparison (no actor training)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smoke_gate(smoke,source):
    smoke=Path(smoke);m=json.loads((smoke/'manifest.json').read_text())
    gate=json.loads((smoke/'verification.json').read_text())
    if (m['protocol']!='fixed-policy-critics-v1' or m['phase']!='smoke' or not gate.get('passed')
            or gate['manifest_sha256']!=digest(smoke/'manifest.json')
            or m['source_manifest']!=digest(Path(source)/'manifest.json')):
        raise ValueError('Matching fixed-policy smoke required')
    for name,h in gate['files'].items():
        if Path(name).name!=name or digest(smoke/name)!=h:raise ValueError('Smoke artifact changed')
    for name,h in m['runtime'].items():
        if digest(ROOT/name)!=h:raise ValueError('Code/environment changed since smoke')


def submit(source,out,mode,smoke=None,concurrency=4):
    source=Path(source).resolve();out=Path(out).resolve()
    if mode not in ('smoke','main') or not 1<=concurrency<=4:raise ValueError('Invalid mode/concurrency')
    if out.exists() or not (source/'pilot/verification.json').exists():
        raise ValueError('Use a new output directory and a completed original value audit as source')
    if mode=='main':
        if smoke is None:raise ValueError('Main requires --smoke')
        smoke_gate(smoke,source)
    ledger=Path(str(out)+'-submission');ledger.mkdir(parents=True,exist_ok=True)
    (ledger/'.submission').mkdir(exist_ok=False)
    logs=ledger/'logs';logs.mkdir(exist_ok=True)
    def job(stage,dependency='',array=None,gpus=1,hours=2):
        args=['sbatch','--parsable','--partition=compsci-gpu',f'--gres=gpu:a5000:{gpus}',
              '--nodes=1','--ntasks=1',f'--cpus-per-task={8 if gpus==2 else 4}',
              f'--mem={128 if gpus==2 else 64}G',f'--time={hours:02d}:00:00',
              '--exclude=compsci-cluster-fitz-15',f'--job-name=fixed-value-{stage}',
              f'--output={logs}/{stage}-%A_%a.out']
        if dependency:args += [f'--dependency=afterok:{dependency}','--kill-on-invalid-dep=yes']
        if array:args += [f'--array={array}']
        args += ['scripts/run_fixed_policy.sh',stage,str(out),mode,str(source)]
        sidecar=ledger/f'{stage}-submission.json'
        sidecar.write_text(json.dumps(dict(state='submitting',command=args),indent=2)+'\n')
        try:r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=90)
        except (OSError,subprocess.TimeoutExpired) as exc:
            sidecar.write_text(json.dumps(dict(state='ambiguous_or_failed',error=str(exc)),indent=2)+'\n')
            raise RuntimeError('Check Slurm before retrying: submission outcome unknown') from exc
        if r.returncode or not re.fullmatch(r'\d+(;[\w.-]+)?',r.stdout.strip()):
            sidecar.write_text(json.dumps(dict(state='ambiguous_or_failed',stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
            raise RuntimeError(f'{stage}: {r.stderr}. Check Slurm before retrying; no automatic resubmission.')
        value=r.stdout.strip().split(';')[0]
        sidecar.write_text(json.dumps(dict(state='submitted',job_id=value,command=args),indent=2)+'\n')
        with (ledger/'jobs.tsv').open('a') as f:f.write(f'{stage}\t{value}\n')
        print(f'Submitted {stage}: {value}',flush=True);return value
    dep=job('prepare')
    dep=job('generate',dep,f'0-3%{concurrency}',hours=6)
    dep=job('extract',dep,f'0-3%{concurrency}',hours=6)
    dep=job('linear',dep,hours=4)
    dep=job('bank',dep)
    dep=job('neural',dep,gpus=2,hours=12)
    dep=job('predict',dep,f'0-5%{concurrency}',hours=4)
    dep=job('mc',dep,f'0-3%{concurrency}',hours=12)
    job('report',dep)
    print(f'Jobs: {ledger}/jobs.tsv\nReport: {out}/report.txt\nPlot: {out}/value-scatter.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','main'],required=True)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',type=Path);p.add_argument('--concurrency',type=int,default=4)
    a=p.parse_args();submit(a.source,a.out,a.mode,a.smoke,a.concurrency)
