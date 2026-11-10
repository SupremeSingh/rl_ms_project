"""Staged multi-seed repeat; original actor-training algorithms, GPU partition only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_smoke(out):
    m=json.loads((out/'manifest.json').read_text());g=json.loads((out/'smoke.json').read_text())
    path=out/'seed1101/smoke/verification.json'
    if (m['protocol']!='value-repeat-v1' or not g.get('passed')
            or g['manifest_sha256']!=digest(out/'manifest.json') or g['child_verification_sha256']!=digest(path)):
        raise ValueError('Matching repeat smoke required')
    child=json.loads(path.read_text())
    if child['manifest_sha256']!=digest(path.parent.parent/'manifest.json'):raise ValueError('Child changed')
    for name,h in child['files'].items():
        if Path(name).name!=name or digest(path.parent/name)!=h:raise ValueError('Smoke artifact changed')
    for name,h in m['runtime'].items():
        if digest(ROOT/name)!=h:raise ValueError('Runtime changed since smoke')


def submit(out,mode,source=None,concurrency=6):
    out=Path(out).resolve()
    if mode not in ('smoke','main') or not 1<=concurrency<=12:raise ValueError('Invalid mode/concurrency')
    if mode=='smoke':
        if out.exists() or source is None or not (Path(source)/'manifest.json').exists():
            raise ValueError('Smoke requires source study and a new output directory')
    else:
        if source is not None:raise ValueError('Source is sealed during smoke preparation')
        check_smoke(out)
    ledger=Path(str(out)+'-submission');ledger.mkdir(parents=True,exist_ok=True)
    (ledger/f'.submission-{mode}').mkdir(exist_ok=False)
    logs=ledger/'logs';logs.mkdir(exist_ok=True)
    def job(stage,dependency='',tasks=None,gpus=1,hours=2):
        name=f'{mode}-{stage}'
        args=['sbatch','--parsable','--partition=compsci-gpu',f'--gres=gpu:a5000:{gpus}',
            '--nodes=1','--ntasks=1','--cpus-per-task=8',f'--mem={128 if gpus==2 else 64}G',
            '--exclude=compsci-cluster-fitz-15',f'--time={hours:02d}:00:00',
            f'--job-name=repeat-{name}',f'--output={logs}/{name}-%A_%a.out']
        if dependency:args += [f'--dependency=afterok:{dependency}','--kill-on-invalid-dep=yes']
        if tasks:args += [f'--array=0-{tasks-1}%{min(tasks,concurrency)}']
        args += ['scripts/run_value_repeat.sh',stage,str(out),mode,str(gpus),str(Path(source).resolve()) if source else '']
        sidecar=ledger/f'{name}-submission.json'
        sidecar.write_text(json.dumps(dict(state='submitting',command=args),indent=2)+'\n')
        try:r=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=90)
        except (OSError,subprocess.TimeoutExpired) as exc:
            sidecar.write_text(json.dumps(dict(state='ambiguous_or_failed',error=str(exc)),indent=2)+'\n')
            raise RuntimeError('Check Slurm before retrying; submission outcome unknown') from exc
        if r.returncode or not re.fullmatch(r'\d+(;[\w.-]+)?',r.stdout.strip()):
            sidecar.write_text(json.dumps(dict(state='ambiguous_or_failed',stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
            raise RuntimeError(f'{name}: {r.stderr}. Check Slurm before retrying.')
        value=r.stdout.strip().split(';')[0]
        sidecar.write_text(json.dumps(dict(state='submitted',job_id=value,command=args),indent=2)+'\n')
        with (ledger/'jobs.tsv').open('a') as f:f.write(f'{name}\t{value}\n')
        print(f'Submitted {name}: {value}',flush=True);return value
    count=1 if mode=='smoke' else 3;dep=''
    if mode=='smoke':dep=job('prepare')
    dep=job('train',dep,tasks=6*count,gpus=2,hours=4 if mode=='smoke' else 12)
    dep=job('training-gate',dep,tasks=count)
    dep=job('bank',dep,tasks=6*count)
    dep=job('pack',dep,tasks=count)
    dep=job('predict',dep,tasks=6*count,hours=4)
    dep=job('mc',dep,tasks=6*count*(1 if mode=='smoke' else 4),hours=12)
    dep=job('report',dep,tasks=count)
    job('aggregate',dep)
    result=out/'seed1101/smoke' if mode=='smoke' else out/'main'
    print(f'Jobs: {ledger}/jobs.tsv\nReport: {result}/report.txt\nPlot: {result}/value-scatter.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','main'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path);p.add_argument('--concurrency',type=int,default=6)
    a=p.parse_args();submit(a.out,a.mode,a.source,a.concurrency)
