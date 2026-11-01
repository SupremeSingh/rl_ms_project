"""Submit parallel smoke -> CPU gate -> training array -> report, fail closed.

Stdlib only; run on a login node. No shell interpolation or uncertain-ID retries.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def submit_chain(out,concurrency=6,smoke_concurrency=7,mode='all',submit=subprocess.run):
    out=Path(out).resolve()
    manifest=json.loads((out/'manifest.json').read_text())
    if (manifest.get('protocol')!='slice-seven-critics-gpu-v1'
            or len(manifest['conditions'])!=7 or len(manifest['seeds'])!=3):
        raise ValueError('Prepare the GPU seven-arm study first')
    if not 1<=concurrency<=21 or not 1<=smoke_concurrency<=7: raise ValueError('Invalid concurrency')
    (ROOT/'logs').mkdir(exist_ok=True)
    marker=out/'.submission-chain'
    marker.mkdir(exist_ok=True)
    script=ROOT/'tools/diagnostics/submit_online_critics.sh'
    jobs={}
    if (out/'jobs.tsv').exists():
        jobs=dict(line.split('\t') for line in (out/'jobs.tsv').read_text().splitlines())
    stages=[('smoke','smoke-task',[f'--array=0-6%{smoke_concurrency}'],None),
            ('gate','smoke-gate',['--partition=compsci','--gres=none','--mem=32G','--time=01:00:00'],'smoke'),
            ('run','run',[f'--array=0-20%{concurrency}'],'gate'),
            ('report','report',['--partition=compsci','--gres=none','--mem=32G','--time=01:00:00'],'run')]
    if mode=='smoke':stages=stages[:2]
    elif mode=='train':
        gate=json.loads((out/'smoke.json').read_text())
        digest=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
        if not gate.get('passed') or gate['manifest_sha256']!=digest or 'gate' not in jobs:
            raise ValueError('Successful matching smoke gate required')
        stages=stages[2:]
    elif mode!='all':raise ValueError('Unknown submission mode')
    for label,stage,options,parent in stages:
        stage_marker=marker/label
        stage_marker.mkdir(exist_ok=False)  # An uncertain submit must never be repeated blindly.
        command=['sbatch','--parsable',f'--chdir={ROOT}',f'--job-name=slice-critics-{label}',*options]
        if parent:command += [f'--dependency=afterok:{jobs[parent]}','--kill-on-invalid-dep=yes']
        command += [str(script),stage,str(out)]
        (stage_marker/'submission.json').write_text(json.dumps(dict(state='submitting',command=command),indent=2))
        try:
            result=submit(command,cwd=ROOT,text=True,capture_output=True,check=True,timeout=60)
            job=result.stdout.strip().split(';')[0]
            if not re.fullmatch(r'[0-9]+',job): raise ValueError('Unconfirmed Slurm job ID')
        except (subprocess.SubprocessError,ValueError) as exc:
            message=getattr(exc,'stderr',None) or str(exc)
            raise RuntimeError(f'Stopped at {label}: {message}. Submission may have reached Slurm. '
                               'Inspect squeue/sacct and jobs.tsv before retrying; marker retained.') from exc
        jobs[label]=job
        (stage_marker/'submission.json').write_text(json.dumps(dict(state='confirmed',job=job),indent=2))
        with (out/'jobs.tsv').open('a') as stream:stream.write(f'{label}\t{job}\n')
        print(f'Submitted {label}: {job}',flush=True)
    return jobs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=6)
    p.add_argument('--smoke-concurrency',type=int,default=7)
    p.add_argument('--mode',choices=['all','smoke','train'],default='all')
    a=p.parse_args()
    submit_chain(a.out,a.concurrency,a.smoke_concurrency,a.mode)
