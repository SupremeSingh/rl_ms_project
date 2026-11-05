"""Fail-closed Slurm chain; never retry an ambiguous sbatch timeout automatically."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]

# One known smoke-compatible migration: preserve model-path spelling at submission.
# Every other pinned source hash must still match. This does not waive GPU smoke.
PATH_FIX_PREDECESSOR = "cb66a2fa0a4cf5d9a7723d4cf3d7fb0dfddf15148e4322d7c0ec74d93d2c8d13"


def model_argument(value):
    # A path resolved on a login host can be invalid in a container's mount namespace.
    # Make it absolute without resolving symlinks (/usr/xtmp vs /usr/project/xtmp).
    return str(Path(value).expanduser().absolute())


def smoke_changes(manifest):
    import hashlib
    changes={}
    for name,digest in manifest['code'].items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual==digest:continue
        if name=='tools/transfer/submit.py' and digest==PATH_FIX_PREDECESSOR:
            changes[name]=dict(smoke_sha256=digest,current_sha256=actual,
                               reason='Model-path spelling preserved across host/container namespaces')
        else:raise ValueError('Implementation changed after smoke; rerun smoke')
    return changes


def submit(out,mode,shards=4,concurrency=4,model=None):
    out=out.resolve()
    if out.exists():raise ValueError('Use a new output directory')
    if not 1<=concurrency<=shards<=32:raise ValueError('Require 1 <= concurrency <= shards <= 32')
    # Full studies require a successful real-model smoke from this implementation.
    if mode!='smoke':raise ValueError('Use submit_full() for a full study')
    return chain(out,mode,shards,concurrency,model)


def chain(out,mode,shards,concurrency,model,smoke_reuse=None):
    (out.parent).mkdir(parents=True,exist_ok=True)
    # Sidecar holds submission bookkeeping so prepare can enforce a fresh output.
    side=out.with_name(out.name+'-submission');side.mkdir()
    if smoke_reuse is not None:
        (side/'smoke-reuse.json').write_text(json.dumps(smoke_reuse,indent=2)+'\n')
    logdir=side/'logs';logdir.mkdir();previous=None
    for stage in ('prepare','generate','pack','fit','evaluate'):
        gpu=stage in ('generate','fit')
        cmd=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=128G',
             '--time='+('24:00:00' if stage=='generate' else '04:00:00'),
             '--partition='+('compsci-gpu' if gpu else 'compsci'),
             '--job-name=slice-transfer-'+stage,'--output='+str(logdir/f'{stage}-%A_%a.out')]
        if gpu:cmd+=['--gres=gpu:a5000:1','--exclude=compsci-cluster-fitz-15']
        if previous:cmd+=['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes']
        if stage=='generate':cmd+=['--array=0-'+str(shards-1)+'%'+str(concurrency)]
        if stage=='fit':cmd+=['--array=0-2%3']
        cmd += [str(ROOT/'tools/transfer/run.sh'),stage,str(out)]
        if stage=='prepare':
            cmd+=['--shards',str(shards)]
            if mode=='smoke':cmd+=['--train','40','--val','20','--test','20','--epochs','2','--tokens','256']
            if model:cmd+=['--model',model_argument(model)]
        (side/f'{stage}.json').write_text(json.dumps(dict(state='submitting',command=cmd),indent=2))
        result=subprocess.run(cmd,text=True,capture_output=True,timeout=90)
        job=result.stdout.strip().split(';')[0]
        if result.returncode or not job.isdigit():
            raise RuntimeError(f'Submission stopped at {stage}; inspect squeue/sacct before any retry. {result.stderr}')
        (side/f'{stage}.json').write_text(json.dumps(dict(state='submitted',job=job,command=cmd),indent=2))
        with (side/'jobs.tsv').open('a') as f:f.write(f'{stage}\t{job}\n')
        print('Submitted',stage+':',job,flush=True);previous=job
    print('Jobs:',side/'jobs.tsv');print('Final report:',out/'report.txt')


def submit_full(out,smoke,shards,concurrency,model):
    import hashlib
    out=out.resolve();smoke=smoke.resolve()
    if out.exists():raise ValueError('Use a new output directory')
    if not 1<=concurrency<=shards<=32:raise ValueError('Invalid concurrency')
    m=json.loads((smoke/'manifest.json').read_text());status=json.loads((smoke/'status/evaluate-0.json').read_text())
    if m['protocol']!='slice-transfer-v1' or status['state']!='complete' or not (smoke/'report.txt').exists():raise ValueError('Smoke is not complete')
    changes=smoke_changes(m)
    if model and model_argument(model)!=model_argument(m['model_path']):raise ValueError('Smoke used a different model path')
    if changes:print('Reusing passed smoke with recognized submission-path fix only.',flush=True)
    return chain(out,'full',shards,concurrency,m['model_path'],smoke_reuse=dict(
        smoke=str(smoke),manifest_sha256=hashlib.sha256((smoke/'manifest.json').read_bytes()).hexdigest(),
        recognized_changes=changes,model_argument=model_argument(m['model_path'])))



def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['smoke','full'],default='smoke')
    p.add_argument('--smoke',type=Path);p.add_argument('--model');p.add_argument('--shards',type=int,default=4);p.add_argument('--concurrency',type=int,default=4)
    a=p.parse_args()
    if a.mode=='full':
        if not a.smoke:p.error('--smoke SUCCESSFUL_SMOKE_DIRECTORY required for full study')
        submit_full(a.out,a.smoke,a.shards,a.concurrency,a.model)
    else:submit(a.out,'smoke',a.shards,a.concurrency,a.model)
if __name__=='__main__':main()
