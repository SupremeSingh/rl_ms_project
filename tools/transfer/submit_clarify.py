"""Separate study; fail-closed submission; test evaluation is an explicit later action."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]


def read(p):return json.loads(p.read_text())


def validate_files(manifest):
    for name,h in manifest['code'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('Implementation changed; use a new smoke study')


def submit(out,source,mode,smoke=None,shards=4,concurrency=4):
    out=out.absolute()
    if not 1<=concurrency<=shards<=32:raise ValueError('Require 1 <= concurrency <= shards <=32')
    if mode=='evaluate':
        m=read(out/'manifest.json');validate_files(m)
        if read(out/'status/validate-0.json')['state']!='complete':raise ValueError('Validation must complete first')
        if (out/'summary.json').exists():raise ValueError('Already evaluated')
        side=out.with_name(out.name+'-submission');marker=side/'evaluate.json'
        if marker.exists():raise ValueError('Evaluation already submitted; inspect scheduler before retry')
        stages=['evaluate']
    else:
        if source is None:raise ValueError('--source required')
        source=source.absolute()
        if out.exists():raise ValueError('Use a new output directory')
        if mode=='full':
            if smoke is None:raise ValueError('Successful --smoke required')
            m=read(smoke/'manifest.json');validate_files(m)
            if not m['smoke'] or read(smoke/'status/evaluate-0.json')['state']!='complete':raise ValueError('Complete real-model smoke required')
            if str(source)!=m['source']:raise ValueError('Smoke used different source path')
        side=out.with_name(out.name+'-submission');side.mkdir(parents=True)
        stages=['prepare','generate','pack','fit','validate']+(['evaluate'] if mode=='smoke' else [])
    logs=side/'logs';logs.mkdir(exist_ok=True);previous=None
    for stage in stages:
        gpu=stage in ('generate','fit')
        cmd=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=128G',
             '--time='+('24:00:00' if stage=='generate' else '06:00:00'),
             '--partition='+('compsci-gpu' if gpu else 'compsci'),'--job-name=slice-clarify-'+stage,
             '--output='+str(logs/f'{stage}-%A_%a.out')]
        if gpu:cmd+=['--gres=gpu:a5000:1','--exclude=compsci-cluster-fitz-15']
        if previous:cmd+=['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes']
        if stage=='generate':cmd+=['--array=0-'+str(shards-1)+'%'+str(concurrency)]
        if stage=='fit':cmd+=['--array=0-4%'+str(min(5,concurrency))]
        cmd += [str(ROOT/'tools/transfer/run_clarify.sh'),stage,str(out)]
        if stage=='prepare':
            cmd+=['--source',str(source),'--shards',str(shards)]
            if mode=='smoke':cmd+=['--smoke']
        marker=side/f'{stage}.json'
        if marker.exists():raise ValueError('Stage already submitted')
        marker.write_text(json.dumps(dict(state='submitting',command=cmd),indent=2))
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=90);job=r.stdout.strip().split(';')[0]
        if r.returncode or not job.isdigit():raise RuntimeError(f'Submission stopped: {r.stderr}. Check scheduler before retry.')
        marker.write_text(json.dumps(dict(state='submitted',job=job,command=cmd),indent=2))
        with (side/'jobs.tsv').open('a') as f:f.write(f'{stage}\t{job}\n')
        print('Submitted',stage+':',job,flush=True);previous=job
    print('Jobs:',side/'jobs.tsv')
    print('Report:',out/('report.txt' if mode in ('smoke','evaluate') else 'validation-report.txt'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--smoke',type=Path)
    p.add_argument('--mode',choices=['smoke','full','evaluate'],default='smoke');p.add_argument('--shards',type=int,default=4);p.add_argument('--concurrency',type=int,default=4)
    a=p.parse_args();submit(a.out,a.source,a.mode,a.smoke,a.shards,a.concurrency)
if __name__=='__main__':main()
