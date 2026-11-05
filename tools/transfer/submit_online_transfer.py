"""Parallel smoke/pilot/training chains with recorded IDs and fail-closed retries."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def submit(out,mode,source=None,concurrency=4):
    out=out.absolute()
    if not 1<=concurrency<=10:raise ValueError('Concurrency must be 1..10')
    if mode=='smoke':
        if source is None or out.exists():raise ValueError('New output and --source required')
        side=out.with_name(out.name+'-submission');side.mkdir(parents=True)
        stages=['prepare','smoke','gate']
    else:
        side=out.with_name(out.name+'-submission')
        m=json.loads((out/'manifest.json').read_text());gate=json.loads((out/'smoke.json').read_text())
        import hashlib
        if not gate['passed'] or gate['manifest']!=hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest():raise ValueError('Smoke gate failed')
        stages=['pilot','select'] if mode=='pilot' else ['run','report']
        if mode=='train' and not (out/'selection.json').exists():raise ValueError('Complete validation pilots first')
    logs=side/'logs';logs.mkdir(exist_ok=True);previous=None
    for stage in stages:
        marker=side/f'{stage}.json'
        if marker.exists():raise ValueError('Stage already submitted; inspect scheduler before recovery')
        gpu=stage in ('smoke','pilot','run')
        cmd=['sbatch','--parsable','--nodes=1','--ntasks=1','--cpus-per-task=8','--mem=128G',
             '--time='+('12:00:00' if stage=='run' else '04:00:00'),
             '--partition='+('compsci-gpu' if gpu else 'compsci'),'--job-name=transfer-ppo-'+stage,
             '--output='+str(logs/f'{stage}-%A_%a.out')]
        if gpu:cmd+=['--gres=gpu:a5000:2','--exclude=compsci-cluster-fitz-15']
        count={'smoke':2,'pilot':8,'run':10}.get(stage)
        if count:cmd+=['--array=0-'+str(count-1)+'%'+str(min(count,concurrency))]
        if previous:cmd+=['--dependency=afterok:'+previous,'--kill-on-invalid-dep=yes']
        cmd += [str(ROOT/'tools/transfer/run_online_transfer.sh'),stage,str(out)]
        if stage=='prepare':cmd+=['--source',str(source.absolute())]
        marker.write_text(json.dumps(dict(state='submitting',command=cmd),indent=2))
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=90);job=r.stdout.strip().split(';')[0]
        if r.returncode or not job.isdigit():raise RuntimeError(f'Submission stopped: {r.stderr}. Check scheduler; do not retry blindly.')
        marker.write_text(json.dumps(dict(state='submitted',job=job,command=cmd),indent=2))
        with (side/'jobs.tsv').open('a') as f:f.write(f'{stage}\t{job}\n')
        print('Submitted',stage+':',job,flush=True);previous=job
    print('Jobs:',side/'jobs.tsv')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path);p.add_argument('--mode',choices=['smoke','pilot','train'],default='smoke');p.add_argument('--concurrency',type=int,default=4)
    a=p.parse_args();submit(a.out,a.mode,a.source,a.concurrency)
if __name__=='__main__':main()
