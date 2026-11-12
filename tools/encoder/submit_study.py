"""CPU-only Slurm controller; one GPU/job; complete smoke required before main."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def submit(out,mode,source,numina,sanity,official,smoke=None):
    out=out.resolve()
    if mode=='main':
        if smoke is None:raise ValueError('main needs --smoke')
        verification=json.loads((smoke/'verification.json').read_text())
        if not verification.get('passed'):raise ValueError('Smoke did not pass')
        m=json.loads((smoke/'manifest.json').read_text())
        import hashlib
        if verification['manifest_sha256']!=hashlib.sha256((smoke/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Smoke manifest changed')
        for name,digest in m['runtime'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Code changed since smoke')
        for name,digest in verification['files'].items():
            if hashlib.sha256((smoke/name).read_bytes()).hexdigest()!=digest:raise ValueError('Smoke artifacts changed')
        if not m['smoke'] or any(Path(m[k]).resolve()!=v.resolve() for k,v in [('source',source),('numina',numina),('official',official)]):
            raise ValueError('Smoke source mismatch')
    if out.exists():raise FileExistsError('Use a fresh study output')
    submission=out.with_name(out.name+'-submission');submission.mkdir(parents=True,exist_ok=False)
    (submission/'logs').mkdir();previous=None
    prepare=['--source',str(source.resolve()),'--numina',str(numina.resolve()),
             '--sanity',str(sanity.resolve()),'--official',str(official.resolve())]
    if mode=='smoke':prepare+=['--smoke']
    for stage in ('prepare','adapt','extract','fit','evaluate'):
        cmd=['sbatch','--parsable','--partition=compsci-gpu','--gres=gpu:a5000:1',
            '--cpus-per-task=4','--mem=48G','--time='+('02:00:00' if mode=='smoke' else '12:00:00'),
            '--job-name=encoder-'+stage,'--output='+str(submission/'logs'/f'{stage}-%A_%a.out')]
        if previous:cmd+=['--dependency=afterok:'+previous]
        if stage=='extract':cmd+=['--array=0-2%3']
        cmd+=[str(ROOT/'tools/encoder/run_study.sh'),stage,str(out)]
        if stage=='prepare':cmd+=prepare
        (submission/f'{stage}-intent.json').write_text(json.dumps(cmd,indent=2))
        try:r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=90,check=True)
        except Exception as exc:raise RuntimeError(f'Submission stopped at {stage}; inspect Slurm before retrying: {exc}') from exc
        if not re.fullmatch(r'\d+(;[^\s]+)?\s*',r.stdout):raise RuntimeError('Ambiguous submission: inspect Slurm')
        previous=r.stdout.strip().split(';')[0]
        with (submission/'jobs.tsv').open('a') as f:f.write(f'{stage}\t{previous}\n')
        print(f'Submitted {stage}: {previous}',flush=True)
    print('Report:',out/'report.txt')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','main'],required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('outputs/fixed-policy-critics-v1'))
    p.add_argument('--numina',type=Path,default=Path('outputs/numina-no-folds-v1'))
    p.add_argument('--sanity',type=Path,default=Path('outputs/encoder-sanity-v2'))
    p.add_argument('--official',type=Path,default=Path('third_party/llm2vec'))
    p.add_argument('--smoke',type=Path)
    a=p.parse_args();submit(a.out,a.mode,a.source,a.numina,a.sanity,a.official,a.smoke)
