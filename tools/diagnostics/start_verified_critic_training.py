"""Start training from a saved smoke gate without a stale Slurm gate dependency.

An additive launcher: never edits the prepared study's pinned runtime/tool files.
Recovery of a legacy submitting marker requires explicit dependency-rejection
acknowledgment and empty live/accounting queries for the training job name.
"""
import argparse
from datetime import datetime,timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import uuid

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def start(out,concurrency=6,recover=False,run=subprocess.run):
    out=Path(out).resolve()
    m=json.loads((out/'manifest.json').read_text())
    gate=json.loads((out/'smoke.json').read_text())
    if (m['protocol']!='slice-seven-critics-gpu-v1' or m['steps']!=120
            or m['seeds']!=[501,502,503] or len(m['conditions'])!=7
            or gate.get('passed') is not True or set(gate['checks'])!=set(m['conditions'])
            or gate['manifest_sha256']!=digest(out/'manifest.json')):
        raise ValueError('A successful matching seven-arm smoke gate is required')
    if not 1<=concurrency<=21:raise ValueError('Invalid concurrency')
    for name,h in m['provenance']['files'].items():
        if digest(ROOT/name)!=h:raise ValueError(f'Pinned runtime changed: {name}')
    for name,h in m['tools'].items():
        if digest(Path(__file__).with_name(name))!=h:raise ValueError(f'Pinned tool changed: {name}')
    jobs=dict(line.split('\t') for line in (out/'jobs.tsv').read_text().splitlines())
    if 'run' in jobs or 'report' in jobs:raise ValueError('Training/report already recorded; do not duplicate them')
    marker=out/'.submission-chain';marker.mkdir(exist_ok=True)
    previous=marker/'run'
    if previous.exists():
        record=json.loads((previous/'submission.json').read_text())
        old_dependency=f'--dependency=afterok:{jobs.get("gate", "missing")}'
        if (not recover or record.get('state')!='submitting' or record.get('job')
                or old_dependency not in record.get('command',[])):
            raise ValueError('Existing marker is not an explicitly acknowledged dependency rejection')
        user=os.environ.get('USER')
        if not user:raise ValueError('USER is required for duplicate-job checks')
        queries=[['squeue',f'--user={user}','--name=slice-critics-run','--noheader','--format=%A'],
                 ['sacct',f'--user={user}','--name=slice-critics-run',
                  f'--starttime={(datetime.now()-timedelta(days=3)).strftime("%Y-%m-%d")}',
                  '-X','--noheader','--parsable2','--format=JobID,State']]
        for command in queries:
            reply=run(command,text=True,capture_output=True,check=True,timeout=60)
            if reply.stdout.strip():
                raise ValueError('A live or recent training job exists; inspect Slurm rather than resubmitting')
        archived=marker/f'run-rejected-{uuid.uuid4().hex[:12]}'
        previous.rename(archived)
        (archived/'recovery.json').write_text(json.dumps(dict(
            reason='User acknowledged explicit sbatch Job dependency problem; no named jobs found',
            smoke_manifest_sha256=gate['manifest_sha256']),indent=2))
    stages=[('run',[f'--array=0-20%{concurrency}'],None),
            ('report',['--partition=compsci','--gres=none','--mem=32G','--time=01:00:00'],'run')]
    for name,options,parent in stages:
        folder=marker/name;folder.mkdir(exist_ok=False)
        command=['sbatch','--parsable',f'--chdir={ROOT}',f'--job-name=slice-critics-{name}',*options]
        if parent:command += [f'--dependency=afterok:{jobs[parent]}','--kill-on-invalid-dep=yes']
        command += [str(ROOT/'tools/diagnostics/submit_online_critics.sh'),name,str(out)]
        (folder/'submission.json').write_text(json.dumps(dict(state='submitting',command=command),indent=2))
        try:
            reply=run(command,cwd=ROOT,text=True,capture_output=True,check=True,timeout=60)
            job=reply.stdout.strip().split(';')[0]
            if not re.fullmatch('[0-9]+',job):raise ValueError('Unconfirmed Slurm ID')
        except (subprocess.SubprocessError,ValueError) as exc:
            raise RuntimeError(f'Stopped at {name}; inspect Slurm and jobs.tsv before retrying. Marker retained.') from exc
        jobs[name]=job
        (folder/'submission.json').write_text(json.dumps(dict(state='confirmed',job=job),indent=2))
        with (out/'jobs.tsv').open('a') as stream:stream.write(f'{name}\t{job}\n')
        print(f'Submitted {name}: {job}',flush=True)
    return jobs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=6)
    p.add_argument('--recover-dependency-rejection',action='store_true')
    a=p.parse_args()
    start(a.out,a.concurrency,a.recover_dependency_rejection)
