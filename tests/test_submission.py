"""Slurm submission tests use fake commands; never contact the scheduler."""
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def test_ambiguous_submission_stops_dependency_chain_and_prevents_retry(tmp_path):
    out=tmp_path/'study';out.mkdir()
    (out/'manifest.json').write_text(json.dumps(dict(seeds=[301],conditions={'ppo':{}})))
    commands=tmp_path/'bin';commands.mkdir()
    fake=commands/'sbatch'
    fake.write_text('#!/usr/bin/env bash\necho "Socket timed out" >&2\nexit 1\n');fake.chmod(0o755)
    env=dict(os.environ,PATH=str(commands)+os.pathsep+os.environ['PATH'])
    command=['bash','scripts/submit_numina_online.sh','--out',str(out),'--full','--smoke-first']
    first=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
    assert first.returncode and 'Submission failed for smoke' in first.stderr
    assert not (out/'jobs.tsv').exists()
    second=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
    assert second.returncode and 'already attempted' in second.stderr


def test_successful_chain_has_explicit_dependencies(tmp_path):
    out=tmp_path/'study';out.mkdir()
    (out/'manifest.json').write_text(json.dumps(dict(seeds=[301],conditions={'ppo':{}})))
    commands=tmp_path/'bin';commands.mkdir();calls=tmp_path/'calls'
    fake=commands/'sbatch'
    fake.write_text('#!/usr/bin/env bash\necho "$*" >> "$CALLS"\nn=$(wc -l < "$CALLS")\necho "$((n))"\n');fake.chmod(0o755)
    env=dict(os.environ,PATH=str(commands)+os.pathsep+os.environ['PATH'],CALLS=str(calls))
    result=subprocess.run(['bash','scripts/submit_numina_online.sh','--out',str(out),'--full','--smoke-first'],
                          cwd=ROOT,env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    rows=calls.read_text().splitlines()
    assert len(rows)==4 and '--dependency=afterok:1' in rows[1]
    assert '--dependency=afterok:2' in rows[2] and '--dependency=afterok:3' in rows[3]
    assert '--array=0-0%6' in rows[2]
