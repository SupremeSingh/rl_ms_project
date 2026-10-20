"""Submission failures must never silently remove dependency gates."""
import json
import os
from pathlib import Path
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('fail_at,mode', [(1,'error'), (1,'empty'), (1,'invalid'), (2,'error'), (3,'error'), (0,'ok')])
def test_submission_chain(tmp_path, fail_at, mode):
    source=tmp_path/'source'; sparse=tmp_path/'sparse'; out=tmp_path/'out'
    for p in (source,sparse,out,tmp_path/'bin'): p.mkdir()
    (out/'manifest.json').write_text(json.dumps(dict(preset='short',seeds=list(range(5)),conditions=dict.fromkeys('abcdef'))))
    (out/'selection.json').write_text('{}')
    log=tmp_path/'calls'
    stub=tmp_path/'bin/sbatch'
    stub.write_text(r"""#!/usr/bin/env python3
import os,sys,json
from pathlib import Path
p=Path(os.environ['CALL_LOG'])
with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\n')
n=len(p.read_text().splitlines())
if n==int(os.environ['FAIL_AT']):
    mode=os.environ['MODE']
    if mode=='error':
        print('Socket timed out',file=sys.stderr);sys.exit(1)
    if mode=='invalid': print('not a job id')
    sys.exit(0)
print(str(100+n)+';cluster')
""")
    stub.chmod(0o755)
    env=dict(os.environ,PATH=str(stub.parent)+':'+os.environ['PATH'],CALL_LOG=str(log),FAIL_AT=str(fail_at),MODE=mode)
    result=subprocess.run(['bash','scripts/submit_numina_online.sh',str(source),str(sparse),
        '--out',str(out),'--full','--smoke-first','--concurrency','6','--exclude','compsci-cluster-fitz-15'],cwd=ROOT,env=env,capture_output=True,text=True)
    calls=[json.loads(x) for x in log.read_text().splitlines()]
    if fail_at:
        assert result.returncode != 0
        assert len(calls)==fail_at
        recorded=(out/'jobs.tsv').read_text().splitlines() if (out/'jobs.tsv').exists() else []
        assert len(recorded)==fail_at-1
    else:
        assert result.returncode==0, result.stderr
        assert len(calls)==3
        assert '--array=0-29%6' in calls[1]
        assert '--dependency=afterok:101' in calls[1]
        assert '--dependency=afterok:102' in calls[2]

    assert all('--exclude=compsci-cluster-fitz-15' in c for c in calls)
    again=subprocess.run(['bash','scripts/submit_numina_online.sh',str(source),str(sparse),
        '--out',str(out),'--full','--smoke-first'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert again.returncode != 0
    assert len(log.read_text().splitlines())==len(calls)
