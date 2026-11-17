"""Print a recovery job plan by default. --submit explicitly starts new work."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
from recover_compare import checked, combinations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--submit', action='store_true')
    a = p.parse_args(); out = a.out.resolve()
    plan = json.loads((out/'recovery.json').read_text())
    m = checked(out/'manifest.json')
    fits = [i for i,(b,arm,s) in enumerate(combinations(m))
            if not (out/'fits'/f'{b}-{arm}-{s}'/'complete.json').exists()]
    mc = [i for i in range(m['shards']) if not (out/f'mc-{i}.json').exists()]
    # Never automatically retry a partially written recovery task.
    for i in fits:
        b,arm,s = combinations(m)[i]
        if (out/'fits'/f'{b}-{arm}-{s}').exists():
            raise ValueError('Partial recovery fit exists. Prepare a fresh recovery directory before retrying.')
    if a.submit:
        ledger = out/'recovery-jobs.tsv'
        with ledger.open('x') as f:
            f.write('stage\tjob_id\n')
    previous = None
    for stage, tasks in [('fit', fits), ('mc', mc), ('evaluate', None)]:
        if tasks == []:
            continue
        cmd = ['sbatch', '--parsable', '--partition=compsci-gpu', '--nodes=1', '--ntasks=1',
               '--cpus-per-task=4', '--mem='+('64G' if stage=='fit' else '32G'),
               '--time=24:00:00', '--job-name=encoder-recovery-'+stage,
               f'--output={out}/{stage}-%A_%a.log']
        if stage != 'evaluate':
            cmd += ['--gres=gpu:a5000:1', '--array='+','.join(map(str,tasks))+'%2']
        if previous:
            cmd += ['--dependency=afterok:'+previous, '--kill-on-invalid-dep=yes']
        cmd += ['tools/encoder/run_recovery.sh', stage, str(out)]
        print(shlex.join(cmd), flush=True)
        if a.submit:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
            if not re.fullmatch(r'\d+(;[^\s]+)?\s*', result.stdout):
                raise RuntimeError('Ambiguous submission; inspect queue before doing anything else')
            previous = result.stdout.strip().split(';')[0]
            with ledger.open('a') as f:
                f.write(f'{stage}\t{previous}\n')
        else:
            previous = '<'+stage+'-job-id>'
    print('Submitted recovery.' if a.submit else 'Dry run only. No jobs submitted.')


if __name__ == '__main__':
    main()
