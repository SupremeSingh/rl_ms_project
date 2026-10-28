"""Final Numina experiment commands; setup never launches training implicitly."""
import argparse
from pathlib import Path
import time
import torch
from math_rl import experiment
from math_rl.reproduction import prepare_from_artifact, import_final, prepare_no_fold_study
from math_rl.provenance import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=['prepare','import-study','prepare-no-fold-study','smoke','calibrate','run','report'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--artifact', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--source', type=Path)
    p.add_argument('--task', type=int, default=0)
    args=p.parse_args();out=args.out.resolve()
    if args.stage=='prepare':
        if not args.artifact or not args.data: p.error('prepare requires --artifact and --data')
        prepare_from_artifact(args.artifact,args.data,out);return
    if args.stage=='import-study':
        if not args.source: p.error('import-study requires --source')
        import_final(args.source,out);return
    if args.stage=='prepare-no-fold-study':
        if not args.source: p.error('prepare-no-fold-study requires --source')
        prepare_no_fold_study(args.source,out);return
    m=experiment.validate(out)
    torch.set_num_threads(8)
    status=out/f'{args.stage}-{args.task}-status.json';started=time.perf_counter()
    write_json(status,dict(state='running',stage=args.stage,task=args.task))
    try:
        if args.stage=='run': experiment.run_task(out,m,args.task)
        else: getattr(experiment,args.stage)(out,m)
        write_json(status,dict(state='complete',stage=args.stage,seconds=time.perf_counter()-started))
    except BaseException as exc:
        write_json(status,dict(state='failed',stage=args.stage,error=str(exc)));raise

if __name__=='__main__':main()
