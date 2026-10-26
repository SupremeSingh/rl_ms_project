"""Prepare new frozen-critic and online Numina splits without generating answers."""
import argparse
import json
from pathlib import Path
from math_rl.data import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('models/qwen-math'))
    p.add_argument('--revisions',type=Path,help='JSON of immutable Numina/GSM8K/MATH dataset revisions')
    p.add_argument('--offline-questions',type=int,default=13000)
    args=p.parse_args()
    prepare(args.out,args.model,json.loads(args.revisions.read_text()) if args.revisions else None,args.offline_questions)
