"""Standard-library-only Slurm controller for the expanded pooled comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from question_source import provenance,load_questions,main_budget

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def gate(smoke,source,questions):
    m=read(smoke/'manifest.json');v=read(smoke/'verification.json')
    if m['protocol']!='encoder-vine-unpadded-v3' or m['mode']!='smoke' or not v['passed'] or v['manifest_sha256']!=digest(smoke/'manifest.json') or v['summary_sha256']!=digest(smoke/'summary.json'):raise ValueError('Passed matching expanded smoke required')
    if m['source_hash']!=digest(source/'manifest.json'):raise ValueError('Smoke actor source mismatch')
    if digest(m['question_metadata_path'])!=m['question_metadata_hash']:raise ValueError('Question provenance changed since smoke')
    if digest(m['questions_path'])!=m['questions_hash']:raise ValueError('Smoke question source changed')
    if questions and m['questions_path']!=str(questions.resolve()):raise ValueError('Question source differs from smoke')
    for file,h in m['runtime'].items():
        if digest(ROOT/file)!=h:raise ValueError('Code changed since smoke')

def resolve_questions(explicit):
    if explicit is not None:
        path=explicit.resolve();provenance(path);main_budget(load_questions(path));return path
    matches=[]
    for root in ('outputs','data'):
        for p in (ROOT/root).glob('**/questions.json'):
            try:
                provenance(p);main_budget(load_questions(p));matches.append(p.resolve())
            except (KeyError,ValueError,OSError):continue
    if len(matches)!=1:raise ValueError('Specify --questions with saved split assignments and matching run.json/identity.json/manifest.json. Verified candidates: '+str(matches))
    return matches[0]

def submit(args):
    out=args.out.resolve();source=args.source.resolve()
    if args.mode=='main':
        if args.smoke is None:raise ValueError('Main requires --smoke')
        gate(args.smoke.resolve(),source,args.questions)
        if args.questions is None:args.questions=Path(read(args.smoke/'manifest.json')['questions_path'])
    args.questions=resolve_questions(args.questions)
    if out.exists():raise FileExistsError('Use a fresh study output')
    ledger=Path(str(out)+'-submission');ledger.mkdir(parents=True,exist_ok=False);(ledger/'logs').mkdir()
    previous=None
    def job(stage,array=None,gpus=1):
        nonlocal previous
        cmd=['sbatch','--parsable','--partition=compsci-gpu',f'--gres=gpu:a5000:{gpus}','--nodes=1','--ntasks=1',f'--cpus-per-task={16 if gpus==4 else 4}',f'--mem={128 if gpus==4 else 64}G','--time='+('04:00:00' if args.mode=='smoke' else '24:00:00'),f'--job-name=pooled-{stage}',f'--output={ledger}/logs/{stage}-%A_%a.out']
        if previous:cmd += [f'--dependency=afterok:{previous}','--kill-on-invalid-dep=yes']
        if array:cmd += [f'--array={array}']
        cmd += ['tools/encoder/run_compare.sh',stage,str(out)]
        if stage=='prepare':
            cmd += ['--mode',args.mode,'--source',str(source),'--numina',str(args.numina.resolve()),'--official',str(args.official.resolve())]
            if args.questions:cmd += ['--questions',str(args.questions.resolve())]
            elif args.mode=='main':cmd += ['--questions',read(args.smoke/'manifest.json')['questions_path']]
        (ledger/f'{stage}-intent.json').write_text(json.dumps(cmd,indent=2))
        r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=90,check=True)
        if not re.fullmatch(r'\d+(;[^\s]+)?\s*',r.stdout):raise RuntimeError('Ambiguous sbatch response; inspect queue before retry')
        previous=r.stdout.strip().split(';')[0]
        with (ledger/'jobs.tsv').open('a') as f:f.write(f'{stage}\t{previous}\n')
        print(f'Submitted {stage}: {previous}',flush=True)
    job('prepare');job('generate','0-7%4');job('profile');job('distributed-check',gpus=4)
    seeds=1 if args.mode=='smoke' else 3
    job('adapt',f'0-{seeds-1}%1',gpus=4)
    job('extract',f'0-{(seeds+1)*8-1}%4')
    fits=8 if args.mode=='smoke' else 48
    job('fit',f'0-{fits-1}%4');job('mc','0-7%4');job('evaluate')
    print('Jobs:',ledger/'jobs.tsv');print('Report:',out/'report.txt')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['smoke','main'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,default=Path('outputs/fixed-policy-critics-v1'));p.add_argument('--questions',type=Path);p.add_argument('--numina',type=Path,default=Path('outputs/numina-no-folds-v1'));p.add_argument('--official',type=Path,default=Path('third_party/llm2vec'));p.add_argument('--smoke',type=Path)
    submit(p.parse_args())

if __name__=='__main__':main()
