"""Read-only CPU audit of a completed encoder pilot; never fit or alter artifacts."""
import argparse
import ast
from collections import defaultdict
import json
import math
import re
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import study
import torch
from slice_critic.bottleneck import encode
from slice_critic.cumulative import raw_design
from slice_critic.training import _model


def metrics(rows,predictions):
    errors=defaultdict(list);p0=[];all_pred=[]
    for row,pred in zip(rows,predictions):
        pred=pred.double();y=row['y'].double()
        if len(pred)!=len(y) or not torch.isfinite(pred).all():
            raise ValueError('Invalid prediction vector')
        error=(pred-y).square()
        errors[row['question']].extend(error.tolist())
        p0.extend(error[torch.tensor(row['positions'])==0].tolist())
        all_pred.append(pred)
    values=torch.cat(all_pred)
    return dict(question_mean_brier=sum(sum(v)/len(v) for v in errors.values())/len(errors),
                prefix_mean_brier=sum(map(sum,errors.values()))/sum(map(len,errors.values())),
                initial_state_brier=sum(p0)/len(p0) if p0 else None,
                outside_unit_interval=float(((values<0)|(values>1)).double().mean()),
                prediction_min=float(values.min()),prediction_max=float(values.max()))


def row_signature(rows):
    return [(r['question'],r['split'],r['positions'],r['y'].tolist()) for r in rows]


def audit(out):
    torch.set_num_threads(4)
    m=study.validate(out)
    verification=study.read(out/'verification.json')
    if not verification.get('passed') or verification['manifest_sha256']!=study.sha256(out/'manifest.json'):
        raise ValueError('Completed verified study required')
    for name,digest in verification['files'].items():
        if study.sha256(out/name)!=digest:raise ValueError('Completed report changed')
    report=dict(scope='Post-hoc diagnostics on inspected data; no fitting or test-based model selection',
                manifest_sha256=study.sha256(out/'manifest.json'),arms={})
    reference=None
    summary=study.read(out/'summary.json')
    for arm in study.ARMS:
        print(f'Auditing {arm}',file=sys.stderr,flush=True)
        rows=study.feature_rows(out,arm)
        sig=row_signature(rows)
        if reference is None:reference=sig
        if sig!=reference:raise ValueError('Arms differ in row ordering, questions, boundaries or labels')
        metadata=study.read(out/arm/'features.json')
        if metadata['manifest_sha256']!=study.sha256(out/'manifest.json'):
            raise ValueError('Feature manifest mismatch')
        expected=m['model_identity'] if arm!='adapted' else study.read(out/'adaptation.json')['model_identity']
        if metadata['model_identity']!=expected:raise ValueError('Feature model identity mismatch')
        if arm!='causal':study.require_bidirectional(metadata['bidirectional_sensitivity'],0.)
        elif metadata['bidirectional_sensitivity']>1e-5:raise ValueError('Control is not causal')
        splits={s:[r for r in rows if r['split']==s] for s in ('train','val','test')}
        train_y=torch.cat([r['y'] for r in splits['train']])
        rate=float(train_y.double().mean())
        details=dict(bidirectional_sensitivity=metadata['bidirectional_sensitivity'],splits={},seeds={})
        for split,rs in splits.items():
            ys=torch.cat([r['y'] for r in rs])
            details['splits'][split]=dict(questions=len({r['question'] for r in rs}),answers=len(rs),
                prefixes=len(ys),positive_answers=sum(float(r['y'][0]) for r in rs),
                positive_prefix_rate=float(ys.double().mean()),
                train_mean_baseline=metrics(rs,[torch.full_like(r['y'],rate,dtype=torch.float64) for r in rs]))
        n=len(study.sampled(rows,'train')['y'])
        details['compressor_sampled_train_prefixes']=n
        details['compressor_optimizer_steps']=m['compressor_epochs']*math.ceil(n/2048)
        for seed in m['seeds']:
            folder=out/arm/f'fit-{seed}'
            fit=study.read(folder/'fit.json')
            for file,key in [('compressor.pt','compressor_sha256'),('ridge.pt','head_sha256')]:
                if study.sha256(folder/file)!=fit[key]:raise ValueError('Fit artifact changed')
            fitted=torch.load(folder/'compressor.pt',map_location='cpu',weights_only=True)
            head=torch.load(folder/'ridge.pt',map_location='cpu',weights_only=True)
            if head['states']!=len(train_y) or head['folds']!=1 or head['epsilon']!=.01:
                raise ValueError('Head does not match declared training data/recipe')
            compressor=_model(fitted);history=fitted['history']
            best=min(history,key=lambda x:x['validation_brier'])
            sample=study.sampled(rows,'val')
            with torch.inference_mode():
                standardized=(sample['x']-fitted['mean'])/fitted['scale']
                sample_pred=torch.cat([compressor(x)[0].sigmoid() for x in standardized.split(2048)])
                restored_brier=float((sample_pred-sample['y']).square().mean())
            detail=dict(history=history,best_epoch=best['epoch'],best_recorded_validation_brier=best['validation_brier'],
                        restored_sampled_validation_brier=restored_brier,
                        restored_validation_matches=abs(restored_brier-best['validation_brier'])<1e-6,
                        ridge_residual=fit['residual'],splits={})
            if not detail['restored_validation_matches']:raise ValueError('Restored compressor differs from selection metric')
            for split,rs in splits.items():
                raw=[];sigmoid=[]
                with torch.inference_mode():
                    for r in rs:
                        z=encode(compressor,r['x'],fitted['mean'],fitted['scale'],'cpu')
                        raw.append(raw_design(z)@head['weights'])
                        sigmoid.append(compressor.readout(z).flatten().sigmoid())
                detail['splits'][split]=dict(ridge=metrics(rs,raw),compressor_sigmoid=metrics(rs,sigmoid))
            details['seeds'][str(seed)]=detail
        recomputed=[details['seeds'][str(seed)]['splits']['test']['ridge']['question_mean_brier'] for seed in m['seeds']]
        recorded=summary['results'][arm]['by_seed']
        if len(recomputed)!=len(recorded) or any(abs(a-b)>1e-8 for a,b in zip(recomputed,recorded)):
            raise ValueError('Recomputed test scores differ from completed report')
        report['arms'][arm]=details
    adaptation=study.read(out/'adaptation.json')
    report['adaptation']=dict(training_windows=adaptation['training_windows'],seconds=adaptation['seconds'])
    report['trainer_configs']={stage:study.read(out/f'{stage}.json') for stage in ('mntp','simcse')}
    report['training_logs']={}
    for stage in ('mntp','simcse'):
        log=out/f'{stage}.log'
        if not log.exists():continue
        records=[]
        for match in re.finditer(r'\{[^{}\n]*\}',log.read_text(errors='replace')):
            try:record=ast.literal_eval(match.group())
            except (SyntaxError,ValueError):continue
            if isinstance(record,dict) and isinstance(record.get('loss'),(int,float)):
                records.append(record)
        report['training_logs'][stage]=dict(loss_records=len(records),first=records[:3],last=records[-10:])
    with (out/'adaptation.txt').open() as f:lines=[line.strip() for line in f]
    report['corpus']=dict(loader_lines=len(lines),blank_lines=sum(not s for s in lines),
                          distinct_lines=len(set(lines)),text_hash_matches=study.sha256(out/'adaptation.txt')==adaptation['text_sha256'])
    if report['corpus']['blank_lines'] or not report['corpus']['text_hash_matches']:
        raise ValueError('Invalid adaptation corpus')
    report['checks_passed']=True
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True,help='Fresh JSON file outside study directory')
    a=p.parse_args();source=a.study.resolve();dest=a.out.resolve()
    if dest==source or source in dest.parents:raise ValueError('Audit output must be outside original study')
    if dest.exists():raise FileExistsError('Use a fresh audit output')
    result=audit(source)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    for arm,detail in result['arms'].items():
        print(arm,'sampled training prefixes:',detail['compressor_sampled_train_prefixes'],
              'optimizer steps:',detail['compressor_optimizer_steps'])
        for seed,data in detail['seeds'].items():
            print(' seed',seed,'best epoch:',data['best_epoch'],
                  'validation:',json.dumps(data['splits']['val']))
    print('Audit saved:',dest)


if __name__=='__main__':main()
