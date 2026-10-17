"""Question-disjoint Numina comparison of nine PPO critic configurations."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import numpy as np
import torch
import math_comparison as comparison
from math_rl.large_probe import candidate, family_key, digest
from math_rl.provenance import sha256, snapshot, write_json
from math_rl.ppo_baseline import PPO_PROFILES, audit_health, select_profile

ROOT = comparison.ROOT
DATASET = 'AI-MO/NuminaMath-1.5'
SEEDS = [201, 202, 203, 204, 205]
SUBJECTS = ('algebra','counting_and_probability','geometry','intermediate_algebra',
            'number_theory','prealgebra','precalculus')


def read(path):
    return json.loads(Path(path).read_text())


def study_budget(preset):
    if preset == 'standard':
        return dict(counts=dict(train=2000,val=300,test=1000),steps=60,calibration_steps=30)
    if preset == 'large':
        return dict(counts=dict(train=3000,val=500,test=2000),steps=200,calibration_steps=60)
    raise ValueError('Unknown study preset')


def conditions():
    result = {'ppo': dict(method='ppo', representation='transformer')}
    for representation in ('plain', 'sparse', 'two_thirds', 'final'):
        for method in ('ridge', 'lstd'):
            result[f'{method}-{representation}'] = dict(method=method, representation=representation)
    return result


def select_questions(rows, blocked, tokenizer, counts):
    from math_rl.prompts import encode_completion
    eligible, reasons = {}, Counter()
    for index, row in enumerate(rows):
        q, reason = candidate(row)
        if q is None:
            reasons[reason] += 1
            continue
        family = family_key(q['question'])
        if family in blocked or family in eligible:
            reasons['overlap_or_duplicate'] += 1
            continue
        _, tokens = encode_completion(tokenizer, q['question'])
        if not 0 < len(tokens) <= 512:
            reasons['prompt_length'] += 1
            continue
        eligible[family] = dict(q, family=family, id=f'numina-cn-k12/{index}',
                                source_index=index, level=0)
    ordered = sorted(eligible.values(), key=lambda q: digest(['numina-online-v1',q['family']]))
    if len(ordered) < sum(counts.values()):
        raise ValueError(f'Only {len(ordered)} fresh eligible families; require {sum(counts.values())}. No fallback to contaminated questions.')
    selected, offset = [], 0
    for split, count in counts.items():
        selected.extend(dict(q, split=split) for q in ordered[offset:offset+count])
        offset += count
    return selected, dict(eligible=len(eligible), exclusions=dict(reasons), counts=counts)


def lock_filters(source, sparse, out):
    m = read(source/'manifest.json')
    protocol = read(sparse/'protocol.json')
    if protocol['width'] != 768 or protocol['source_manifest'] != sha256(source/'manifest.json'):
        raise ValueError('Require the 768 study fitted on this exact source')
    if sha256(source/'questions.json') != m['questions_sha256']:
        raise ValueError('Source question snapshot changed')
    selected = read(sparse/'screen.json')['selected']
    result = {}
    (out/'filters').mkdir(exist_ok=True)
    for kind in ('plain','sparse'):
        candidates = [(v['validation']['brier'], k, v) for k,v in selected.items() if k.startswith(kind+'-')]
        _, key, item = min(candidates)
        path = sparse/item['path']
        if sha256(path) != item['sha256']:
            raise ValueError('Filter checksum changed')
        record = torch.load(path, map_location='cpu', weights_only=True)
        if record['kind'] != kind or record['width'] != 768:
            raise ValueError('Wrong filter kind/width')
        dest = out/'filters'/f'{kind}.pt'
        if dest.exists():
            if sha256(dest) != item['sha256']:
                raise ValueError(f'Existing filter differs from selection: {dest}')
        else:
            shutil.copyfile(path,dest)
        result[kind] = dict(path=str(dest), sha256=sha256(dest), selected=key,
            validation_brier=item['validation']['brier'],
            training_seconds=record['seconds'])
    return m,result


def filter_training_trials(sparse):
    """Account for encoder training, not downstream Ridge/LSTD fit artifacts."""
    trials = []
    for trial in read(sparse/'screen.json')['trials']:
        path = sparse/f"{trial['kind']}-{trial['seed']}-{trial['lr']}.pt"
        record = torch.load(path, map_location='cpu', weights_only=True)
        seconds = float(record['seconds'])
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError(f'Invalid filter training time: {path}')
        trials.append(dict(path=path.name, seconds=seconds, sha256=sha256(path)))
    if not trials:
        raise ValueError('No filter training trials recorded in screen.json')
    return trials


def prepare(args):
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer
    from math_rl.prompts import make_prompt
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        raise ValueError('Study already prepared; use smoke/full stages or a new output')
    source,sparse=args.source.resolve(),args.sparse.resolve()
    if out == source or source in out.parents or out == sparse or sparse in out.parents:
        raise ValueError('Output must be separate from both source studies')
    m,filters=lock_filters(source,sparse,out)
    filter_trials=filter_training_trials(sparse)
    for name,checksum in m['model_hashes'].items():
        if sha256(ROOT/'models/qwen-math'/name) != checksum:
            raise ValueError('Base model/tokenizer differs from the offline study')
    if m['dataset'] != DATASET:
        raise ValueError('Unexpected source dataset')
    blocked={family_key(q['question']) for q in read(source/'questions.json')}
    revisions=m['revisions']
    for split in ('train','test'):
        for q in load_dataset('openai/gsm8k','main',split=split,revision=revisions['openai/gsm8k']):
            blocked.add(family_key(q['question']))
        for subject in SUBJECTS:
            for q in load_dataset('EleutherAI/hendrycks_math',subject,split=split,
                                  revision=revisions['EleutherAI/hendrycks_math']):
                blocked.add(family_key(q['problem']))
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/qwen-math',local_files_only=True)
    rows=load_dataset(DATASET,split='train',revision=revisions[DATASET])
    budget=study_budget(args.preset)
    questions,eligibility=select_questions(rows,blocked,tokenizer,budget['counts'])
    data=out/'data';data.mkdir()
    for split in ('train','val','test'):
        records=[dict(data_source='numina_numeric',prompt=make_prompt(q['question']),ability='math',
            reward_model=dict(style='rule',ground_truth=q['ground_truth']),
            extra_info=dict(index=q['source_index'],prompt_id=q['id'],split=split,level=0))
            for q in questions if q['split']==split]
        Dataset.from_list(records).to_parquet(str(data/f'{split}.parquet'))
    write_json(data/'questions.json',dict(questions=questions))
    manifest=dict(protocol='numina-online-v1',source=str(source),sparse_source=str(sparse),
        source_manifest_sha256=sha256(source/'manifest.json'),filters=filters,
        filter_pretraining=dict(trials=filter_trials,total_trial_seconds=sum(t['seconds'] for t in filter_trials),
            scope='Offline trial wall time, reported separately; excludes original generation and feature extraction. Not added to online run GPUh.'),
        layers=m['layers'],rms_eps=m['rms_eps'],model_hashes=m['model_hashes'],
        seeds=SEEDS,preset=args.preset,steps=budget['steps'],answers_per_update=64,calibration_steps=budget['calibration_steps'],
        conditions=conditions(),epsilon=.01,critic_lambda=.99,actor_lambda=.95,
        data_hashes={p.name:sha256(p) for p in data.iterdir()},eligibility=eligibility,
        revisions=revisions,provenance=snapshot(ROOT),
        primary_comparisons=['lstd-plain_minus_lstd-two_thirds','lstd-sparse_minus_lstd-two_thirds',
                             'ridge-plain_minus_ridge-two_thirds','ridge-sparse_minus_ridge-two_thirds'],
        scope='Fresh number-template-disjoint Numina questions relative to the source study, GSM8K and MATH. '
              'Not semantic/pretraining deduplication. Frozen validation-selected filters; zero online heads; '
              'historical-policy statistics uncorrected; fresh actor data only. Fixed 2048-token capped task.')
    write_json(out/'manifest.json',manifest)
    print('Prepared fresh splits:',eligibility['counts'],flush=True)


FOCUSED_METHODS = ('ppo', 'lstd-two_thirds', 'lstd-plain', 'lstd-sparse', 'ridge-sparse')
FOCUSED_COMPARISONS = (
    'lstd-two_thirds_minus_ppo',
    'lstd-plain_minus_lstd-two_thirds',
    'lstd-sparse_minus_lstd-plain',
    'lstd-sparse_minus_ridge-sparse',
)


def prepare_short(source, out, focused=False):
    """Fork data and validation decisions, never actor checkpoints or test results."""
    source, out = source.resolve(), out.resolve()
    if source == out or source in out.parents:
        raise ValueError('Short study needs a separate output directory')
    if any(out.iterdir()):
        raise ValueError('Short study output must be empty')
    m = read(source/'manifest.json')
    if m['conditions'] != conditions() or m['answers_per_update'] != 64:
        raise ValueError('Incompatible source protocol')
    smoke = read(source/'smoke.json')
    selection = read(source/'selection.json')
    if not smoke['passed'] or selection['profile'] not in PPO_PROFILES:
        raise ValueError('Require passed smoke and completed PPO calibration')
    for name, checksum in m['data_hashes'].items():
        if sha256(source/'data'/name) != checksum:
            raise ValueError('Source data changed')
    for f in m['filters'].values():
        if sha256(f['path']) != f['sha256']:
            raise ValueError('Source filter changed')
    for name, checksum in m['model_hashes'].items():
        if sha256(ROOT/'models/qwen-math'/name) != checksum:
            raise ValueError('Model changed')
    names = list(FOCUSED_METHODS) if focused else list(m['conditions'])
    if len(m['seeds']) < 3 or len(set(m['seeds'])) != len(m['seeds']):
        raise ValueError('Require at least three distinct source seeds')
    if not set(names).issubset(smoke.get('conditions', [])):
        raise ValueError('Source smoke does not cover requested methods')
    shutil.copytree(source/'data', out/'data')
    (out/'filters').mkdir()
    for kind, f in m['filters'].items():
        dest = out/'filters'/f'{kind}.pt'
        shutil.copyfile(f['path'], dest)
        f['path'] = str(dest)
    m['parent_study'] = dict(path=str(source), manifest_sha256=sha256(source/'manifest.json'),
                             provenance=m['provenance'])
    m.update(preset='short', steps=60, seeds=m['seeds'][:3], provenance=snapshot(ROOT))
    if focused:
        m['conditions'] = {name: m['conditions'][name] for name in names}
        m['primary_comparisons'] = list(FOCUSED_COMPARISONS)
    m['method_subset'] = 'focused' if focused else 'all'
    m['scope'] += ' Short-budget follow-up; same questions and reused validation calibration, not an independent benchmark.'
    for name, record in [('smoke', smoke), ('selection', selection)]:
        record['reused_from'] = str(source)
        write_json(out/f'{name}.json', record)
    write_json(out/'manifest.json', m)


def validate(out):
    m=read(out/'manifest.json')
    if m['provenance']['files'] != snapshot(ROOT)['files']:
        raise ValueError('Code/config changed since preparation; use a new study')
    for name,checksum in m['data_hashes'].items():
        if sha256(out/'data'/name)!=checksum: raise ValueError('Data changed')
    for f in m['filters'].values():
        if sha256(f['path'])!=f['sha256']: raise ValueError('Filter changed')
    for name,checksum in m['model_hashes'].items():
        if sha256(ROOT/'models/qwen-math'/name)!=checksum: raise ValueError('Model changed')
    return m


def command(out,m,name,seed,steps,profile='more_fitting',calibration=False,case=None):
    setting=m['conditions'][name];method=setting['method'];rep=setting['representation']
    cmd=comparison.command(out,method,seed,steps,feature_mode='frozen',
        ppo_profile=profile if method=='ppo' else 'legacy',
        evaluation_split='val' if calibration else 'test',case_name=case or f'{name}-seed{seed}')
    if method!='ppo':
        cmd += ['experiment.critic_statistics=cumulative','experiment.cumulative_epsilon=0.01',
                'experiment.critic_lambda=0.99','actor_rollout_ref.model.critic_encoder_precision=fp32',
                f'actor_rollout_ref.model.critic_feature_layer={"bottleneck" if rep in ("plain","sparse") else rep}']
        if rep in ('plain','sparse'):
            f=m['filters'][rep]
            cmd += [f'actor_rollout_ref.model.critic_bottleneck_path={f["path"]}',
                    f'actor_rollout_ref.model.critic_bottleneck_sha256={f["sha256"]}',
                    f'actor_rollout_ref.model.critic_bottleneck_layers={json.dumps(m["layers"],separators=(",",":"))}',
                    f'actor_rollout_ref.model.critic_bottleneck_rms_eps={m["rms_eps"]}']
    return cmd


def launch(out,m,name,seed,steps,profile='more_fitting',calibration=False,case=None):
    case=case or f'{name}-seed{seed}'
    result=out/case/'result.json'
    if result.exists():
        r=read(result)
        if r['updates']!=steps or r['seed']!=seed or r['evaluation_split']!=('val' if calibration else 'test'):
            raise ValueError('Incompatible completed run')
        return r
    if (out/case).exists():
        raise ValueError(f'Partial run {case}: preserve/rename its directory and log before retrying; no partial actor resume')
    env=dict(os.environ,RAY_ADDRESS='local',MATH_RL_SEED=str(seed),
             RAY_TMPDIR=tempfile.mkdtemp(prefix='nr-',dir='/tmp'))
    env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',None)
    with (out/f'{case}.log').open('w') as log:
        subprocess.run(command(out,m,name,seed,steps,profile,calibration,case),cwd=ROOT,env=env,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    r=read(result)
    if r['updates']!=steps or r['answers']!=steps*64: raise ValueError('Incomplete training')
    return r


def smoke(out,m):
    # Only validation questions, never final test scores, are read by smoke/calibration.
    for name in m['conditions']:
        launch(out,m,name,19,2,calibration=True,case=f'smoke-{name}')
    write_json(out/'smoke.json',dict(passed=True,conditions=list(m['conditions']),steps=2,
        scope='Integration check only; not a method ranking'))


def calibrate(out,m):
    if not read(out/'smoke.json')['passed']: raise ValueError('Smoke must pass first')
    candidates=[]
    for profile in PPO_PROFILES:
        case=f'calibration-{profile}'
        r=launch(out,m,'ppo',17,m['calibration_steps'],profile,True,case)
        health=audit_health(out/case/'ppo-critic-audit',m['calibration_steps'])
        candidates.append(dict(r,profile=profile,health=health))
    # Explicitly fail rather than silently proceeding with a weak/broken baseline.
    profile=select_profile(candidates)
    write_json(out/'selection.json',dict(profile=profile,candidates=candidates,
        tuning_training_gpu_hours=sum(c['training_gpu_hours'] for c in candidates),
        scope='Validation-only profile selection; no test access'))


def run_task(out,m,task):
    # Rotate arm order across seeds; array index does not change random seeds.
    jobs=[];names=list(m['conditions'])
    for i,seed in enumerate(m['seeds']):
        jobs.extend((name,seed) for name in names[i:]+names[:i])
    if not 0<=task<len(jobs): raise ValueError('Invalid task index')
    name,seed=jobs[task]
    launch(out,m,name,seed,m['steps'],read(out/'selection.json')['profile'])


def report(out,m):
    expected=[(q['id'],q['ground_truth']) for q in read(out/'data/questions.json')['questions'] if q['split']=='test']
    scores,results,base_ref={}, {}, None
    audit=(out/'audit.jsonl').open('w')
    health={}
    try:
        for name,setting in m['conditions'].items():
            arrays=[];runs=[]
            for seed in m['seeds']:
                case=out/f'{name}-seed{seed}';r=read(case/'result.json')
                if r['updates']!=m['steps'] or r['answers']!=m['steps']*64: raise ValueError('Unequal budgets')
                base=comparison.read_answers(case/'base-test.jsonl');rows=comparison.read_answers(case/'final-test.jsonl')
                for batch in (base,rows):
                    if [(v['id'],v['ground_truth']) for v in batch]!=expected: raise ValueError('Unpaired test set')
                b=[v['score'] for v in base];a=[v['score'] for v in rows]
                if base_ref is None: base_ref=b
                if b!=base_ref: raise ValueError('Starting model scores differ')
                if not np.isclose(np.mean(a),r['final_test']['accuracy'],rtol=0,atol=1e-10): raise ValueError('Inconsistent accuracy')
                for row in rows:
                    audit.write(json.dumps(dict(row,method=name,seed=seed))+'\n')
                if setting['method']=='ppo':
                    health[str(seed)]=audit_health(case/'ppo-critic-audit',m['steps'])
                else:
                    metrics=[json.loads(x) for x in (case/'linear-critic/metrics.jsonl').read_text().splitlines()]
                    if [v['step'] for v in metrics]!=list(range(1,m['steps']+1)): raise ValueError('Missing critic metrics')
                    r['critic_fit_seconds']=sum(v['fit_seconds'] for v in metrics)
                    r['last_critic']=metrics[-1]
                r['verifier_statuses']=dict(Counter(v['verifier_status'] for v in rows))
                r['mean_test_response_tokens']=float(np.mean([v['response_tokens'] for v in rows]))
                r['test_truncation_rate']=float(np.mean([v['capped'] for v in rows]))
                r['accounted_gpu_hours']=r['training_gpu_hours']+2*(r['initialization_seconds']+r['initial_test']['seconds']+r['final_test']['seconds'])/3600
                arrays.append(a);runs.append(r)
            a=np.array(arrays);scores[name]=a
            results[name]=dict(accuracy_mean=float(a.mean()),accuracy_seed_sd=float(a.mean(1).std(ddof=1)),
                accuracy_by_seed=dict(zip(map(str,m['seeds']),a.mean(1).tolist())),
                gain_pp=100*float(a.mean()-np.mean(base_ref)),
                training_gpu_hours_mean=float(np.mean([r['training_gpu_hours'] for r in runs])),
                accounted_gpu_hours_mean=float(np.mean([r['accounted_gpu_hours'] for r in runs])),runs=runs)
    finally: audit.close()
    paired={};names=list(scores)
    for i,name in enumerate(names):
        for ref in names[:i]:
            d=scores[name]-scores[ref];rng=np.random.default_rng(813)
            draws=np.array([d[np.ix_(rng.integers(len(d),size=len(d)),rng.integers(d.shape[1],size=d.shape[1]))].mean()*100 for _ in range(4000)])
            key=f'{name}_minus_{ref}'
            # Symmetric output for predeclared primary directions as well.
            paired[key]=dict(mean_pp=float(d.mean()*100),seed_differences_pp=(d.mean(1)*100).tolist(),
                descriptive_95_interval_pp=np.quantile(draws,[.025,.975]).tolist(),
                bonferroni_four_primary_interval_pp=np.quantile(draws,[.00625,.99375]).tolist())
            reverse=f'{ref}_minus_{name}'
            if reverse in m['primary_comparisons']:
                paired[reverse]=dict(mean_pp=float(-d.mean()*100),seed_differences_pp=(-d.mean(1)*100).tolist(),
                    descriptive_95_interval_pp=np.quantile(-draws,[.025,.975]).tolist(),
                    bonferroni_four_primary_interval_pp=np.quantile(-draws,[.00625,.99375]).tolist())
    summary=dict(methods=results,paired=paired,primary_comparisons=m['primary_comparisons'],
        base_accuracy=float(np.mean(base_ref)),ppo_health=health,selection=read(out/'selection.json'),
        filter_pretraining=m.get('filter_pretraining'),
        scope=m['scope'],limitations=f'{len(m["seeds"])} actor seeds, one selected filter per arm. Approximate seed/question bootstrap; '
        'four primary intervals use Bonferroni nominal coverage; other comparisons exploratory. '
        'No automatic superiority claim. Audit verifier answers before claims. Offline filter learning cost is additional.')
    write_json(out/'summary.json',summary)
    lines=['Numina online: fixed 768 bottlenecks and frozen single-layer controls; no buffer.',
        f'{m["steps"]} updates x 64 answers; {len(m["seeds"])} seeds; final greedy evaluation.',
        'method                   accuracy    training GPUh   accounted GPUh']
    lines += [f'{name:25} {r["accuracy_mean"]:.4f}      {r["training_gpu_hours_mean"]:.3f}          {r["accounted_gpu_hours_mean"]:.3f}' for name,r in results.items()]
    lines += [f'{k}: {paired[k]}' for k in m['primary_comparisons']]
    lines += [f'PPO calibration training GPUh (additional): {summary["selection"]["tuning_training_gpu_hours"]:.3f}',
        'Offline filter training/generation/extraction are additional costs; see summary.json.',
        f'PPO health passed for all seeds: {all(v["pass_checks"] for v in health.values())}',summary['scope'],summary['limitations']]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--sparse',type=Path);p.add_argument('--stage',choices=['prepare','short','smoke','calibrate','run','report'],required=True)
    p.add_argument('--focused', action='store_true', help='Five-method subset; only with --stage short')
    p.add_argument('--preset',choices=['standard','large','short'],default='standard');p.add_argument('--task',type=int,default=0);args=p.parse_args();args.out=args.out.resolve()
    if args.focused and args.stage != 'short': p.error('--focused requires --stage short')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.stage == 'short':
        if not args.source: p.error('Short preparation requires --source')
        prepare_short(args.source, args.out, focused=args.focused)
        return
    torch.set_num_threads(8)
    status=args.out/f'{args.stage}-{args.task}-status.json';started=time.perf_counter()
    write_json(status,dict(state='running',stage=args.stage,task=args.task))
    try:
        if args.stage=='prepare':
            if not args.source or not args.sparse: p.error('Preparation requires --source and --sparse')
            prepare(args)
        else:
            m=validate(args.out)
            if args.stage=='run': run_task(args.out,m,args.task)
            else: globals()[args.stage](args.out,m)
        write_json(status,dict(state='complete',stage=args.stage,seconds=time.perf_counter()-started))
    except BaseException as exc:
        write_json(status,dict(state='failed',stage=args.stage,error=str(exc),seconds=time.perf_counter()-started));raise


if __name__=='__main__': main()
