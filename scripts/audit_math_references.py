"""Audit stored original MATH solutions and recompute scores on an eligible subset.

Never mutates source outputs or relabels model responses. Existing verifier scores
are retained only when the reference passes the explicit single-number policy.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from math_rl.reference_audit import reference_decision
from math_rl.provenance import sha256, write_json


def audit(source, out):
    source = source.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source/'manifest.json').read_text())
    snapshot = source/'data/questions.json'
    saved = json.loads(snapshot.read_text())
    if sha256(snapshot) != manifest['data']['questions_sha256']:
        raise ValueError('Question snapshot hash mismatch')
    questions = saved['questions']
    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate question IDs')
    decisions = []
    for q in questions:
        if not q.get('reference_solution'):
            raise ValueError(f"Missing original reference solution: {q['id']}")
        decisions.append(dict(id=q['id'], split=q['split'], level=q['level'],
            question=q['question'], saved_reference=q['ground_truth'],
            reference_solution=q['reference_solution'],
            **reference_decision(q['question'],q['reference_solution'],q['ground_truth'])))
    with (out/'reference-audit.jsonl').open('w') as f:
        for d in decisions:
            f.write(json.dumps(d)+'\n')
    test = [q for q in questions if q['split']=='test']
    keep = np.array([next(d['eligible'] for d in decisions if d['id']==q['id']) for q in test])
    if not keep.any():
        raise ValueError('No eligible test questions')
    methods, scores, bases, hashes = {}, {}, {}, {}
    expected = [(q['id'], q['ground_truth']) for q in test]
    for name in manifest['conditions']:
        final, base = [], []
        for seed in manifest['seeds']:
            for filename, collection in [('final-test.jsonl',final),('base-test.jsonl',base)]:
                path = source/f'{name}-seed{seed}'/filename
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if [(r['id'],r['ground_truth']) for r in rows] != expected:
                    raise ValueError('Evaluation questions/references differ')
                if any(r['score'] not in (0,1) or r['score'] != int(r['verifier_status']=='correct') for r in rows):
                    raise ValueError('Invalid saved score/status')
                hashes[str(path)] = sha256(path)
                collection.append([r['score'] for r in rows])
                destination = out/f'{name}-seed{seed}'
                destination.mkdir(exist_ok=True)
                with (destination/filename).open('w') as f:
                    for r, eligible in zip(rows,keep):
                        if eligible: f.write(json.dumps(r)+'\n')
        final, base = np.asarray(final), np.asarray(base)
        scores[name], bases[name] = final[:,keep], base[:,keep]
        means = scores[name].mean(1)
        methods[name] = dict(original_accuracy=float(final.mean()), accuracy_mean=float(means.mean()),
            accuracy_seed_sd=float(means.std(ddof=1)) if len(means)>1 else None,
            accuracy_by_seed=dict(zip(map(str,manifest['seeds']),means.tolist())),
            base_accuracy=float(bases[name].mean()), gain_over_base_pp=float((scores[name]-bases[name]).mean()*100))
    base_arrays=list(bases.values())
    if any(not np.array_equal(b,base_arrays[0]) for b in base_arrays):
        raise ValueError('Base evaluation differs between methods')
    paired={}
    names=list(scores)
    for i,name in enumerate(names):
        for reference in names[:i]:
            delta=scores[name]-scores[reference]
            rng=np.random.default_rng(912)
            draws=[delta[np.ix_(rng.integers(len(delta),size=len(delta)),rng.integers(delta.shape[1],size=delta.shape[1]))].mean()*100 for _ in range(2000)]
            paired[f'{name}_minus_{reference}']=dict(mean_pp=float(delta.mean()*100),
                seed_differences_pp=dict(zip(map(str,manifest['seeds']),(delta.mean(1)*100).tolist())),
                descriptive_95_interval_pp=np.quantile(draws,[.025,.975]).tolist())
    excluded=[d for d in decisions if not d['eligible']]
    summary=dict(source=str(source),policy='single-reference-box-decimal-v1',
        original_test_questions=len(test),retained_test_questions=int(keep.sum()),
        excluded_by_split=dict(Counter(d['split'] for d in excluded)),
        exclusion_reasons=dict(Counter(reason for d in excluded for reason in d['reasons'])),
        methods=methods,paired=paired,manual_reference_review_complete=False,
        provenance=dict(questions_sha256=sha256(snapshot),manifest_sha256=sha256(source/'manifest.json'),
            policy_sha256=sha256(ROOT/'src/math_rl/reference_audit.py'),answer_hashes=hashes),
        scope='Post-hoc conservative eligibility correction; same rule for all methods/seeds/base. Original outputs unchanged. Existing verifier scores retained. Does not audit reasoning or repair historical training rewards. Inspected questions; exploratory intervals.')
    write_json(out/'summary.json',summary)
    lines=[f"REFERENCE AUDIT: retained {keep.sum()}/{len(test)} test questions",f"Excluded by split: {summary['excluded_by_split']}",
           'method | original accuracy | filtered accuracy | filtered base']
    lines += [f"{name}: {v['original_accuracy']:.4f} | {v['accuracy_mean']:.4f} | {v['base_accuracy']:.4f}" for name,v in methods.items()]
    lines += [f'{name}: {v}' for name,v in paired.items()]
    lines += ['Review reference-audit.jsonl; conservative exclusions can include valid single-answer questions.',summary['scope']]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    audit(a.source,a.out)
