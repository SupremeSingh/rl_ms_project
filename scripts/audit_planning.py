"""Read-only rescoring and boundary audit of saved planning answers; no GPU generation."""
import argparse
from collections import Counter
import json
from pathlib import Path

from frozen_critics import atomic_json
from math_rl.ppo_reward import compute_score
from math_rl.provenance import sha256
from planning_checkpoints import headings, format_audit
from math_rl.conclusion_reward import RULE


def audit(source, out):
    source, out = source.resolve(), out.resolve()
    if out == source or source in out.parents:
        raise ValueError('Audit output must be separate from source')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'audit-manifest.json'
    identity = dict(source=str(source), rule=RULE,
                    code=sha256(Path(__file__)), boundary_code=sha256(Path(__file__).with_name('planning_checkpoints.py')))
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Audit resume requires unchanged source path and code')
    elif any(out.iterdir()):
        raise ValueError('Use an empty audit directory to preserve previous reports')
    atomic_json(manifest, identity)
    totals, hashes, changes = {}, {}, []
    import re
    for style in ('completion', 'plan'):
        data = source / f'{style}-2048/data'
        hashes[str(data / 'questions.json')] = sha256(data / 'questions.json')
        questions = json.loads((data / 'questions.json').read_text())['questions']
        counts, rows = Counter(), []
        for i, q in enumerate(questions):
            path = data / f'trajectories/{i:05d}.json'
            hashes[str(path)] = sha256(path)
            shard = json.loads(path.read_text())
            if shard['question_id'] != q['id']:
                raise ValueError('Question/trajectory mismatch')
            for j, answer in enumerate(shard['responses']):
                rows.append(dict(id=q['id'], sample=j, split=q['split'], question=q['question'],
                    ground_truth=q['ground_truth'], old=answer))
        with (out / f'{style}-audit.jsonl').open('w') as stream:
            for start in range(0, len(rows), 32):
                batch = rows[start:start + 32]
                scores = compute_score(['math_numeric'] * len(batch), [r['old']['response'] for r in batch],
                    [r['ground_truth'] for r in batch], exclude_errors=True, rule=RULE)
                for row, score in zip(batch, scores, strict=True):
                    text = row['old']['response']
                    old_boundary = list(re.finditer(r'(?im)^\s*(?:\*\*)?Solution:(?:\*\*)?\s*\n', text))
                    new_boundary = headings(text, r'(?:Step[- ]by[- ]step[ \t]+)?Solution')
                    counts['answers'] += 1
                    counts['old_correct'] += row['old']['verifier_status'] == 'correct'
                    counts['new_correct'] += score['verifier_status'] == 'correct'
                    counts['old_unique_heading'] += len(old_boundary) == 1
                    counts['new_unique_heading'] += len(new_boundary) == 1
                    counts['new_' + score['verifier_status']] += 1
                    changed = (score['score'], score['verifier_status'], score['extracted']) != (
                        row['old'].get('score'), row['old']['verifier_status'], row['old'].get('extracted', ''))
                    item = dict(row, new=score, format=format_audit(text), changed=changed)
                    stream.write(json.dumps(item) + '\n')
                    if changed:
                        changes.append(dict(condition=style, **item))
                print(f'Audited {style}: {min(start + 32, len(rows))}/{len(rows)}', flush=True)
        totals[style] = dict(counts)
    atomic_json(out / 'changes.json', changes)
    atomic_json(out / 'summary.json', dict(rule=RULE, totals=totals, changed=len(changes), source_hashes=hashes,
        scope='Diagnostic rescore only. No source labels, fitted heads or original metrics changed. '
              'Inspect changes and unchanged random examples; new rule is not ground truth. Counts pool all source splits.'))
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError('Source changed during audit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.source, args.out)
