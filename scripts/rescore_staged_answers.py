"""Read-only final-section Math-Verify audit of saved staged answers; no GPU."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from math_rl.ppo_reward import compute_score, verifier_command
from math_rl.provenance import sha256


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def audit_flags(text):
    flags = []
    if len(set(re.findall(r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)', text))) > 1:
        flags.append('multiple numeric values: inspect extraction and relevance')
    if text.count(r'\boxed') > 1:
        flags.append('multiple boxed expressions')
    return flags


def render(path, rows):
    parts = ['<!doctype html><meta charset="utf-8"><title>Staged answer review</title>',
        '<style>body{max-width:1000px;margin:30px auto;font:16px sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border-bottom:1px solid #ccc;padding:12px}</style>',
        '<h1>Staged answer review</h1><p>All saved attempts. Scores are automated; flags request human review, not automatic rejection. Generated code is text, not executed.</p>']
    for r in rows:
        parts.append('<details><summary>' + escape(f"{r['id']} | {r['old_status']} → {r['new_status']} | flags: {', '.join(r['flags']) or 'none'}") + '</summary><pre>' + escape(
            f"QUESTION: {r['question']}\nREFERENCE: {r['reference']}\nFINAL SECTION:\n{r['final']}\nEXTRACTED: {r['extracted']}\nFULL RESPONSE:\n{r['response']}\nSTAGES:\n{json.dumps(r.get('stages', []), indent=2)}") + '</pre></details>')
    path.write_text('\n'.join(parts))


def run(source, out):
    import os
    import subprocess
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Audit output must be separate from the source')
    data = source / 'data'
    questions_path = data / 'questions.json'
    questions = {q['id']: q for q in json.loads(questions_path.read_text())['questions']}
    files = sorted((data / 'trajectories').glob('*.json'))
    if not files:
        raise ValueError('No completed batches to audit')
    provenance = json.loads(subprocess.check_output(verifier_command() + ['--provenance'], text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / 'src'))))
    manifest = dict(source=str(source), files={str(p): sha256(p) for p in [questions_path, *files]},
        code={str(p): sha256(p) for p in [Path(__file__), ROOT / 'src/math_rl/math_verify_reward.py',
              ROOT / 'src/math_rl/ppo_reward.py', ROOT / 'src/math_rl/verifier_batch.py']}, verifier=provenance)
    out.mkdir(parents=True, exist_ok=True)
    mp = out / 'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text()) != manifest:
            raise ValueError('Audit inputs or code changed; choose a fresh audit directory')
    elif any(out.iterdir()):
        raise ValueError('Choose an empty audit output')
    else:
        save(mp, manifest)
    (out / 'batches').mkdir(exist_ok=True)
    rows = []
    for path in files:
        target = out / 'batches' / path.name
        if target.exists():
            rows.extend(json.loads(target.read_text()))
            continue
        batch = json.loads(path.read_text())
        q = questions[batch['question_id']]
        answers = batch['responses']
        eligible = [i for i, r in enumerate(answers) if r.get('final_stage_text', '').strip()
                    and r.get('verifier_status') != 'invalid_token_id']
        scores = compute_score(['math_numeric'] * len(eligible),
            ['Final answer: ' + answers[i]['final_stage_text'] for i in eligible],
            [q['ground_truth']] * len(eligible), exclude_errors=True) if eligible else []
        scores = dict(zip(eligible, scores, strict=True))
        results = []
        for i, answer in enumerate(answers):
            score = scores.get(i, dict(score=None, verifier_status='invalid_token_id' if
                answer.get('verifier_status') == 'invalid_token_id' else 'empty_final_stage', extracted=''))
            final = answer.get('final_stage_text', '')
            flags = audit_flags(final)
            if score['verifier_status'] not in ('correct', 'incorrect'):
                flags.append('unresolved verifier result')
            results.append(dict(id=f"{q['id']}/{i}", question=q['question'], reference=q['ground_truth'],
                response=answer['response'], final=final, stages=answer.get('stages', []), extracted=score.get('extracted', ''),
                old_status=answer['verifier_status'], old_score=answer.get('score'),
                new_status=score['verifier_status'], new_score=score['score'], flags=flags))
        save(target, results)
        rows.extend(results)
        print(f'Audited {len(rows)} saved answers', flush=True)
    old, new = Counter(r['old_status'] for r in rows), Counter(r['new_status'] for r in rows)
    summary = dict(answers=len(rows), old_statuses=dict(old), new_statuses=dict(new),
        answer_accuracy=new['correct'] / len(rows),
        recovered_incorrect=sum(r['old_status'] == 'invalid_final_stage' and r['new_status'] == 'incorrect' for r in rows),
        recovered_correct=sum(r['old_status'] == 'invalid_final_stage' and r['new_status'] == 'correct' for r in rows),
        flagged_for_review=sum(bool(r['flags']) for r in rows),
        scope='Saved completed batches only, not a representative benchmark. Final section only. Source files unchanged. Flags do not establish ambiguity or correctness.')
    save(out / 'summary.json', summary)
    save(out / 'answers.json', rows)
    render(out / 'responses.html', rows)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.out)
