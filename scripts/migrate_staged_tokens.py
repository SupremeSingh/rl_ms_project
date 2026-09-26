"""Recover a stopped, generation-only staged run without changing sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from math_rl.provenance import sha256
from math_rl.staged_generation import check_cached, VERSION

CODE = 'src/math_rl/staged_generation.py'
OLD_SHA = '19f08b7663734dd0a77b3a458d93361e18f85eba4b4b7bfc26cf50304ad4e401'


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def migrate(out):
    if VERSION != 'staged-plan-v1':
        raise ValueError('The corrected prompts/scoring require a fresh run. Rescore old answers with rescore_staged_answers.py; do not mix protocols.')
    out = Path(out).resolve()
    backup = out / 'token-guard-backup'
    updates, originals, checked = {}, {}, {}
    totals = dict(batches=0, answers=0, pending=0, excluded=0)
    vocabulary = json.loads((ROOT / 'models/qwen-math/tokenizer.json').read_text())
    valid = set(vocabulary['model']['vocab'].values())
    valid.update(t['id'] for t in vocabulary.get('added_tokens', []))
    if not valid:
        raise ValueError('Empty tokenizer vocabulary')

    def original(path):
        saved = backup / path.relative_to(out)
        data = saved.read_bytes() if saved.exists() else path.read_bytes()
        originals[path] = data
        return json.loads(data)

    for name in ('completion-2048', 'plan-2048'):
        run = out / name
        if not run.exists():
            continue
        status = json.loads((run / 'status.json').read_text())
        if status.get('stage') != 'generate' or status.get('state') != 'failed':
            raise ValueError(f'{name}: only stopped generation failures can be migrated')
        if any((run / 'data/features').glob('*.pt')) or (run / 'fit').exists() or list((run / 'data').glob('*.pt')):
            raise ValueError('Derived features/fits exist; refusing generation-only migration')
        top = original(run / 'manifest.json')
        data = original(run / 'data/manifest.json')
        if not top['config'].get('multistage'):
            raise ValueError('Expected multi-stage generation')
        if data != dict(top, questions_sha256=sha256(run / 'data/questions.json')):
            raise ValueError('Question/data manifest mismatch')
        code_key = str(ROOT / CODE)
        if top['files'].get(code_key) != OLD_SHA:
            raise ValueError('Unknown old staged code; refusing to bypass provenance')
        for path, digest in top['files'].items():
            if path != code_key and sha256(path) != digest:
                raise ValueError(f'Unrelated file changed: {path}')
        questions = json.loads((run / 'data/questions.json').read_text())['questions']
        if any(set(q['prompt_token_ids']) - valid for q in questions):
            raise ValueError('Saved prompts contain invalid tokens')
        for folder in ('trajectories', 'pending'):
            for path in sorted((run / 'data' / folder).glob('*.json')):
                raw = original(path)
                index = int(path.stem.split('-')[0])
                if not 0 <= index < len(questions) or raw['question_id'] != questions[index]['id']:
                    raise ValueError(f'Invalid cached question: {path}')
                rows = raw['responses'] if folder == 'trajectories' else [raw['response']]
                if folder == 'trajectories':
                    if len(rows) != top['config']['responses']:
                        raise ValueError('Incomplete completed batch')
                    totals['batches'] += 1
                    totals['answers'] += len(rows)
                else:
                    sample = int(path.stem.split('-')[1])
                    if not 0 <= sample < top['config']['responses']:
                        raise ValueError('Invalid pending sample')
                    totals['pending'] += 1
                repaired = [check_cached(row, valid) for row in rows]
                totals['excluded'] += sum(r.get('verifier_status') == 'invalid_token_id' for r in repaired)
                checked[str(path.relative_to(out))] = hashlib.sha256(originals[path]).hexdigest()
                if repaired != rows:
                    raw['responses' if folder == 'trajectories' else 'response'] = repaired if folder == 'trajectories' else repaired[0]
                    updates[path] = encoded(raw)
        for path, manifest in ((run / 'manifest.json', top), (run / 'data/manifest.json', data)):
            manifest['files'][code_key] = sha256(ROOT / CODE)
            updates[path] = encoded(manifest)
    if not updates:
        raise ValueError('No eligible condition found')
    # Validate every file before changing anything; backups make interrupted writes resumable.
    for path, content in updates.items():
        if path.read_bytes() not in (originals[path], content):
            raise ValueError(f'File changed since migration backup: {path}')
    for path, content in updates.items():
        saved = backup / path.relative_to(out)
        saved.parent.mkdir(parents=True, exist_ok=True)
        if not saved.exists():
            saved.write_bytes(originals[path])
        temporary = path.with_suffix('.token-guard.tmp')
        temporary.write_bytes(content)
        temporary.replace(path)
    receipt = dict(**totals, original_cache_sha256=checked, old_code_sha256=OLD_SHA,
        new_code_sha256=sha256(ROOT / CODE),
        note='Sampling unchanged. Invalid-token attempts excluded, not repaired or resampled. Valid saved answers preserved.')
    (out / 'token-guard-migration.json').write_bytes(encoded(receipt))
    print(f"Checked {totals['batches']} batches ({totals['answers']} answers) and {totals['pending']} pending answers; excluded {totals['excluded']} invalid attempts.")
    print('Backups and receipt saved. Resume with the original flags and --out pointing to this run.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    migrate(parser.parse_args().out)
