"""Preserve a failed critic run while applying the narrow SympifyError parser fix."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from math_rl.math_verify_reward import PARSE_ERROR_POLICY
from math_rl.provenance import sha256, write_json

GRADER = 'src/math_rl/math_verify_reward.py'
OLD_GRADER_SHA256 = '06627193f736d0ed8b43ac1bbf4a0a0781c44b945ce9d7c2f0a6ddd5f13af276'


def migrate(out):
    out = Path(out).resolve()
    path = out / 'manifest.json'
    original = path.read_bytes()
    manifest = json.loads(original)
    status = json.loads((out / 'status.json').read_text())
    if status.get('stage') != 'generate' or status.get('state') != 'failed':
        raise ValueError('Only migrate a stopped run that failed during generation')
    if manifest['config'].get('protocol') != 'frozen-critics-v2':
        raise ValueError('Expected frozen-critics-v2; do not migrate a different protocol')
    current = sha256(ROOT / GRADER)
    if manifest['files'].get(GRADER) == current and manifest['verifier'].get('parse_error_policy') == PARSE_ERROR_POLICY:
        if not (out / 'parser-fix-migration.json').exists():
            raise ValueError('Already patched without a migration receipt')
        print('Parser fix already recorded; use the normal resume command.')
        return
    if manifest['files'].get(GRADER) != OLD_GRADER_SHA256 or 'parse_error_policy' in manifest['verifier']:
        raise ValueError('Unknown prior grader; refusing to bypass provenance checks')
    for name, digest in manifest['files'].items():
        if name != GRADER and sha256(ROOT / name) != digest:
            raise ValueError(f'Unrelated file changed: {name}. Restore it before migration.')
    if sha256(out / 'questions.json') != manifest['questions_sha256']:
        raise ValueError('Frozen questions changed')
    questions = json.loads((out / 'questions.json').read_text())['questions']
    shards = {}
    for shard in sorted((out / 'trajectories').glob('*.json')):
        index = int(shard.stem)
        batch = json.loads(shard.read_text())
        if not 0 <= index < len(questions) or batch['question_id'] != questions[index]['id']:
            raise ValueError(f'Question mismatch: {shard.name}')
        if len(batch['responses']) != manifest['config']['responses']:
            raise ValueError(f'Incomplete batch: {shard.name}')
        for row in batch['responses']:
            if row['verifier_status'] not in {'correct', 'incorrect', 'parse_failure', 'unsupported_parse'}:
                raise ValueError(f'Unexpected saved verifier status: {shard.name}')
            if row['score'] != int(row['verifier_status'] == 'correct') or not row['token_ids']:
                raise ValueError(f'Invalid saved score/tokens: {shard.name}')
        shards[shard.name] = sha256(shard)
    backup = out / 'manifest.before-sympify-fix.json'
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError('Existing backup differs; refusing to overwrite it')
    else:
        with backup.open('xb') as handle:
            handle.write(original)
    manifest['files'][GRADER] = current
    manifest['verifier']['parse_error_policy'] = PARSE_ERROR_POLICY
    temporary = out / 'manifest.parser-fix.tmp'
    write_json(temporary, manifest)
    write_json(out / 'parser-fix-migration.json', dict(
        change=PARSE_ERROR_POLICY, old_manifest_sha256=sha256(backup),
        new_manifest_sha256=sha256(temporary), preserved_trajectory_sha256=shards,
        note='Saved scores unchanged. Resume regenerates only unsaved question batches.'))
    temporary.replace(path)
    print(f'Preserved {len(shards)} batches ({len(shards) * manifest["config"]["responses"]} answers).')
    print('Recorded parser fix and backed up original manifest. Ready to resume.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    migrate(parser.parse_args().out)
