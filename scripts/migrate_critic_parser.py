"""Preserve a failed critic run while applying known extraction/timeout handling fixes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from math_rl.math_verify_reward import PARSE_ERROR_POLICY, TIMEOUT_POLICY
from math_rl.ppo_reward import EXCLUSION_POLICY
from math_rl.provenance import sha256, write_json

GRADER = 'src/math_rl/math_verify_reward.py'
OLD_GRADER_SHA256 = '06627193f736d0ed8b43ac1bbf4a0a0781c44b945ce9d7c2f0a6ddd5f13af276'
PREVIOUS_FILE_HASHES = {
    GRADER: {OLD_GRADER_SHA256, '01a394be454cf2bc563e4d5f786390eb85271110d777de0e18fd75c494698dc4',
             'd94ee408f9c6526b923e9e30b5f25f717ebda88e1bce7c10d812ff9c8dbe7556'},
    'src/math_rl/verifier_batch.py': {'5e003088f3b44ec9474961e8b12adbd723ec03693035000ed80a2a0a433d6a3e',
                                    'cb79d1df136d9d92e957f24f07259653c69055103c9262862be1509345a95575'},
    'scripts/frozen_critics.py': {'d69750c53a905fcd1945e60198b893e7eece4d33d48860d373c89a755e6609e6',
                                '5bb1504faebc97b4df8ef5998a0f7d413e516744bf42e71a406946f671869981'},
    'src/math_rl/ppo_reward.py': {'e5b8aa89b5fca0db411683ac1560ee8e238ea58fcb79f8307a166dedef6bc523'},
    'src/math_rl/critic_probe.py': {'43f6525c22057b7fd5248dac371be2f5e162e2590527a7b504747d1c4e8f317d'},
}


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
    current = {name: sha256(ROOT / name) for name in manifest['files']}
    policies = dict(parse_error_policy=PARSE_ERROR_POLICY, timeout_policy=TIMEOUT_POLICY)
    if (manifest['files'] == current and all(manifest['verifier'].get(k) == v for k, v in policies.items())
            and manifest['config'].get('exclusion_policy') == EXCLUSION_POLICY):
        if not (out / 'exclusion-fix-migration.json').exists():
            raise ValueError('Already patched without a migration receipt')
        print('Exclusion policy already recorded; use the normal resume command.')
        return
    if (manifest['files'].get(GRADER) not in PREVIOUS_FILE_HASHES[GRADER]
            or manifest['verifier'].get('timeout_policy') not in (None, TIMEOUT_POLICY)
            or manifest['verifier'].get('parse_error_policy') not in (None, PARSE_ERROR_POLICY)):
        raise ValueError('Unknown prior grader; refusing to bypass provenance checks')
    if not PREVIOUS_FILE_HASHES.keys() <= manifest['files'].keys():
        raise ValueError('Incomplete code provenance')
    if any((out / 'features').glob('*.pt')) or any((out / 'heads').glob('*.pt')) or any(out.glob('*.pt')):
        raise ValueError('Existing feature/fit caches require a separate rebuild; only migrate generation-only runs')
    for name, digest in manifest['files'].items():
        if name in PREVIOUS_FILE_HASHES:
            if digest not in PREVIOUS_FILE_HASHES[name]:
                raise ValueError(f'Unknown prior code version: {name}')
        elif current[name] != digest:
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
            if row['verifier_status'] not in {'correct', 'incorrect', 'parse_failure', 'unsupported_parse', 'parse_timeout', 'verify_timeout'}:
                raise ValueError(f'Unexpected saved verifier status: {shard.name}')
            if row['score'] != int(row['verifier_status'] == 'correct') or not row['token_ids']:
                raise ValueError(f'Invalid saved score/tokens: {shard.name}')
        shards[shard.name] = sha256(shard)
    backup = out / 'manifest.before-exclusion-fix.json'
    if backup.exists():
        if backup.read_bytes() != original:
            raise ValueError('Existing backup differs; refusing to overwrite it')
    else:
        with backup.open('xb') as handle:
            handle.write(original)
    manifest['files'] = current
    manifest['verifier'].update(policies)
    manifest['config']['exclusion_policy'] = EXCLUSION_POLICY
    temporary = out / 'manifest.parser-fix.tmp'
    write_json(temporary, manifest)
    write_json(out / 'exclusion-fix-migration.json', dict(
        changes=dict(**policies, exclusion_policy=EXCLUSION_POLICY), old_manifest_sha256=sha256(backup),
        new_manifest_sha256=sha256(temporary), preserved_trajectory_sha256=shards,
        note='Saved scores unchanged. Resume regenerates only unsaved question batches.'))
    temporary.replace(path)
    print(f'Preserved {len(shards)} batches ({len(shards) * manifest["config"]["responses"]} answers).')
    print('Recorded answer-exclusion policy and backed up original manifest. Ready to resume.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    migrate(parser.parse_args().out)
