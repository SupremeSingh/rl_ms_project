"""Read-only progress and chunk-size diagnostics from saved TRAINING rollouts.

No generation, fitting, GPU, network, or test-set selection. Decoded-only logs
are re-tokenized for inspection: their sizes need not match sampled action IDs.
"""
import argparse
from collections import Counter
from bisect import bisect_right
from types import SimpleNamespace
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def json_lines(path):
    # A running writer may leave its last line incomplete. Other corruption fails.
    with path.open() as handle:
        for line in handle:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith('\n'):
                    return
                raise


def cases(study):
    return [p for p in [study, *sorted(study.iterdir())] if p.is_dir() and
            any((p / name).exists() for name in ('run.json', 'result.json', 'chunk-metrics.jsonl', 'rollouts', 'linear-critic/metrics.jsonl'))]


def progress(study):
    print(f'\nSTUDY: {study}')
    manifest = read(study/'manifest.json') if (study/'manifest.json').exists() else {}
    print(f"Declared fallback: {manifest.get('chunk_fallback_tokens', 'not recorded in this older study')}")
    for path in sorted(study.glob('*status.json')):
        status = read(path)
        print(f"{path.name}: {status.get('state', '?')}" + (f"; {status['error'][:180]}" if 'error' in status else ''))
    print(f"{'case':38} {'update':>7} {'fallback answers':>18} {'final accuracy':>20}")
    for case in cases(study):
        chunk = list(json_lines(case/'chunk-metrics.jsonl')) if (case/'chunk-metrics.jsonl').exists() else []
        linear = list(json_lines(case/'linear-critic/metrics.jsonl')) if (case/'linear-critic/metrics.jsonl').exists() else []
        result = read(case/'result.json') if (case/'result.json').exists() else {}
        step = result.get('updates', max((r.get('step', 0) for r in chunk+linear), default=0))
        fallback = str(sum(r['segmentation_fallback_answers'] for r in chunk)) if chunk and all('segmentation_fallback_answers' in r for r in chunk) else 'not recorded'
        accuracy = 'not finished'
        for split in ('test', 'validation'):
            if f'final_{split}' in result:
                accuracy = f"{100*result[f'final_{split}']['accuracy']:.2f}% ({split})"
        print(f'{case.name:38} {step:7} {fallback:>18} {accuracy:>20}')
    return manifest


def training_samples(study, limit):
    """Round-robin over cases' latest training dump; never open test predictions."""
    streams = []
    for case in cases(study):
        files = list((case/'rollouts').glob('*.jsonl'))
        if files:
            latest = max(files, key=lambda p: (int(p.stem) if p.stem.isdigit() else -1, p.name))
            streams.append((latest, iter(json_lines(latest))))
    count = 0
    while streams and count < limit:
        active = []
        for path, stream in streams:
            try:
                row = next(stream)
            except StopIteration:
                continue
            yield path, row
            count += 1
            active.append((path, stream))
            if count == limit:
                return
        streams = active


def action_tokens(row, tokenizer):
    for key in ('response_token_ids', 'token_ids'):
        value = row.get(key)
        if isinstance(value, list) and value and all(type(v) is int for v in value):
            return value, 'original IDs'
    text = row.get('output', row.get('response'))
    if not isinstance(text, str) or not text:
        return None, 'missing response'
    return tokenizer.encode(text, add_special_tokens=False), 're-tokenized text (approximate)'


def distribution(values):
    if not values:
        return 'no observations'
    values = sorted(values)
    def q(p):
        pos = (len(values)-1)*p
        lo = int(pos); hi = min(lo+1, len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(pos-lo)
    return f'n={len(values)}, mean={sum(values)/len(values):.1f}, p25={q(.25):.1f}, median={q(.5):.1f}, p75={q(.75):.1f}, p95={q(.95):.1f}'


def inspection_plan(tokenizer, tokens, fallback_tokens):
    # Call the pinned upstream splitter directly so this read-only tool also works
    # with older v3 installs whose wrapper lacked configurable fallback/status.
    from slice_critic.chunks import token_character_starts
    from slice_critic._vendor.vineppo_math import split_solution_inplace
    body = tokens[:-1] if tokens[-1] == tokenizer.eos_token_id else tokens
    reason = None
    starts = [0]
    if body:
        try:
            text, owners = token_character_starts(tokenizer, body)
        except ValueError as exc:
            # Narrow compatibility with both old and new unmapped-token errors.
            if not str(exc).startswith('Empty or unknown sampled ByteLevel token'):
                raise
            starts = list(range(0, len(tokens), fallback_tokens))
            return SimpleNamespace(starts=starts, ends=starts[1:]+[len(tokens)], fallback_reason='unmapped_token_id')
        try:
            points = sorted(set(split_solution_inplace(text) if text else [0]))
            if not points or points[0] != 0 or points[-1] != len(text):
                raise ValueError('Invalid text partition')
        except (AssertionError, ValueError) as exc:
            reason = 'upstream_assertion' if isinstance(exc, AssertionError) else 'invalid_partition'
            starts = list(range(0, len(tokens), fallback_tokens))
        else:
            groups = [bisect_right(points, a)-1 for a in owners]
            starts += [i for i in range(1,len(body)) if groups[i] != groups[i-1]]
    return SimpleNamespace(starts=starts, ends=starts[1:]+[len(tokens)], fallback_reason=reason)


def inspect(studies, tokenizer, limit, examples, fallback_tokens):
    normal_lengths, fallback_lengths, response_lengths, counts = [], [], [], Counter()
    shown = []
    for study in studies:
        for path, row in training_samples(study, limit):
            tokens, source = action_tokens(row, tokenizer)
            counts[source] += 1
            if not tokens:
                continue
            plan = inspection_plan(tokenizer, tokens, fallback_tokens)
            key = 'fallback' if plan.fallback_reason else 'normal'
            counts[key] += 1
            lengths = [b-a for a,b in zip(plan.starts, plan.ends)]
            (fallback_lengths if plan.fallback_reason else normal_lengths).extend(lengths)
            response_lengths.append(len(tokens))
            # Keep examples of both paths when present, without selecting by reward.
            if not any(v[0] == key for v in shown):
                shown.insert(0, (key, path, tokens, plan, source))
            elif len(shown) < examples:
                shown.append((key, path, tokens, plan, source))
    print('\nCHUNK DIAGNOSTICS — latest saved training dumps only; bounded convenience sample')
    print(f'Inspection fallback size: {fallback_tokens}; record counts: {dict(counts)}')
    n = counts['normal']+counts['fallback']
    print(f"Observed segmentation failures: {counts['fallback']}/{n} ({counts['fallback']/n:.1%})" if n else 'No saved training responses found yet.')
    print('Response tokens: '+distribution(response_lengths))
    print('Successful heuristic chunk tokens: '+distribution(normal_lengths))
    print('Fixed fallback chunk tokens: '+distribution(fallback_lengths))
    print('Decoded-only dumps omit original action IDs/EOS: re-tokenized sizes are approximate, not a replay audit.')
    print('Use normal-chunk median/p75 to judge granularity; this does not select the best size for PPO accuracy.')
    for key, path, tokens, plan, source in shown[:examples]:
        print(f'\nEXAMPLE {key}: {path}; {source}; reason={plan.fallback_reason}; {len(tokens)} tokens / {len(plan.starts)} chunks')
        for a,b in list(zip(plan.starts, plan.ends))[:8]:
            text = tokenizer.decode(tokens[a:b], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            print(f'  [{a}:{b}] ({b-a} tokens) {text[:240]!r}')
        if len(plan.starts)>8:
            print(f'  ... {len(plan.starts)-8} more chunks')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, action='append', required=True)
    p.add_argument('--model', type=Path, default=Path('models/qwen-math'))
    p.add_argument('--max-answers', type=int, default=256, help='Per study; latest training dumps only')
    p.add_argument('--examples', type=int, default=3)
    p.add_argument('--fallback-tokens', type=int, help='Inspection only; default is the first study manifest setting or 32')
    p.add_argument('--status-only', action='store_true', help='No tokenizer or sample analysis')
    a = p.parse_args()
    if min(a.max_answers, a.examples) < 1 or (a.fallback_tokens is not None and a.fallback_tokens < 1):
        p.error('Sizes and sample limits must be positive')
    for study in a.study:
        if not study.is_dir():
            p.error(f'Study directory not found: {study}')
    manifests = [progress(study) for study in a.study]
    if a.status_only:
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    size = a.fallback_tokens if a.fallback_tokens is not None else manifests[0].get('chunk_fallback_tokens', 32)
    inspect(a.study, tokenizer, a.max_answers, a.examples, size)


if __name__ == '__main__':
    main()
