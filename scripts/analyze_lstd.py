"""Paired length diagnostics for frozen LSTD predictions; never changes critic inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from math_rl.critic_probe import PREFIX_SAMPLING
from math_rl.provenance import sha256, write_json


def metadata(source, questions, data):
    """Reconstruct exact sampled-prefix ordering from saved raw trajectories."""
    rows, statuses, fingerprints = [], Counter(), {}
    total = 0
    for index, q in enumerate(questions):
        if q['split'] != 'test':
            continue
        path = source / 'trajectories' / f'{index:05d}.json'
        raw = json.loads(path.read_text())
        if raw['question_id'] != q['id']:
            raise ValueError('Question/trajectory mismatch')
        fingerprints[path.name] = sha256(path)
        question_index = q.get('source_index', index)
        for response_index, response in enumerate(raw['responses']):
            total += 1
            statuses[response['verifier_status']] += 1
            length = len(response['token_ids'])
            if response['verifier_status'] not in ('correct', 'incorrect') or not length:
                continue
            if response['score'] not in (0, 1) or response['finish_reason'] not in ('stop', 'length'):
                raise ValueError('Invalid reward or termination metadata')
            rng = np.random.default_rng(np.random.SeedSequence(
                [PREFIX_SAMPLING['seed'], question_index, response_index]))
            count = PREFIX_SAMPLING['per_response'] - 1
            positions = [0] + (rng.integers(1, length, count).tolist() if length > 1 else [0] * count)
            for position in positions:
                rows.append((question_index, response_index, position, length,
                             length - position, response['finish_reason'] == 'length', response['score']))
    if not rows:
        raise ValueError('No retained test responses')
    columns = np.asarray(rows)
    result = {name: columns[:, i] for i, name in enumerate(
        ('question', 'answer', 'position', 'length', 'remaining', 'capped', 'y'))}
    for key in ('question', 'position', 'y'):
        if not np.array_equal(result[key], data[key].numpy()):
            raise ValueError(f'Saved predictions/probes do not match reconstructed {key}')
    return result, dict(total_answers=total, retained_answers=len(rows) // PREFIX_SAMPLING['per_response'],
                        statuses=dict(statuses), trajectory_hashes=fingerprints)


def paired(meta, left, right, mask, bootstrap=2000):
    n = int(mask.sum())
    if not n:
        return dict(prefixes=0, questions=0, answers=0, sparse=True)
    y, q = meta['y'][mask], meta['question'][mask]
    l, r = (left[mask] - y) ** 2, (right[mask] - y) ** 2
    _, groups = np.unique(q, return_inverse=True)
    losses = np.bincount(groups, weights=l - r) / np.bincount(groups)
    rng = np.random.default_rng(42)
    draws = [rng.choice(losses, len(losses), replace=True).mean() for _ in range(bootstrap)]
    return dict(prefixes=n, questions=len(losses), sparse=len(losses) < 20,
        answers=len(np.unique(np.stack((q, meta['answer'][mask]), axis=1), axis=0)),
        success_rate=float(y.mean()), lstd_brier=float(l.mean()), ridge_brier=float(r.mean()),
        question_mean_difference=float(losses.mean()),
        question_bootstrap_95pct_interval=np.quantile(draws, [.025, .975]).tolist())


def groups(meta):
    masks = {'overall': np.ones(len(meta['y']), dtype=bool),
             'uncapped_only': meta['capped'] == 0, 'capped_only': meta['capped'] == 1}
    for name in ('length', 'position', 'remaining'):
        values = meta[name]
        if name == 'position':
            masks[f'{name}/question_only'] = values == 0
        masks[f'{name}/1_128'] = (values >= 1) & (values <= 128)
        masks[f'{name}/129_512'] = (values >= 129) & (values <= 512)
        masks[f'{name}/513_plus'] = values > 512
    for pos in ('question_only', '1_128', '129_512', '513_plus'):
        for remaining in ('1_128', '129_512', '513_plus'):
            masks[f'position_x_remaining/{pos}/{remaining}'] = (
                masks[f'position/{pos}'] & masks[f'remaining/{remaining}'])
    return masks


def diagnostics(source, questions, data, left, right):
    if (left.shape != right.shape or left.shape != (len(data['y']),)
            or not np.isfinite(left).all() or not np.isfinite(right).all()):
        raise ValueError('Invalid or mismatched prediction arrays')
    meta, provenance = metadata(source, questions, data)
    return dict(groups={name: paired(meta, left, right, mask) for name, mask in groups(meta).items()},
        provenance=provenance,
        scope='Exploratory paired diagnostics, conditional on retained answers. No retuning.',
        bins='Fixed before this analysis: 1-128, 129-512, 513+ tokens; position 0 separate.',
        remaining='T-L: number of remaining actions including the next token; cap counts as termination.',
        uncertainty='Equal-question mean Brier differences within each group; 2000 paired question bootstraps. '
                    'Brier columns pool prefixes. Intervals are descriptive, not multiplicity-adjusted; '
                    'groups with fewer than 20 questions are flagged sparse.',
        caveat='Length is realized future information for analysis ONLY. Groups differ in questions, difficulty '
               'and survival; these are not causal effects of reasoning length.')


def report_text(report):
    lines = ['Paired LSTD minus ridge by length and position (negative favors LSTD)',
             'group                               questions prefixes  LSTD     ridge    difference   95% interval']
    for name, row in report['groups'].items():
        if not row['prefixes']:
            lines.append(f'{name}: no examples')
            continue
        lo, hi = row['question_bootstrap_95pct_interval']
        lines.append(f"{name:38} {row['questions']:4} {row['prefixes']:7}  {row['lstd_brier']:.5f}  "
                     f"{row['ridge_brier']:.5f}  {row['question_mean_difference']:+.5f}  "
                     f"[{lo:+.5f}, {hi:+.5f}]" + (' SPARSE' if row['sparse'] else ''))
    lines += [report['uncertainty'], report['caveat']]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fit', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    fit, out = args.fit.resolve(), args.out.resolve()
    manifest = json.loads((fit / 'manifest.json').read_text())
    summary = json.loads((fit / 'summary.json').read_text())
    source = Path(manifest['source'])
    if source == out or fit == out or source in out.parents or fit in out.parents:
        raise ValueError('Choose a separate diagnostic output directory')
    for name in ('questions.json', 'test.pt'):
        if sha256(source / name) != manifest['source_hashes'][name]:
            raise ValueError('Source data changed since fitting')
    chosen = summary['selected_lstd']
    data = torch.load(source / 'test.pt', weights_only=True, map_location='cpu')
    questions = json.loads((source / 'questions.json').read_text())['questions']
    paths = [fit / f'{chosen}-predictions.npy', fit / 'ridge-predictions.npy']
    result = diagnostics(source, questions, data, *(np.load(p) for p in paths))
    result['selected_lstd'] = chosen
    result['files'] = {str(p): sha256(p) for p in [fit / 'manifest.json', fit / 'summary.json', *paths]}
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'summary.json', result)
    (out / 'report.txt').write_text(report_text(result))
    print(report_text(result))


if __name__ == '__main__':
    main()
