"""Blind V2 final-answer review; standard library only."""
import argparse
import hashlib
import json
from pathlib import Path


def load_review(run, rule='final-numeric-v2'):
    payload = (run / 'responses.jsonl').read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if json.loads((run / 'metadata.json').read_text())['responses_sha256'] != digest:
        raise ValueError('Responses changed since generation')
    report = json.loads((run / f'{rule}-rescore.json').read_text())
    conditions = report['conditions']
    if report['rule_version'] != rule or len(conditions) != 1 or conditions[0]['responses_sha256'] != digest:
        raise ValueError('Expected matching single-condition report')
    rows = [json.loads(line) for line in payload.decode().splitlines()]
    predictions = {r['id']: r for r in conditions[0]['results']}
    ids = {r['id'] for r in rows}
    if not rows or len(ids) != len(rows) or ids != predictions.keys() or len(predictions) != len(conditions[0]['results']):
        raise ValueError('Duplicate or mismatched response IDs')
    path = run / 'final-numeric-v2-human.jsonl'
    labels = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if len({r['id'] for r in labels}) != len(labels):
        raise ValueError('Duplicate labels')
    for label in labels:
        if label['id'] not in ids or label['responses_sha256'] != digest:
            raise ValueError('Labels do not match responses')
        if (label['unambiguous'], label['correct']) not in ((0, 0), (1, 0), (1, 1)):
            raise ValueError('Invalid human labels')
    return rows, predictions, labels, digest, path


def summarize(predictions, labels, total):
    disagreements = [dict(x, predicted_reward=predictions[x['id']]['diagnostic_reward'], extracted=predictions[x['id']]['extracted']) for x in labels if predictions[x['id']]['diagnostic_reward'] != x['correct']]
    return dict(responses=total, reviewed=len(labels), review_complete=len(labels) == total,
                verifier_agreement=1-len(disagreements)/len(labels) if labels else None,
                human_correct_rate=sum(x['correct'] for x in labels)/len(labels) if labels else None,
                false_accept_ids=[x['id'] for x in disagreements if x['predicted_reward'] == 1],
                false_reject_ids=[x['id'] for x in disagreements if x['predicted_reward'] == 0],
                disagreements=disagreements, stage1_pass=False,
                note='Development review only; a separate fresh 200-response audit is required.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--rule', choices=['final-numeric-v2', 'final-numeric-v3'], default='final-numeric-v2')
    args = parser.parse_args()
    rows, predictions, labels, digest, path = load_review(args.run, args.rule)
    if not args.report:
        done = {x['id'] for x in labels}
        print('Ignore boxed formatting. Judge the final answer to the original question.')
        print('Use 1 1 for clear and correct; 1 0 for clear but wrong; 0 0 for missing or ambiguous.')
        print('Explicit corrections may supersede earlier working. Wrong reasoning alone is not failure.')
        print('Unrelated conclusions or unresolved competing answers are not correct; note difficult cases.')
        for index, row in enumerate(rows, 1):
            if row['id'] in done:
                continue
            print(f"\n--- {index}/{len(rows)}: {row['id']} ---")
            print('QUESTION:', row['prompt'])
            print('REFERENCE:', row['ground_truth'])
            print('RESPONSE:\n', row['response'])
            while True:
                value = input('Enter unambiguous correct, or q to pause: ').strip()
                if value.lower() == 'q':
                    print('Saved. Resume with the same command; --report shows agreement.')
                    return
                if value in ('0 0', '1 0', '1 1'):
                    break
                print('Use 0 0, 1 0, or 1 1.')
            unambiguous, correct = map(int, value.split())
            label = dict(id=row['id'], unambiguous=unambiguous, correct=correct,
                         notes=input('Notes (Enter to skip): '), responses_sha256=digest)
            with path.open('a') as handle:
                handle.write(json.dumps(label)+'\n')
            labels.append(label)
    result = summarize(predictions, labels, len(rows))
    result['rule_version'] = args.rule
    result['responses_sha256'] = digest
    result['rescore_sha256'] = hashlib.sha256((args.run / f'{args.rule}-rescore.json').read_bytes()).hexdigest()
    (args.run / f'{args.rule}-human-report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
