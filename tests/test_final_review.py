import importlib.util
from pathlib import Path
import hashlib
import json
import pytest

spec = importlib.util.spec_from_file_location('review_final', Path(__file__).parents[1] / 'scripts/review_final_answers.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_false_accepts_and_rejects():
    predictions = {'a': {'diagnostic_reward': 1, 'extracted': '3'},
                   'b': {'diagnostic_reward': 0, 'extracted': None}}
    labels = [{'id': 'a', 'correct': 0}, {'id': 'b', 'correct': 1}]
    result = module.summarize(predictions, labels, 2)
    assert result['false_accept_ids'] == ['a']
    assert result['false_reject_ids'] == ['b']
    assert result['verifier_agreement'] == 0
    assert result['review_complete']
    assert not result['stage1_pass']
    assert module.summarize(predictions, [], 2)['verifier_agreement'] is None


def test_integrity_and_separate_labels(tmp_path):
    payload = json.dumps({'id': 'a'}).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / 'responses.jsonl').write_bytes(payload)
    (tmp_path / 'metadata.json').write_text(json.dumps({'responses_sha256': digest}))
    (tmp_path / 'labels.jsonl').write_text('old labels untouched')
    (tmp_path / 'math-verify-v1-rescore.json').write_text(json.dumps({
        'rule_version': 'math-verify-v1', 'conditions': [{'responses_sha256': digest,
        'results': [{'id': 'a', 'diagnostic_reward': 1, 'extracted': '3'}]}]}))
    assert module.load_review(tmp_path)[2] == []
    assert (tmp_path / 'labels.jsonl').read_text() == 'old labels untouched'
    (tmp_path / 'responses.jsonl').write_bytes(payload + b' ')
    with pytest.raises(ValueError, match='changed'):
        module.load_review(tmp_path)


def test_v3_reuses_existing_labels(tmp_path):
    payload = json.dumps({'id': 'a'}).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / 'responses.jsonl').write_bytes(payload)
    (tmp_path / 'metadata.json').write_text(json.dumps({'responses_sha256': digest}))
    label_path = tmp_path / 'final-numeric-v2-human.jsonl'
    original = json.dumps(dict(id='a', unambiguous=1, correct=1, responses_sha256=digest))
    label_path.write_text(original)
    (tmp_path / 'math-verify-v1-rescore.json').write_text(json.dumps({
        'rule_version': 'math-verify-v1', 'conditions': [{'responses_sha256': digest,
        'results': [{'id': 'a', 'diagnostic_reward': 1, 'extracted': '3'}]}]}))
    rows, predictions, labels, _, path = module.load_review(tmp_path, 'math-verify-v1')
    assert path == label_path
    assert label_path.read_text() == original
    assert module.summarize(predictions, labels, len(rows))['verifier_agreement'] == 1
