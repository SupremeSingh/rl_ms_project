import pytest
from math_rl.audit import summarize


def fixture_data():
    records, labels = [], []
    for pair in range(100):
        for sample in range(2):
            rid = f"{pair}/{sample}"
            records.append({"id": rid, "prompt_id": str(pair),
                            "reward": sample, "finish_reason": "stop"})
            labels.append({"id": rid, "human_reward": sample,
                           "human_format": 1, "human_answer_correct": sample})
    return records, labels


def test_complete_audit_passes():
    records, labels = fixture_data()
    result = summarize(records, labels)
    assert result["stage1_pass"]
    assert result["mixed_pair_rate"] == 1
    assert result["reward_accuracy"] == 0.5


def test_missing_human_labels_cannot_pass():
    records, labels = fixture_data()
    assert not summarize(records, [])['stage1_pass']
    assert not summarize(records, labels[:-1])['stage1_pass']


def test_three_disagreements_fail_99_percent_gate():
    records, labels = fixture_data()
    for label in labels[:3]:
        label['human_reward'] = 1 - label['human_reward']
        label['human_answer_correct'] = label['human_reward']
    assert not summarize(records, labels)['gates']['verifier_agreement']


def test_unpaired_data_rejected():
    records, labels = fixture_data()
    with pytest.raises(ValueError):
        summarize(records[:-1], labels[:-1])


def test_all_equal_rewards_fail_mixed_pair_gate():
    records, _ = fixture_data()
    for row in records:
        row['reward'] = 0
    assert not summarize(records, [])['gates']['mixed_pairs_at_least_10_percent']
