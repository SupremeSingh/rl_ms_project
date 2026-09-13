from math_rl.audit import final_answer_gates


def inputs():
    metadata = dict(mode='audit', audit_offset=160,
                    excluded_development_ids=[str(i) for i in range(160)],
                    prompt_ids=[str(i) for i in range(160, 260)])
    scoring = dict(responses=200, status_counts={'correct': 100, 'incorrect': 100},
                   diagnostic_reward_rate=0.5, mixed_pair_rate=0.3, truncation_rate=0)
    return metadata, scoring


def test_complete_fresh_audit():
    meta, scoring = inputs()
    assert final_answer_gates(meta, scoring, 200, 0.99)['stage1_pass']
    assert not final_answer_gates(meta, scoring, 199, 1)['stage1_pass']
    assert not final_answer_gates(meta, scoring, 200, 0.985)['stage1_pass']


def test_easy_dataset_is_not_verifier_failure():
    meta, scoring = inputs()
    scoring['diagnostic_reward_rate'] = 0.86
    result = final_answer_gates(meta, scoring, 200, 1)
    assert result['verifier_audit_pass']
    assert not result['stage1_pass']


def test_overlap_diagnostic_and_errors_fail():
    meta, scoring = inputs()
    meta['prompt_ids'][0] = '0'
    assert not final_answer_gates(meta, scoring, 200, 1)['verifier_audit_pass']
    meta, scoring = inputs()
    meta['mode'] = 'diagnostic'
    assert not final_answer_gates(meta, scoring, 200, 1)['verifier_audit_pass']
    meta, scoring = inputs()
    scoring['status_counts']['verify_timeout'] = 1
    assert not final_answer_gates(meta, scoring, 200, 1)['verifier_audit_pass']
