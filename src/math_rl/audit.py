"""CPU-only Stage 1 metrics. Missing human labels can never pass the gate."""
from collections import defaultdict


def summarize(records, labels, min_format=0.95, max_truncation=0.05):
    if not records:
        raise ValueError("No responses")
    by_id = {x["id"]: x for x in records}
    humans = {x["id"]: x for x in labels}
    if len(by_id) != len(records) or len(humans) != len(labels):
        raise ValueError("Duplicate response or label IDs")
    if not humans.keys() <= by_id.keys():
        raise ValueError("Unknown label IDs")
    for label in labels:
        for key in ("human_reward", "human_format", "human_answer_correct"):
            if label[key] not in (0, 1):
                raise ValueError("Labels must be binary")
        if label["human_reward"] != label["human_format"] * label["human_answer_correct"]:
            raise ValueError("Inconsistent human labels")
    groups = defaultdict(list)
    for row in records:
        if row["reward"] not in (0, 1):
            raise ValueError("Rewards must be binary")
        groups[row["prompt_id"]].append(row["reward"])
    if any(len(values) != 2 for values in groups.values()):
        raise ValueError("Every prompt must have exactly two responses")
    n, reviewed = len(records), len(labels)
    agreement = sum(by_id[x["id"]]["reward"] == x["human_reward"] for x in labels) / reviewed if reviewed else None
    fmt = sum(x["human_format"] for x in labels) / reviewed if reviewed else None
    human_accuracy = sum(x["human_answer_correct"] for x in labels) / reviewed if reviewed else None
    accuracy = sum(x["reward"] for x in records) / n
    mixed = sum(a != b for a, b in groups.values()) / len(groups)
    truncation = sum(x["finish_reason"] == "length" for x in records) / n
    gates = {
        "manual_review_complete": reviewed == n and reviewed >= 200,
        "verifier_agreement": agreement is not None and agreement >= 0.99,
        "reward_accuracy_5_to_70_percent": 0.05 <= accuracy <= 0.70,
        "mixed_pairs_at_least_10_percent": mixed >= 0.10,
        "format_rate": fmt is not None and fmt >= min_format,
        "truncation_rate": truncation <= max_truncation,
    }
    return {
        "responses": n, "reviewed": reviewed, "prompt_pairs": len(groups),
        "verifier_agreement": agreement, "reward_accuracy": accuracy,
        "human_answer_accuracy": human_accuracy, "human_format_rate": fmt,
        "mixed_pair_rate": mixed, "truncation_rate": truncation,
        "provisional_thresholds": {"min_format": min_format, "max_truncation": max_truncation},
        "gates": gates, "stage1_pass": all(gates.values()),
        "disagreement_ids": [x["id"] for x in labels if x["human_reward"] != by_id[x["id"]]["reward"]],
    }
