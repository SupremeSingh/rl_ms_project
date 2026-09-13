"""CPU-only diagnostic; preserve original responses, scores, and labels."""
import argparse
import hashlib
import json
from pathlib import Path


def number(text):
    from math_rl.reward import parse_number
    return parse_number(text)


def rescore(run, grader):
    payload = (run / "responses.jsonl").read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    metadata = json.loads((run / "metadata.json").read_text())
    if digest != metadata["responses_sha256"]:
        raise ValueError(f"Responses changed: {run}")
    rows = [json.loads(line) for line in payload.decode().splitlines()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Empty responses or duplicate IDs")
    groups, details = {}, []
    for row in rows:
        target = number(row["ground_truth"])
        if target is None or row.get("reward") not in (None, 0, 1):
            raise ValueError(f"Invalid reference/reward: {row['id']}")
        assessment = grader(row["response"], row["ground_truth"])
        groups.setdefault(row["prompt_id"], []).append(assessment["diagnostic_reward"])
        details.append({"id": row["id"], "original_reward": row.get("reward"), **assessment})
    if any(len(pair) != 2 for pair in groups.values()):
        raise ValueError("Expected two responses per prompt")
    return {"condition": run.name, "responses_sha256": digest,
            "responses": len(rows),
            "original_reward_rate": sum(r["reward"] for r in rows) / len(rows) if all(r.get("reward") is not None for r in rows) else None,
            "extraction_rate": sum(r["extracted"] is not None for r in details) / len(rows),
            "diagnostic_reward_rate": sum(r["diagnostic_reward"] for r in details) / len(rows),
            "mixed_pair_rate": sum(a != b for a, b in groups.values()) / len(groups),
            "truncation_rate": sum(r["finish_reason"] == "length" for r in rows) / len(rows),
            "status_counts": {status: sum(r.get("status") == status for r in details)
                              for status in sorted({r["status"] for r in details if "status" in r})},
            "stage1_pass": False, "results": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--rule", choices=["math-verify-v1"], default="math-verify-v1")
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from math_rl.math_verify_reward import grade, provenance
    verifier_metadata = provenance()
    runs = ([args.root] if (args.root / "responses.jsonl").exists() else
            sorted(p.parent for p in args.root.glob("*/responses.jsonl")))
    if not runs:
        parser.error("No saved responses found")
    output = args.root / f"{args.rule}-rescore.json"
    if output.exists():
        parser.error(f"Preserving existing report: {output}")
    report = {
        "rule_version": args.rule,
        "rule": "Pinned Math-Verify parse then verify, boxed-priority first match",
        "purpose": "Existing format-independent correctness labels remain valid",
        "limitations": "Extraction does not establish relevance or semantic consistency; human audit required",
        "verifier": verifier_metadata,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "extractor_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "src/math_rl/math_verify_reward.py").read_bytes()).hexdigest(),
        "conditions": [rescore(run, grade) for run in runs],
    }
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    for condition in report["conditions"]:
        print(json.dumps({k: v for k, v in condition.items() if k != "results"}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
