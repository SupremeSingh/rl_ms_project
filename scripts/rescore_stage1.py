"""CPU-only diagnostic; preserve original responses, scores, and labels."""
import argparse
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")


def number(text):
    text = str(text).strip()
    return Decimal(text.replace(",", "")) if NUMBER.fullmatch(text) else None


def extract(text):
    # Never select a candidate using the reference answer.
    if text.count(r"\boxed") != 1:
        return None
    match = re.search(r"\\boxed\s*\{([^{}]*)\}", text)
    return number(match.group(1)) if match else None


def rescore(run, extractor=extract):
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
        if target is None or row["reward"] not in (0, 1):
            raise ValueError(f"Invalid reference/reward: {row['id']}")
        predicted = extractor(row["response"])
        reward = int(predicted is not None and predicted == target)
        groups.setdefault(row["prompt_id"], []).append(reward)
        details.append({"id": row["id"], "original_reward": row["reward"],
                        "extracted": str(predicted) if predicted is not None else None,
                        "diagnostic_reward": reward})
    if any(len(pair) != 2 for pair in groups.values()):
        raise ValueError("Expected two responses per prompt")
    return {"condition": run.name, "responses_sha256": digest,
            "responses": len(rows),
            "original_reward_rate": sum(r["reward"] for r in rows) / len(rows),
            "extraction_rate": sum(r["extracted"] is not None for r in details) / len(rows),
            "diagnostic_reward_rate": sum(r["diagnostic_reward"] for r in details) / len(rows),
            "mixed_pair_rate": sum(a != b for a, b in groups.values()) / len(groups),
            "truncation_rate": sum(r["finish_reason"] == "length" for r in rows) / len(rows),
            "stage1_pass": False, "results": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--rule", choices=["numeric-box", "final-numeric-v1"], default="numeric-box")
    args = parser.parse_args()
    extractor = extract
    if args.rule == "final-numeric-v1":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from math_rl.final_answer import extract_final_number
        extractor = extract_final_number
    runs = ([args.root] if (args.root / "responses.jsonl").exists() else
            sorted(p.parent for p in args.root.glob("*/responses.jsonl")))
    if not runs:
        parser.error("No saved responses found")
    output = args.root / f"{args.rule}-rescore.json"
    if output.exists():
        parser.error(f"Preserving existing report: {output}")
    report = {"rule_version": args.rule, "rule": "Single numeric box" if args.rule == "numeric-box" else "Conservative explicit final numeric answer; matching boxes/statements permitted",
              "purpose": "Diagnostic only. Original labels do not apply to this rule.",
              "limitations": "Does not detect unrelated, quoted, or semantically contradicted answers. Human audit required.",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "conditions": [rescore(run, extractor) for run in runs]}
    if args.rule == "final-numeric-v1":
        report["extractor_sha256"] = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "src/math_rl/final_answer.py").read_bytes()).hexdigest()
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    for condition in report["conditions"]:
        print(json.dumps({k: v for k, v in condition.items() if k != "results"}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
