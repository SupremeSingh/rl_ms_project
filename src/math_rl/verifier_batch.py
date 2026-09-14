"""JSON batch in/out for the isolated, pinned Math-Verify interpreter."""
from contextlib import redirect_stdout
import json
import sys

from math_rl.math_verify_reward import grade, provenance


def score_batch(rows):
    provenance()
    scores = []
    for row in rows:
        result = grade(row["response"], row["ground_truth"])
        if result["status"] not in {"correct", "incorrect", "parse_failure", "unsupported_parse"}:
            raise RuntimeError(f"Verifier failed: {result}")
        scores.append(dict(score=result["diagnostic_reward"],
                           verifier_status=result["status"],
                           extracted=result["extracted"] or ""))
    return scores


def main():
    if sys.argv[1:] == ["--provenance"]:
        print(json.dumps(provenance()))
        return
    rows = json.load(sys.stdin)
    # Parser diagnostics cannot corrupt the JSON protocol.
    with redirect_stdout(sys.stderr):
        scores = score_batch(rows)
    json.dump(scores, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
