"""Compare a four-step control with a two-step + resumed four-step run."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from math_rl.train import comparable_config


def metrics(run, paths=None):
    result = {}
    for path in paths if paths is not None else sorted((run / "attempts").glob("*/console.log")):
        for line in path.read_text().splitlines():
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            step = re.search(r"\bstep:(\d+)", line)
            if step:
                fields = dict(re.findall(r"([\w./-]+):\s*([-+\deE.]+|nan|inf)(?=\s|$)", line))
                if "actor/grad_norm" in fields:
                    result[int(step[1])] = {k: float(v) for k, v in fields.items()}
    return result


def checkpoint_complete(run, step):
    path = run / f"global_step_{step}"
    return (path / "data.pt").is_file() and all(
        list((path / "actor").glob(pattern))
        for pattern in ("model*.pt", "optim*.pt", "extra*.pt"))


def compare(control, resumed, atol=0.01, rtol=0.1):
    if atol < 0 or rtol < 0 or not math.isfinite(atol + rtol):
        raise ValueError("Tolerances must be finite and nonnegative")
    manifests = [json.loads((run / "run.json").read_text()) for run in (control, resumed)]
    gates = {
        "same_config": comparable_config(manifests[0]["config"]) == comparable_config(manifests[1]["config"]),
        "same_sources": manifests[0]["provenance"]["files"] == manifests[1]["provenance"]["files"],
        "checkpoints_present": all(checkpoint_complete(run, step) for run in (control, resumed) for step in (2, 4)),
    }
    gates["reward_only_smoke_objective"] = all(
        not m["config"].get("algorithm", {}).get("use_kl_in_reward", False) and
        not m["config"]["actor_rollout_ref"]["actor"].get("use_kl_loss", False) and
        m["config"]["actor_rollout_ref"]["actor"].get("entropy_coeff", 0) == 0 and
        m["config"]["actor_rollout_ref"]["actor"]["optim"].get("weight_decay", 0) == 0
        for m in manifests)
    attempt_paths = sorted((resumed / "attempts").glob("*/run.json"))
    attempts = [json.loads(p.read_text()) for p in attempt_paths]
    gates["explicit_resume_from_step_2"] = any(
        a["config"]["trainer"].get("resume_mode") == "resume_path" and
        Path(a["config"]["trainer"].get("resume_from_path", "")).name == "global_step_2"
        for a in attempts)
    gates["resumed_attempt_logs_only_steps_3_and_4"] = any(
        a["config"]["trainer"].get("resume_mode") == "resume_path" and
        set(metrics(resumed, [p.parent / "console.log"])) == {3, 4}
        for p, a in zip(attempt_paths, attempts) if (p.parent / "console.log").exists())
    logs = [metrics(run) for run in (control, resumed)]
    gates["four_steps_logged"] = all(set(log) == {1, 2, 3, 4} for log in logs)
    differences, reward_signal = [], [False, False]
    prompt_match = True
    exact_responses = True
    metrics_match = True
    for step in range(1, 5):
        batches = []
        for index, run in enumerate((control, resumed)):
            path = run / "rollouts" / f"{step}.jsonl"
            rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
            batches.append(rows)
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["input"]].append(row["score"])
            mixed = any(len(set(rewards)) > 1 for rewards in grouped.values())
            grad = logs[index].get(step, {}).get("actor/grad_norm", float("nan"))
            reward_signal[index] |= mixed and math.isfinite(grad) and grad > 0
        prompt_match &= bool(batches[0]) and [r["input"] for r in batches[0]] == [r["input"] for r in batches[1]]
        exact_responses &= batches[0] == batches[1]
        means = [sum(r["score"] for r in rows) / len(rows) if rows else float("nan") for rows in batches]
        if not math.isclose(*means, abs_tol=atol, rel_tol=rtol):
            differences.append({"step": step, "metric": "reward_mean", "values": [v if math.isfinite(v) else None for v in means]})
        for key in ("actor/grad_norm", "actor/pg_loss"):
            values = [log.get(step, {}).get(key, float("nan")) for log in logs]
            good = all(math.isfinite(v) for v in values) and math.isclose(*values, abs_tol=atol, rel_tol=rtol)
            metrics_match &= good
            if not good:
                differences.append({"step": step, "metric": key, "values": [v if math.isfinite(v) else None for v in values]})
    gates.update(prompt_order=prompt_match, reward_driven_update=all(reward_signal),
                 rewards_within_tolerance=not any(d["metric"] == "reward_mean" for d in differences),
                 learning_metrics_within_tolerance=metrics_match)
    return {"gates": gates, "resume_check_pass": all(gates.values()), "differences": differences,
            "exact_rollouts_match": exact_responses, "atol": atol, "rtol": rtol,
            "scope": "Behavioral smoke check; file presence does not prove bitwise RNG or optimizer-state equivalence"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--rtol", type=float, default=0.1)
    args = parser.parse_args()
    report = compare(args.control, args.resumed, args.atol, args.rtol)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.resumed / "resume_report.json").write_text(payload)
    print(payload)
    raise SystemExit(0 if report["resume_check_pass"] else 1)


if __name__ == "__main__":
    main()
