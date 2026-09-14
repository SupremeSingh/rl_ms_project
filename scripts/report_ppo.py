"""Check PPO/GAE or GRPO execution evidence, not improved accuracy."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from math_rl.ppo_checks import checkpoint_complete
from report_resume import metrics


def report(run):
    record = json.loads((run / "run.json").read_text())
    config = record["config"]
    algorithm = config["algorithm"]["adv_estimator"]
    if algorithm not in ("gae", "grpo"):
        raise ValueError("This report supports PPO/GAE and GRPO")
    use_critic = algorithm == "gae"
    samples = config.get("actor_rollout_ref", {}).get("rollout", {}).get("n", 1)
    steps = config["trainer"]["total_training_steps"]
    world_size = config["trainer"]["n_gpus_per_node"] * config["trainer"]["nnodes"]
    logged = metrics(run)
    required = ("actor/grad_norm", "actor/pg_loss")
    if use_critic:
        required += ("critic/grad_norm", "critic/vf_loss")
    finite = bool(logged) and all(all(k in row and math.isfinite(row[k]) for k in required)
                                  for row in logged.values())
    rewards, count = [], 0
    mixed_group_update = False
    complete_groups = True
    for path in sorted((run / "rollouts").glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        count += len(rows)
        rewards.extend(row["score"] for row in rows)
        if not use_critic:
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["input"]].append(row["score"])
            complete_groups &= len(grouped) == config["data"]["train_batch_size"] and all(
                len(scores) == samples for scores in grouped.values())
            mixed = any(len(set(scores)) > 1 for scores in grouped.values())
            grad = logged.get(int(path.stem), {}).get("actor/grad_norm", math.nan)
            mixed_group_update |= mixed and math.isfinite(grad) and grad > 0
    all_logs = "\n".join(p.read_text() for p in (run / "attempts").glob("*/console.log"))
    attempts = sorted((run / "attempts").glob("*"))
    exit_path = attempts[-1] / "exit.json" if attempts else run / "missing-exit.json"
    clean_exit = exit_path.is_file() and json.loads(exit_path.read_text()).get("returncode") == 0
    gates = dict(
        process_exited_successfully=clean_exit,
        all_steps_logged=set(logged) == set(range(1, steps + 1)),
        finite_training_metrics=finite,
        no_nonfinite_warning="grad_norm is not finite" not in all_logs and all(
            math.isfinite(value) for row in logged.values() for value in row.values()),
        actor_updated=any(math.isfinite(r.get("actor/grad_norm", math.nan)) and r["actor/grad_norm"] > 0 for r in logged.values()),
        all_rollouts_saved=count == steps * config["data"]["train_batch_size"] * samples,
        informative_rewards=set(rewards) == {0, 1},
        final_checkpoint_complete=checkpoint_complete(run / f"global_step_{steps}", world_size,
                                                       ("actor", "critic") if use_critic else ("actor",)),
    )
    if use_critic:
        gates["critic_updated"] = any(math.isfinite(r.get("critic/grad_norm", math.nan)) and r["critic/grad_norm"] > 0 for r in logged.values())
    else:
        gates["complete_response_groups"] = samples >= 2 and complete_groups
        gates["mixed_group_actor_update"] = mixed_group_update
    diagnostics = [{"step": step, **{k: v if math.isfinite(v) else None for k, v in row.items()
                                    if k in required or "rollout_probs_diff" in k or "response_length" in k}}
                   for step, row in sorted(logged.items())]
    return dict(algorithm=algorithm, execution_check_pass=all(gates.values()), gates=gates,
                responses=count, reward_rate=sum(rewards) / count if count else None,
                metrics=diagnostics, stage1_status=record["provenance"].get("stage1_status"),
                scope="Pilot execution only. GPU resume equivalence and sustained learning remain separate checks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    result = report(args.run)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    (args.run / "ppo-report.json").write_text(text)
    print(text)
    raise SystemExit(0 if result["execution_check_pass"] else 1)


if __name__ == "__main__":
    main()
