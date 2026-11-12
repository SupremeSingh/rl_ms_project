"""Read-only cost breakdown from completed value-repeat runs; standard library only."""
import argparse
import json
import re
from pathlib import Path
from statistics import mean

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
METRIC = re.compile(r"([A-Za-z_][A-Za-z_0-9./-]*):\s*(" + NUMBER + r")")
STEP = re.compile(r"\bstep:(\d+)\s+-")
PHASES = ("gen", "old_log_prob", "adv", "update_actor", "step")


def read(path):
    return json.loads(path.read_text())


def logged_steps(path):
    records = {}
    if path.exists():
        with path.open(errors="replace") as handle:
            for line in handle:
                match = STEP.search(line)
                if not match:
                    continue
                step = int(match[1])
                metrics = {k: float(v) for k, v in METRIC.findall(line)}
                if step > 0 and "timing_s/step" in metrics:
                    if step in records and records[step] != metrics:
                        raise ValueError(f"Conflicting duplicate step {step}: {path}")
                    records[step] = metrics
    return records


def inspect(root, seed, arm, steps):
    child = root / f"seed{seed}"
    name = f"{arm}-seed{seed}"
    case = child / name
    result = read(case / "result.json")
    if result["updates"] != steps:
        raise ValueError(f"Incomplete run: {case}")
    records = logged_steps(child / f"{name}.log")
    rows = list(records.values())
    full = set(records) == set(range(1, steps + 1))
    measured = {key: sum(r[f"timing_s/{key}"] for r in rows)
                if full and all(f"timing_s/{key}" in r for r in rows) else None
                for key in PHASES}
    lengths = [r["response_length/mean"] for r in rows if "response_length/mean" in r]
    fit_path = case / "linear-critic/metrics.jsonl"
    fit = [json.loads(line) for line in fit_path.read_text().splitlines()] if fit_path.exists() else []
    fit_full = [r["step"] for r in fit] == list(range(1, steps + 1))
    chunks_path = case / "chunk-metrics.jsonl"
    chunks = [json.loads(line) for line in chunks_path.read_text().splitlines()] if chunks_path.exists() else []
    seconds = result["training_seconds"]
    print(f"\n{arm} seed {seed}: train={seconds/3600:.3f} wall h; "
          f"{result['training_gpu_hours']:.3f} GPUh; log steps={len(rows)}/{steps}")
    print("  Mean response tokens: " + (f"{mean(lengths):.1f}" if full and len(lengths)==steps else "unavailable"))
    for key, value in measured.items():
        print(f"  {key:<15}: " + (f"{value/60:8.2f} wall min" if value is not None else "unavailable/incomplete log"))
    if measured["step"] is not None:
        print(f"  Outside step timers: {(seconds-measured['step'])/60:.2f} wall min")
    fit_seconds = None
    if fit_full:
        fit_seconds = sum(r["fit_seconds"] for r in fit)
        print(f"  Ridge fit        : {fit_seconds/60:.2f} wall min (INSIDE adv, not additional)")
        print(f"  Fitted states    : {sum(r['transitions'] for r in fit):,}")
    if chunks and [r["step"] for r in chunks] == list(range(1, steps+1)):
        print(f"  Chunked tokens   : {sum(r['response_tokens'] for r in chunks):,}")
        print(f"  Chunk states     : {sum(r['chunks'] for r in chunks):,}")
    return dict(seconds=seconds, phases=measured,
                length=mean(lengths) if full and len(lengths)==steps else None,
                fit=fit_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("outputs/numina-value-repeat-v1"))
    args = parser.parse_args()
    manifest = read(args.study / "manifest.json")
    seeds, steps = manifest["seeds"], manifest["steps"]
    for representation in ("slice", "rms-layer"):
        by_mode = {}
        for mode in ("token", "chunk"):
            by_mode[mode] = [inspect(args.study, seed, f"{representation}-{mode}", steps) for seed in seeds]
        print(f"\n{representation.upper()}: paired chunk minus token, mean across {len(seeds)} seeds")
        token_mean = mean(r["seconds"] for r in by_mode["token"])
        delta = mean(c["seconds"]-t["seconds"] for t,c in zip(by_mode["token"],by_mode["chunk"]))
        print(f"  Total training: {delta/60:+.2f} wall min ({100*delta/token_mean:+.2f}%)")
        for key in PHASES:
            pairs = [(t["phases"][key],c["phases"][key]) for t,c in zip(by_mode["token"],by_mode["chunk"])]
            if all(t is not None and c is not None for t,c in pairs):
                print(f"  {key:<15}: {mean(c-t for t,c in pairs)/60:+.2f} wall min")
    print("\nTimers may nest: do NOT add gen to generate_sequences or fit to adv.")
    print("old_log_prob includes actor log probabilities, frozen feature extraction and transport.")
    print("adv includes Ridge accumulation/solving and credit assignment; segmentation occurs in workers and driver.")
    print("Token counts help explain work, but do not remove hardware/load confounding.")
    print("Training GPUh excludes separate MC evaluation; this script reads files only.")


if __name__ == "__main__":
    main()
