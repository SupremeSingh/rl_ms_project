"""Generate base-policy data, cache causal features, then compare supervised value heads."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from math_rl.provenance import sha256, write_json
from math_rl.prompts import validate_assets


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def save_tensor(path, value):
    import torch
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def select_data(counts):
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    from evaluate_pilots import normalize, select_questions

    assets = json.loads((ROOT / "configs/assets.json").read_text())
    validate_assets(assets)
    source = load_dataset("openai/gsm8k", "main", split="train", revision=assets["dataset_revision"])
    source = source.add_column("original_index", list(range(len(source)))).shuffle(seed=assets["seed"])
    excluded = {f"gsm8k/train/{source[i]['original_index']}" for i in range(min(1024, len(source)))}
    wording = set()
    for split in ("train", "val"):
        for row in Dataset.from_parquet(str(ROOT / f"data/gsm8k/{split}.parquet")):
            excluded.add(row["extra_info"]["prompt_id"])
            wording.add(normalize(row["prompt"][1]["content"]))
    for path in (ROOT / "outputs").rglob("metadata.json"):
        metadata = json.loads(path.read_text())
        if metadata.get("mode") in ("audit", "diagnostic"):
            for line in (path.parent / "responses.jsonl").read_text().splitlines():
                row = json.loads(line)
                excluded.add(row["prompt_id"] if "prompt_id" in row else row["id"].rsplit("/", 1)[0])
    # Preserve every previously frozen evaluation set, including the 500-question run.
    for path in list((ROOT / "data/gsm8k").glob("eval-*.json")) + list((ROOT / "outputs").glob("evaluation-*/questions.json")):
        for row in json.loads(path.read_text())["questions"]:
            excluded.add(row["id"])
            wording.add(normalize(row["question"]))
    wording.update(normalize(row["question"]) for row in source
                   if f"gsm8k/train/{row['original_index']}" in excluded)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models/qwen-math", local_files_only=True)
    rows = select_questions(source, tokenizer, sum(counts), excluded, wording)
    start = 0
    for split, count in zip(("train", "val", "test"), counts):
        for row in rows[start:start + count]:
            row["split"] = split
        start += count
    return dict(questions=rows, excluded_ids=sorted(excluded))


def shard_path(out, index, kind):
    return out / kind / f"{index:05d}.{'json' if kind == 'trajectories' else 'pt'}"


def generate(out, config, questions):
    from vllm import LLM, SamplingParams
    from math_rl.ppo_reward import compute_score
    if all(shard_path(out, i, "trajectories").exists() for i in range(len(questions))):
        return
    engine = LLM(model=str(ROOT / "models/qwen-math"), dtype="bfloat16", tensor_parallel_size=1,
                 max_model_len=2560, gpu_memory_utilization=.6, max_num_seqs=16,
                 max_num_batched_tokens=2560, enforce_eager=True, seed=42, generation_config="vllm")
    for i, q in enumerate(questions):
        path = shard_path(out, i, "trajectories")
        if path.exists():
            continue
        params = SamplingParams(n=config["responses"], temperature=1., top_p=1., top_k=-1,
                                max_tokens=2048, seed=42 + i)
        generated = engine.generate([{"prompt_token_ids": q["prompt_token_ids"]}], params)[0].outputs
        if len(generated) != config["responses"] or any(not r.token_ids for r in generated):
            raise RuntimeError("Incomplete generation")
        scores = compute_score(["gsm8k"] * len(generated), [r.text for r in generated],
                               [q["ground_truth"]] * len(generated))
        rows = [dict(response=r.text, token_ids=list(r.token_ids), finish_reason=r.finish_reason,
                     stop_reason=r.stop_reason, **score) for r, score in zip(generated, scores, strict=True)]
        atomic_json(path, dict(question_id=q["id"], responses=rows))
        print(f"Generated/scored question {i + 1}/{len(questions)}", flush=True)


def extract(out, config, questions):
    import torch
    from transformers import AutoModel
    from math_rl.critic_probe import prefix_states
    if all(shard_path(out, i, "features").exists() for i in range(len(questions))):
        return
    # AutoModel omits the vocabulary head; last_hidden_state includes Qwen's final RMSNorm.
    model = AutoModel.from_pretrained(ROOT / "models/qwen-math", local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    model.requires_grad_(False)
    for i, q in enumerate(questions):
        target = shard_path(out, i, "features")
        source = shard_path(out, i, "trajectories")
        if target.exists():
            continue
        trajectories = json.loads(source.read_text())
        if trajectories["question_id"] != q["id"] or len(trajectories["responses"]) != config["responses"]:
            raise ValueError("Trajectory manifest mismatch")
        features, offsets, rewards = [], [0], []
        for row in trajectories["responses"]:
            ids = torch.tensor([q["prompt_token_ids"] + row["token_ids"]], device="cuda")
            states = prefix_states(model, ids, len(q["prompt_token_ids"]), len(row["token_ids"]))
            features.append(states)
            offsets.append(offsets[-1] + len(states))
            rewards.append(row["score"])
        save_tensor(target, dict(features=torch.cat(features), offsets=torch.tensor(offsets),
                                rewards=torch.tensor(rewards), question_id=q["id"],
                                trajectories_sha256=sha256(source)))
        print(f"Cached features {i + 1}/{len(questions)}", flush=True)


def prepare_probes(out, questions):
    import torch
    from math_rl.critic_probe import sample_shard
    for split in ("train", "val", "test"):
        path = out / f"{split}.pt"
        if path.exists():
            continue
        parts = []
        for i, q in enumerate(questions):
            if q["split"] != split:
                continue
            shard = torch.load(shard_path(out, i, "features"), weights_only=True, map_location="cpu")
            if shard["question_id"] != q["id"] or shard["trajectories_sha256"] != sha256(shard_path(out, i, "trajectories")):
                raise ValueError("Features do not match the saved trajectories")
            parts.append(sample_shard(shard, i))
        save_tensor(path, {k: torch.cat([p[k] for p in parts]) for k in parts[0]})


def fit(out, config, questions):
    import torch
    import numpy as np
    from math_rl.critic_probe import HEADS, assessment, fit_trial, make_head, predict
    if (out / "summary.json").exists() and (out / "report.txt").exists():
        return
    prepare_probes(out, questions)
    train, val = [torch.load(out / f"{s}.pt", weights_only=True) for s in ("train", "val")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean = train["x"].float().mean(0).to(device)
    scale = train["x"].float().std(0, unbiased=False).clamp_min(1e-5).to(device)
    baseline = float(train["y"].mean())
    save_tensor(out / "normalization.pt", dict(mean=mean.cpu(), scale=scale.cpu(), baseline=baseline))
    # Select all heads using validation only. The test tensors are opened afterward.
    for kind in HEADS:
        for seed in config["seeds"]:
            path = out / "heads" / f"{kind}-{seed}.pt"
            if path.exists():
                continue
            trials = [fit_trial(kind, train, val, mean, scale, device, seed, lr, config["epochs"])
                      for lr in (1e-3, 1e-4)]
            winner = min(trials, key=lambda t: t["validation_brier"])
            winner["trials"] = [{k: v for k, v in t.items() if k != "state"} for t in trials]
            save_tensor(path, winner)
            print(f"Selected {kind} seed {seed}: validation Brier {winner['validation_brier']:.5f}", flush=True)
    test = torch.load(out / "test.pt", weights_only=True)
    results = []
    for kind in HEADS:
        for seed in config["seeds"]:
            saved = torch.load(out / "heads" / f"{kind}-{seed}.pt", weights_only=True)
            head = make_head(kind, train["x"].shape[1], saved["width"]).to(device)
            head.load_state_dict(saved["state"])
            predict(head, kind, test, mean, scale, device)  # warmup, excluded from timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = predict(head, kind, test, mean, scale, device)
            elapsed = time.perf_counter() - started
            np.save(out / "heads" / f"{kind}-{seed}-predictions.npy", prediction)
            results.append(dict(kind=kind, seed=seed, chosen_lr=saved["lr"], parameters=saved["parameters"],
                total_tuning_seconds=sum(t["fitting_seconds"] for t in saved["trials"]),
                prediction_seconds=elapsed, prediction_count=len(prediction),
                peak_gpu_bytes=max(t["peak_gpu_bytes"] or 0 for t in saved["trials"]),
                test=assessment(test, prediction, baseline), fitting_curves=saved["trials"]))
    trajectory_count = correct = truncated = 0
    review, rng = [], random.Random(42)
    for i in range(len(questions)):
        for row in json.loads(shard_path(out, i, "trajectories").read_text())["responses"]:
            trajectory_count += 1
            correct += row["score"]
            truncated += row["finish_reason"] == "length"
            slot = rng.randrange(trajectory_count)
            if len(review) < 20:
                review.append((questions[i], row))
            elif slot < 20:
                review[slot] = (questions[i], row)
    with (out / "review.txt").open("w") as handle:
        for question, response in review:
            handle.write(f"\n{question['id']} ({question['split']})\nQUESTION: {question['question']}\n"
                         f"REFERENCE: {question['ground_truth']}\nREWARD: {response['score']}\n"
                         f"RESPONSE: {response['response']}\n")
    summary = dict(scope="Frozen base-policy supervised value prediction; no LSTD or PPO updates.",
        metrics_note="Brier uses raw predictions. CE clips predictions to (0,1); linear-value out-of-range rate is reported.",
        weighting="Equal weight per retained preset prefix; uncertainty clustered by question.",
        budget_note="Same epochs, LR grid and seeds for every head; curves expose fitting-time tradeoffs, not parameter-matched depth effects.",
        trajectories=trajectory_count, reward_rate=correct / trajectory_count,
        truncation_rate=truncated / trajectory_count,
        prefix_counts={s: len(d["y"]) for s, d in (("train", train), ("val", val), ("test", test))},
        constant=assessment(test, np.full(len(test["y"]), baseline), baseline), heads=results)
    lines = ["Frozen base-policy critic comparison (test set)",
             f"Trajectories: {trajectory_count}; fixed-prefix examples: {summary['prefix_counts']}",
             f"Constant baseline Brier: {summary['constant']['brier']:.5f}",
             "head          Brier mean/std       question-only Brier   accuracy   total tuning seconds"]
    for kind in HEADS:
        rows = [r for r in results if r["kind"] == kind]
        scores = [r["test"]["brier"] for r in rows]
        early = np.mean([r["test"]["by_prefix"]["question_only"]["brier"] for r in rows])
        accuracy = np.mean([r["test"]["accuracy"] for r in rows])
        lines.append(f"{kind:13} {np.mean(scores):.5f} / {np.std(scores):.5f}       {early:.5f}              "
                     f"{accuracy:.3f}      {sum(r['total_tuning_seconds'] for r in rows):.1f}")
    lines += ["Lower Brier is better. Std is across head seeds, not independent datasets.",
              "See summary.json for calibration, AUROC, intervals, fitting curves and costs.",
              "Inspect review.txt for verifier errors. No LSTD or PPO learning claim follows from this report."]
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    atomic_json(out / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full", action="store_true", help="4000/500/500 questions, 16 responses; default 128/32/32, 8 responses")
    parser.add_argument("--stage", choices=("generate", "extract", "fit"))
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage:
        manifest = json.loads((out / "manifest.json").read_text())
        questions = json.loads((out / "questions.json").read_text())["questions"]
        {"generate": generate, "extract": extract, "fit": fit}[args.stage](out, manifest["config"], questions)
        return
    import torch
    from math_rl.ppo_reward import verifier_command
    from math_rl.critic_probe import PREFIX_POSITIONS
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Run inside the container with one BF16-capable GPU")
    config = dict(counts=[4000, 500, 500] if args.full else [128, 32, 32], responses=16 if args.full else 8,
                  seeds=[42, 43, 44], epochs=10, prefix_positions=list(PREFIX_POSITIONS),
                  feature="Qwen2Model.last_hidden_state: final RMSNorm, last prefix position, all dimensions; no added budget feature",
                  sampling=dict(temperature=1., top_p=1., top_k=-1, max_tokens=2048, seed="42 + question index"),
                  terminal="EOS or cap; last action gets outcome reward; post-terminal value zero")
    verifier = json.loads(subprocess.check_output(verifier_command() + ["--provenance"], text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src"))))
    tracked = [Path(__file__).resolve(), ROOT / "src/math_rl/critic_probe.py", ROOT / "src/math_rl/prompts.py",
               ROOT / "src/math_rl/ppo_reward.py", ROOT / "src/math_rl/verifier_batch.py", ROOT / "src/math_rl/math_verify_reward.py",
               ROOT / "src/math_rl/reward.py", ROOT / "src/math_rl/provenance.py",
               ROOT / "scripts/evaluate_pilots.py", ROOT / "configs/assets.json"]
    tracked += sorted((ROOT / "models/qwen-math").glob("*.safetensors"))
    tracked += sorted((ROOT / "models/qwen-math").glob("*.json"))
    manifest = dict(config=config, verifier=verifier, files={str(p.relative_to(ROOT)): sha256(p) for p in tracked},
                    versions={p: version(p) for p in ("torch", "transformers", "vllm", "datasets", "numpy")},
                    gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)
    out.mkdir(parents=True, exist_ok=True)
    question_hash = None
    if (out / "manifest.json").exists():
        previous = json.loads((out / "manifest.json").read_text())
        question_hash = previous.pop("questions_sha256", None)
        if previous != manifest:
            raise ValueError("Resume requires unchanged model, code, verifier, runtime and settings")
        if question_hash is not None and (not (out / "questions.json").exists() or sha256(out / "questions.json") != question_hash):
            raise ValueError("Frozen question manifest changed")
    else:
        if any(out.iterdir()):
            raise ValueError("Nonempty directory without a manifest; choose a new output path")
        atomic_json(out / "manifest.json", manifest)
    if not (out / "questions.json").exists() or question_hash is None:
        selected = select_data(config["counts"])
        if (out / "questions.json").exists() and json.loads((out / "questions.json").read_text()) != selected:
            raise ValueError("Unsealed questions differ from deterministic selection")
        atomic_json(out / "questions.json", selected)
    manifest["questions_sha256"] = sha256(out / "questions.json")
    atomic_json(out / "manifest.json", manifest)
    for name in ("trajectories", "features", "heads"):
        (out / name).mkdir(exist_ok=True)
    for stage in ("generate", "extract", "fit"):
        started = time.time()
        atomic_json(out / "status.json", dict(stage=stage, state="running"))
        with (out / f"{stage}.log").open("a") as log:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--out", str(out), "--stage", stage],
                                    stdout=log, stderr=subprocess.STDOUT)
        atomic_json(out / "status.json", dict(stage=stage, state="complete" if result.returncode == 0 else "failed",
                    exit_code=result.returncode, seconds=time.time() - started))
        with (out / "timings.jsonl").open("a") as log:
            log.write(json.dumps(dict(stage=stage, seconds=time.time() - started, exit_code=result.returncode)) + "\n")
        if result.returncode:
            raise RuntimeError(f"{stage} failed; see {out / (stage + '.log')}")
    print(f"Finished: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
