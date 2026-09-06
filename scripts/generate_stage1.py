"""Generate 100 prompt pairs with the untrained initial checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from math_rl.reward import compute_score

    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=root / "outputs/stage1")
    p.add_argument("--prompts", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.prompts < 100:
        p.error("Stage 1 requires at least 100 pairs / 200 responses")
    if args.out.exists():
        p.error("Output directory exists; choose a new --out to preserve the audit")
    assets = json.loads((root / "configs/assets.json").read_text())
    model = root / "models/qwen-math"
    data_path = root / "data/gsm8k/train.parquet"
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    dataset = Dataset.from_parquet(str(data_path))
    eligible = []
    excluded = 0
    for row in dataset:
        ids = tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                                             add_generation_prompt=True)
        if len(ids) > 512:
            excluded += 1
        else:
            eligible.append((row, ids))
    if len(eligible) < args.prompts:
        p.error(f"Only {len(eligible)} eligible prompts; prepare more data")
    chosen = eligible[:args.prompts]
    llm = LLM(model=str(model), dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=1024, gpu_memory_utilization=0.6,
              enforce_eager=True, seed=args.seed, generation_config="vllm")
    params = SamplingParams(n=2, temperature=1.0, top_p=1.0, top_k=-1,
                            max_tokens=512, seed=args.seed)
    outputs = llm.generate([{"prompt_token_ids": ids} for _, ids in chosen], params)
    records = []
    for (row, ids), output in zip(chosen, outputs, strict=True):
        if len(output.outputs) != 2:
            raise RuntimeError("Expected two responses per prompt")
        for completion in output.outputs:
            records.append({
                "id": f"{row['extra_info']['prompt_id']}/{completion.index}",
                "prompt_id": row["extra_info"]["prompt_id"],
                "prompt": row["prompt"][0]["content"],
                "ground_truth": row["reward_model"]["ground_truth"],
                "response": completion.text,
                "reward": compute_score("gsm8k", completion.text,
                                         row["reward_model"]["ground_truth"]),
                "finish_reason": completion.finish_reason,
                "prompt_tokens": len(ids),
                "response_tokens": len(completion.token_ids),
            })
    args.out.mkdir(parents=True)
    payload = "".join(json.dumps(x) + "\n" for x in records)
    (args.out / "responses.jsonl").write_text(payload)
    metadata = {
        "assets": assets, "seed": args.seed, "n": 2,
        "temperature": 1.0, "top_p": 1.0, "top_k": -1,
        "max_tokens": 512, "dtype": "bfloat16",
        "prompt_count": args.prompts, "excluded_overlong": excluded,
        "eligible_count": len(eligible), "source": "GSM8K train subset",
        "responses_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "reward_sha256": hashlib.sha256((root / "src/math_rl/reward.py").read_bytes()).hexdigest(),
        "checkpoint_note": "Initial downloaded checkpoint, never the Stage 0 RL checkpoint",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(records)} responses to {args.out}")


if __name__ == "__main__":
    main()
