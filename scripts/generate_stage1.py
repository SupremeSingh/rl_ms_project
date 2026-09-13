"""Generate 100 prompt pairs with the untrained initial checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    from datasets import Dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from math_rl.reward import score_contract, extract_answer
    from math_rl.prompts import validate_assets, validate_prompt, display_prompt, prompt_for_contract
    from math_rl.provenance import snapshot

    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=root / "outputs/stage1")
    p.add_argument("--mode", choices=["audit", "diagnostic"], default="audit")
    p.add_argument("--contract", choices=["answer", "boxed"], default="answer")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--prompts", type=int)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.prompts is None:
        args.prompts = 32 if args.mode == "diagnostic" else 100
    if args.max_tokens < 1 or args.temperature <= 0 or args.prompts < 1:
        p.error("Token budget, temperature, and prompt count must be positive")
    if args.mode == "audit" and args.prompts < 100:
        p.error("Stage 1 requires at least 100 pairs / 200 responses")
    if args.out.exists():
        p.error("Output directory exists; choose a new --out to preserve the audit")
    assets = json.loads((root / "configs/assets.json").read_text())
    validate_assets(assets)
    model = root / "models/qwen-math"
    data_path = root / "data/gsm8k" / ("val.parquet" if args.mode == "diagnostic" else "train.parquet")
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    dataset = Dataset.from_parquet(str(data_path))
    eligible = []
    excluded = 0
    for row in dataset:
        validate_prompt(row["prompt"])
        row["prompt"] = prompt_for_contract(row["prompt"][1]["content"], args.contract)
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
              max_model_len=512 + args.max_tokens, gpu_memory_utilization=0.6,
              max_num_seqs=8, max_num_batched_tokens=512 + args.max_tokens,
              enforce_eager=True, seed=args.seed, generation_config="vllm")
    params = SamplingParams(n=2, temperature=args.temperature, top_p=1.0, top_k=-1,
                            max_tokens=args.max_tokens, seed=args.seed)
    outputs = llm.generate([{"prompt_token_ids": ids} for _, ids in chosen], params)
    records = []
    for (row, ids), output in zip(chosen, outputs, strict=True):
        if len(output.outputs) != 2:
            raise RuntimeError("Expected two responses per prompt")
        for completion in output.outputs:
            records.append({
                "id": f"{row['extra_info']['prompt_id']}/{completion.index}",
                "prompt_id": row["extra_info"]["prompt_id"],
                "prompt": display_prompt(row["prompt"]),
                "messages": row["prompt"],
                "ground_truth": row["reward_model"]["ground_truth"],
                "response": completion.text,
                "contract": args.contract,
                "format_valid": extract_answer(completion.text, args.contract) is not None,
                "reward": score_contract(completion.text,
                                         row["reward_model"]["ground_truth"], args.contract),
                "finish_reason": completion.finish_reason,
                "prompt_tokens": len(ids),
                "response_tokens": len(completion.token_ids),
            })
    args.out.mkdir(parents=True)
    payload = "".join(json.dumps(x) + "\n" for x in records)
    (args.out / "responses.jsonl").write_text(payload)
    metadata = {
        "provenance": snapshot(root),
        "assets": assets, "seed": args.seed, "n": 2,
        "mode": args.mode, "contract": args.contract,
        "temperature": args.temperature, "top_p": 1.0, "top_k": -1,
        "max_tokens": args.max_tokens, "dtype": "bfloat16",
        "prompt_count": args.prompts, "excluded_overlong": excluded,
        "eligible_count": len(eligible), "source": str(data_path), "prompt_ids": [row["extra_info"]["prompt_id"] for row, _ in chosen],
        "responses_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "reward_sha256": hashlib.sha256((root / "src/math_rl/reward.py").read_bytes()).hexdigest(),
        "checkpoint_note": "Initial downloaded checkpoint, never the Stage 0 RL checkpoint",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(records)} responses to {args.out}")
    from math_rl.audit import summarize
    report = summarize(records, [])
    report.update(mode=args.mode, contract=args.contract,
                  automatic_format_rate=sum(r["format_valid"] for r in records) / len(records),
                  mean_response_tokens=sum(r["response_tokens"] for r in records) / len(records),
                  max_response_tokens=max(r["response_tokens"] for r in records))
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
