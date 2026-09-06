"""Fail before GPU work when the prepared experiment is incompatible."""
import json
from importlib.metadata import distribution
from pathlib import Path
from math_rl.prompts import validate_assets, validate_prompt


def main():
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    for name in ("configs/environment.txt", "configs/container.sha256"):
        if not (root / name).is_file():
            raise RuntimeError(f"Missing runtime provenance: {name}; follow README setup")
    validate_assets(json.loads((root / "configs/assets.json").read_text()))
    direct = json.loads(distribution("verl").read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != "8d9e350ea58c7ad4b50dd14d9dcb50577242c55f":
        raise RuntimeError("VERL is not installed at the pinned commit; rerun container setup")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("Request an Ampere GPU: --gres=gpu:a6000:1")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable in this GPU/runtime combination")
    tokenizer = AutoTokenizer.from_pretrained(root / "models/qwen-math", local_files_only=True)
    for split in ("train", "val"):
        rows = Dataset.from_parquet(str(root / f"data/gsm8k/{split}.parquet"))
        for row in rows:
            validate_prompt(row["prompt"])
        rendered = tokenizer.apply_chat_template(rows[0]["prompt"], tokenize=False, add_generation_prompt=True)
        if "\\boxed" in rendered:
            raise RuntimeError("Tokenizer injected a conflicting boxed-answer instruction")
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                      "cuda": torch.version.cuda, "bf16": True}))


if __name__ == "__main__":
    main()
