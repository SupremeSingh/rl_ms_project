import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from math_rl.data import prepare_data
from math_rl.prompts import MODEL_ID, validate_assets

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / "configs/assets.json"
    if manifest.exists():
        assets = json.loads(manifest.read_text())
    else:
        api = HfApi()
        model_id = MODEL_ID
        assets = {
            "model_id": model_id,
            "model_revision": api.model_info(model_id).sha,
            "dataset_revision": api.dataset_info("openai/gsm8k").sha,
            "seed": 42,
        }
    validate_assets(assets)
    prepare_data(ROOT / "data/gsm8k", assets["dataset_revision"], assets["seed"])
    snapshot_download(assets["model_id"], revision=assets["model_revision"],
                      local_dir=ROOT / "models/qwen-math")
    manifest.write_text(json.dumps(assets, indent=2) + "\n")
    print("Prepared 128 training prompts, 32 validation prompts, and model weights.")


if __name__ == "__main__":
    main()
