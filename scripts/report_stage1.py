import argparse
import hashlib
import json
from pathlib import Path
from math_rl.audit import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--min-format", type=float, default=0.95)
    p.add_argument("--max-truncation", type=float, default=0.05)
    args = p.parse_args()
    if not 0 <= args.min_format <= 1 or not 0 <= args.max_truncation <= 1:
        p.error("Thresholds must be in [0, 1]")
    path = args.run / "responses.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = json.loads((args.run / "metadata.json").read_text())
    if digest != metadata["responses_sha256"]:
        raise ValueError("Response file has changed since generation")
    records = [json.loads(x) for x in path.read_text().splitlines()]
    label_path = args.run / "labels.jsonl"
    labels = [json.loads(x) for x in label_path.read_text().splitlines()] if label_path.exists() else []
    if any(x["responses_sha256"] != digest for x in labels):
        raise ValueError("Labels belong to different responses")
    report = summarize(records, labels, args.min_format, args.max_truncation)
    report["metadata"] = metadata
    text = json.dumps(report, indent=2) + "\n"
    (args.run / "report.json").write_text(text)
    print(text)
    raise SystemExit(0 if report["stage1_pass"] else 1)


if __name__ == "__main__":
    main()
