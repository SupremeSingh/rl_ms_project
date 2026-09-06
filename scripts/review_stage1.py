"""Manual review; automatic scores are hidden and progress is saved per response."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    response_path = args.run / "responses.jsonl"
    digest = hashlib.sha256(response_path.read_bytes()).hexdigest()
    metadata = json.loads((args.run / "metadata.json").read_text())
    if digest != metadata["responses_sha256"]:
        raise ValueError("Response file has changed since generation")
    records = [json.loads(line) for line in response_path.read_text().splitlines()]
    label_path = args.run / "labels.jsonl"
    prior = [json.loads(x) for x in label_path.read_text().splitlines()] if label_path.exists() else []
    if any(x["responses_sha256"] != digest for x in prior):
        raise ValueError("Labels belong to different responses")
    done = {x["id"] for x in prior}
    for index, row in enumerate(records, 1):
        if row["id"] in done:
            continue
        print(f"\n--- {index}/{len(records)}: {row['id']} ---")
        print("QUESTION:", row["prompt"])
        print("REFERENCE ANSWER:", row["ground_truth"])
        print("MODEL RESPONSE:\n", row["response"])
        print("Enter: reward format answer_correct (three 0/1 values), or q to pause.")
        print("Reward: should our specified strict verifier accept this response?")
        print("Format: does the final line satisfy Answer: <decimal number>?")
        print("Answer_correct: is its unambiguous final answer numerically correct, regardless of format?")
        while True:
            value = input("> ").strip().lower()
            if value == "q":
                return
            parts = value.split()
            if len(parts) == 3 and all(x in {"0", "1"} for x in parts):
                reward, fmt, correct = map(int, parts)
                if reward != fmt * correct:
                    print("Reward must be 1 exactly when format and answer are both correct.")
                    continue
                break
            print("Use three values, such as 1 1 1 or 0 0 1; q pauses.")
        notes = input("Notes (Enter to skip): ")
        with label_path.open("a") as handle:
            handle.write(json.dumps({"id": row["id"], "human_reward": reward,
                                    "human_format": fmt, "human_answer_correct": correct,
                                    "notes": notes, "responses_sha256": digest}) + "\n")
    print("All responses reviewed.")


if __name__ == "__main__":
    main()
