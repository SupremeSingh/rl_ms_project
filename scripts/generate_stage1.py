"""Generate frozen completion prompts; score separately in the verifier environment."""
import argparse
import hashlib
import json
from pathlib import Path

AUDIT_OFFSET = 160  # Earlier train (128) and validation (32) prompts are excluded.


def select_rows(source, tokenizer, count):
    from math_rl.prompts import encode_completion
    chosen, excluded = [], 0
    seen = set()
    for row in source:
        question = row['prompt'][1]['content']
        key = ' '.join(question.split())
        if key in seen:
            continue
        seen.add(key)
        rendered, ids = encode_completion(tokenizer, question)
        if len(ids) > 512:
            excluded += 1
            continue
        chosen.append((row, rendered, ids))
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f'Only {len(chosen)} eligible prompts; need {count}')
    return chosen, excluded


def main():
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from math_rl.data import make_row
    from math_rl.prompts import validate_assets
    from math_rl.provenance import snapshot

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=['audit', 'diagnostic'], default='audit')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output exists; use a new directory to preserve earlier results')
    root = Path(__file__).resolve().parents[1]
    assets = json.loads((root / 'configs/assets.json').read_text())
    validate_assets(assets)
    if args.mode == 'audit':
        source = load_dataset('openai/gsm8k', 'main', split='train', revision=assets['dataset_revision'])
        source = source.add_column('original_index', list(range(len(source))))
        source = source.shuffle(seed=assets['seed'])
        # Original indexed IDs survive shuffling. Never sample the development prefix.
        excluded_ids = [f"gsm8k/train/{source[i]['original_index']}" for i in range(AUDIT_OFFSET)]
        development_questions = {' '.join(source[i]['question'].split()) for i in range(AUDIT_OFFSET)}
        candidates = (make_row(source[i], source[i]['original_index'])
                      for i in range(AUDIT_OFFSET, len(source))
                      if ' '.join(source[i]['question'].split()) not in development_questions)
        count = 100
    else:
        candidates = Dataset.from_parquet(str(root / 'data/gsm8k/val.parquet'))
        excluded_ids = []
        count = 32
    model = root / 'models/qwen-math'
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    chosen, excluded = select_rows(candidates, tokenizer, count)
    prompt_ids = [row['extra_info']['prompt_id'] for row, _, _ in chosen]
    if set(prompt_ids) & set(excluded_ids):
        raise RuntimeError('Audit overlaps development prompts')
    engine = LLM(model=str(model), dtype='bfloat16', tensor_parallel_size=1,
                 max_model_len=2560, gpu_memory_utilization=0.6, max_num_seqs=8,
                 max_num_batched_tokens=2560, enforce_eager=True, seed=42,
                 generation_config='vllm')
    params = SamplingParams(n=2, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=2048, seed=42)
    outputs = engine.generate([{'prompt_token_ids': ids} for _, _, ids in chosen], params)
    records = []
    for (row, rendered, ids), output in zip(chosen, outputs, strict=True):
        if len(output.outputs) != 2:
            raise RuntimeError('Expected exactly two responses per prompt')
        for completion in output.outputs:
            records.append(dict(id=f"{row['extra_info']['prompt_id']}/{completion.index}",
                                prompt_id=row['extra_info']['prompt_id'], prompt=rendered,
                                prompt_token_ids=ids, ground_truth=row['reward_model']['ground_truth'],
                                response=completion.text, reward=None,
                                finish_reason=completion.finish_reason,
                                response_tokens=len(completion.token_ids)))
    args.out.mkdir(parents=True)
    payload = ''.join(json.dumps(r) + '\n' for r in records)
    (args.out / 'responses.jsonl').write_text(payload)
    metadata = dict(mode=args.mode, rule_version='math-verify-v1', assets=assets,
                    prompt_style='completion', seed=42, temperature=1.0, n=2,
                    top_p=1.0, top_k=-1, max_tokens=2048, dtype='bfloat16',
                    source='openai/gsm8k main train', audit_offset=AUDIT_OFFSET if args.mode == 'audit' else None,
                    excluded_development_ids=excluded_ids, prompt_ids=prompt_ids,
                    excluded_overlong=excluded, provenance=snapshot(root),
                    responses_sha256=hashlib.sha256(payload.encode()).hexdigest())
    (args.out / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved {len(records)} responses to {args.out}; Math-Verify scoring is the next step.')


if __name__ == '__main__':
    main()
