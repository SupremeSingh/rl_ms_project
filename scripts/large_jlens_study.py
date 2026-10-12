"""Resumable large-N frozen-policy probes, with official full J-lens readouts.

Run stages through submit_large_jlens.sh. Test scores are produced only by report,
after every arm/head has a validation-selected checkpoint. No actor updates.
"""
import argparse
from collections import Counter
from importlib.metadata import version
import json
import html
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch

import frozen_critics as frozen
from math_rl.large_probe import (ARMS, HEADS, SPLITS, assign_splits, candidate,
    digest, family_key, paired_errors, prefix_positions, streaming_normalization)
from math_rl.provenance import sha256, snapshot

DATASET = 'AI-MO/NuminaMath-1.5'
MODEL = ROOT/'models/qwen-math'
SUBJECTS = ('algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra',
            'number_theory', 'prealgebra', 'precalculus')
LINEAR_HEADS = ('ridge', 'lstd-0.99')


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    frozen.atomic_json(path, value)


def review_html(path, records):
    """Escape model output; readable examples do not execute generated markup."""
    rows = ['<!doctype html><meta charset="utf-8"><title>Frozen-model answer review</title>',
        '<style>body{max-width:1000px;margin:30px auto;font:16px system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border-top:1px solid #aaa;padding:12px}</style>',
        '<h1>Saved answer audit</h1><p>Automated labels require review. All text below is escaped.</p>']
    for r in records:
        answer = r['answer']
        rows.append('<details><summary>'+html.escape(f'{r["id"]} — {answer["verifier_status"]}')+'</summary><pre>'+html.escape(
            f'QUESTION\n{r["question"]}\n\nREFERENCE: {r["reference"]}\n\nRESPONSE\n{answer["response"]}'
            f'\n\nEXTRACTED: {answer.get("extracted")}\nFINISH: {answer["finish_reason"]}'
            f'\n\nREFERENCE SOLUTION\n{r["reference_solution"]}')+'</pre></details>')
    path.write_text('\n'.join(rows))


def config(smoke):
    return dict(version='full-readout-large-v3', smoke=smoke,
        counts=dict(calibration=2, screen=8, train=12, val=6, test=6) if smoke else
               dict(calibration=512, screen=256, train=10000, val=1000, test=2000),
        responses=2 if smoke else 8, screen_responses=2 if smoke else 4,
        max_tokens=512 if smoke else 3584, max_prompt_tokens=512,
        generation_seed=271828, temperature=1., prefix_seed=1729,
        lens_max_tokens=128 if smoke else 512, lens_dim_batch=2,
        seeds=[42] if smoke else [42, 43, 44], lrs=[.001] if smoke else [.0003, .001],
        epochs=2 if smoke else 120, min_epochs=1 if smoke else 30,
        patience=1 if smoke else 20, width=256, shards=4,
        linear_epsilon=.01, linear_gamma=1., linear_lambda=.99, linear_checkpoint_questions=64,
        verifier_rule='math-verify-v1', arms=list(ARMS), heads=list(HEADS), linear_heads=list(LINEAR_HEADS),
        primary_contrasts=['ridge/direct_minus_hidden_two_thirds', 'lstd-0.99/direct_minus_hidden_two_thirds'],
        screen_gate=dict(min_accuracy=.2, max_accuracy=.8, max_exclusion=.1, max_truncation=.15))


def context_budget(cfg, model_config, questions=()):
    """Prompt and answer share the model's native context; never extend RoPE here."""
    limit = model_config.get('max_position_embeddings')
    prompt, answer = cfg['max_prompt_tokens'], cfg['max_tokens']
    if not isinstance(limit, int) or limit < 1 or prompt < 1 or answer < 1:
        raise ValueError('Invalid or missing native context budget')
    total = prompt+answer
    if total > limit:
        raise ValueError(f'Prompt reserve {prompt} + answer budget {answer} = {total} exceeds native context {limit}')
    if any(not q['prompt_token_ids'] or len(q['prompt_token_ids']) > prompt for q in questions):
        raise ValueError('Saved prompt exceeds its reserved context budget')
    return total


def repair_context(out):
    """Explicit migration ONLY before any generation; preserve the locked questions."""
    path = out/'manifest.json'; m = read(path)
    new = config(m['config']['smoke'])
    if m['config'] == new:
        validate(out)
        print('Context repair already applied; ready to resume screen.', flush=True)
        return
    expected = dict(new, version='full-readout-large-v2', max_tokens=512 if new['smoke'] else 4096)
    if m['config'] != expected:
        raise ValueError('Not the known v2 context-budget failure; use a new study')
    for folder in ('trajectories', 'pending', 'features', 'packed', 'fits', 'linear-statistics', 'linear-fits'):
        if (out/folder).exists() and any(p.is_file() for p in (out/folder).rglob('*')):
            raise ValueError(f'Cannot change budgets after data exist: {folder}')
    if any((out/name).exists() for name in ('screen.json', 'lens.pt', 'lens-checkpoint.pt', 'summary.json')):
        raise ValueError('Cannot migrate a study with generated or fitted results')
    if sha256(out/'questions.json') != m['questions_sha256']:
        raise ValueError('Question manifest changed')
    for name, checksum in m['model_hashes'].items():
        if sha256(MODEL/name) != checksum:
            raise ValueError(f'Model/tokenizer changed: {name}')
    current = snapshot(ROOT)['files']
    # Only this script implements the fix. Do not bless unrelated source changes.
    changed = [name for name in set(current)|set(m['code']) if current.get(name) != m['code'].get(name)]
    if set(changed)-{'scripts/large_jlens_study.py'}:
        raise ValueError(f'Unrelated code/config changes prevent migration: {changed}')
    total = context_budget(new, read(MODEL/'config.json'), read(out/'questions.json'))
    backup = out/'manifest.before-context-fix.json'
    if backup.exists() and backup.read_bytes() != path.read_bytes():
        raise ValueError('Existing migration backup differs; inspect before proceeding')
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    m.update(config=new, code=current)
    m['context_repair'] = dict(previous_manifest_sha256=sha256(backup), changed_code=changed,
        old_answer_budget=expected['max_tokens'], answer_budget=new['max_tokens'],
        total_context=total, reason='Fix prompt-plus-answer overflow before any generation; questions/model unchanged')
    write(path, m)
    print(f'Preserved all prepared questions. Answer budget {new["max_tokens"]}; total context {total}. Resume screen.', flush=True)


def repair_verifier(out):
    """Repair explicit rule dispatch before scoring; keep pending token batches intact."""
    path = out/'manifest.json'; m = read(path)
    if m['config'] != config(m['config']['smoke']):
        raise ValueError('Expected current context-safe protocol; repair context first')
    if sha256(out/'questions.json') != m['questions_sha256']:
        raise ValueError('Question manifest changed')
    for name, checksum in m['model_hashes'].items():
        if sha256(MODEL/name) != checksum:
            raise ValueError(f'Model/tokenizer changed: {name}')
    current = snapshot(ROOT)['files']
    changed = sorted(name for name in set(current)|set(m['code'])
                     if current.get(name) != m['code'].get(name))
    if not changed and 'verifier_repair' in m:
        print('Verifier repair already applied; ready to resume screen.', flush=True)
        return
    allowed = {'scripts/large_jlens_study.py', 'src/math_rl/verifier_batch.py'}
    if set(changed)-allowed:
        raise ValueError(f'Unrelated code/config changes prevent migration: {changed}')
    for folder in ('trajectories', 'features', 'packed', 'fits', 'linear-statistics', 'linear-fits'):
        if (out/folder).exists() and any(p.is_file() for p in (out/folder).rglob('*')):
            raise ValueError(f'Cannot repair dispatch after scored/fitted data exist: {folder}')
    if any((out/name).exists() for name in ('screen.json', 'lens.pt', 'summary.json')):
        raise ValueError('Cannot repair a study with completed results')
    backup = out/'manifest.before-verifier-fix.json'
    if backup.exists() and backup.read_bytes() != path.read_bytes():
        raise ValueError('Existing migration backup differs; inspect before proceeding')
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    m['code'] = current
    m['verifier_repair'] = dict(previous_manifest_sha256=sha256(backup), changed_code=changed,
        reason='Accept explicit math-verify-v1 as the existing default grader; no scoring or generation policy change')
    write(path, m)
    print('Verifier dispatch repaired; pending answers preserved. Resume screen.', flush=True)


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from math_rl.prompts import encode_completion, validate_assets
    from math_rl.jspace import geometric_layers
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        m = read(out/'manifest.json')
        if m['config'] != config(args.smoke):
            raise ValueError('Existing protocol differs; use a new output directory')
        return
    cfg = config(args.smoke)
    context_budget(cfg, read(MODEL/'config.json'))
    assets = read(ROOT/'configs/assets.json'); validate_assets(assets)
    revisions_path = out/'dataset-revisions.json'
    if revisions_path.exists():
        revisions = read(revisions_path)
    else:
        revisions = {name: HfApi().dataset_info(name).sha for name in
                     (DATASET, 'openai/gsm8k', 'EleutherAI/hendrycks_math')}
        write(revisions_path, revisions)
    blocked = set()
    for split in ('train', 'test'):
        for row in load_dataset('openai/gsm8k', 'main', split=split, revision=revisions['openai/gsm8k']):
            blocked.add(family_key(row['question']))
        for subject in SUBJECTS:
            for row in load_dataset('EleutherAI/hendrycks_math', subject, split=split,
                                    revision=revisions['EleutherAI/hendrycks_math']):
                blocked.add(family_key(row['problem']))
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    reasons = Counter(); eligible = {}; source = load_dataset(DATASET, split='train', revision=revisions[DATASET])
    for i, row in enumerate(source):
        q, reason = candidate(row)
        if q is None:
            reasons[reason] += 1; continue
        family = family_key(q['question'])
        if family in blocked or family in eligible:
            reasons['blocked_or_duplicate_numeric_template'] += 1; continue
        text, tokens = encode_completion(tok, q['question'])
        if len(tokens) > cfg['max_prompt_tokens']:
            reasons['prompt_too_long'] += 1; continue
        eligible[family] = dict(q, id=f'numina-cn-k12/{i}', family=family,
                                prompt=text, prompt_token_ids=tokens)
    write(out/'eligibility.json', dict(source_questions=len(source), eligible_families=len(eligible),
        exclusions=dict(reasons), blocked_families=len(blocked),
        scope='Reference-only conservative eligibility; exact/number-template overlap blocked, not semantic or pretraining overlap.'))
    questions = assign_splits(eligible.values(), cfg['counts'])
    context_budget(cfg, read(MODEL/'config.json'), questions)
    write(out/'questions.json', questions)
    model_config = read(MODEL/'config.json')
    model_hashes = {p.name: sha256(p) for p in MODEL.iterdir()
                    if p.is_file() and (p.suffix in ('.json', '.safetensors') or p.name in ('merges.txt', 'vocab.json'))}
    if not any(name.endswith('.safetensors') for name in model_hashes):
        raise ValueError('Missing base-model weights')
    m = dict(config=cfg, dataset=DATASET, revisions=revisions, assets=assets,
        generation_environment={name: version(name) for name in ('torch', 'transformers', 'numpy', 'vllm')},
        questions_sha256=sha256(out/'questions.json'), model_hashes=model_hashes,
        layers=geometric_layers(model_config['num_hidden_layers'], 2),
        two_thirds=(2*model_config['num_hidden_layers']+2)//3-1,
        dimension=model_config['hidden_size'], rms_eps=model_config.get('rms_norm_eps', 1e-6),
        code=snapshot(ROOT)['files'],
        protocol='Fixed frozen base; one question per numeric template; disjoint calibration/screen/train/val/test. '
                 'All attempts reported; correct/incorrect statuses only fit probes. Eight reward-blind prefixes per retained answer. '
                 'BCE probes plus raw-sum all-token Ridge/LSTD(.99). Accuracy and Brier reported. No sparse readout or PPO.')
    write(out/'manifest.json', m)
    print(f'Locked {len(questions)} questions; {sum(cfg["counts"][s] for s in SPLITS)*cfg["responses"]} main answers', flush=True)


def validate(out):
    m = read(out/'manifest.json')
    if sha256(out/'questions.json') != m['questions_sha256']:
        raise ValueError('Question manifest changed')
    for name, checksum in m['model_hashes'].items():
        if sha256(MODEL/name) != checksum:
            raise ValueError(f'Base model/tokenizer changed: {name}')
    for name, checksum in m['code'].items():
        if sha256(ROOT/name) != checksum:
            raise ValueError(f'Code/config changed after protocol lock: {name}; restore it or use a new study')
    return m, read(out/'questions.json')


def trajectory(out, q):
    return out/'trajectories'/f'{q["index"]:05d}.json'


def feature(out, q):
    return out/'features'/f'{q["index"]:05d}.pt'


def generate(out, m, questions, stage, shard):
    from vllm import LLM, SamplingParams
    from math_rl.ppo_reward import compute_score
    cfg = m['config']
    total_context = context_budget(cfg, read(MODEL/'config.json'), questions)
    if {name: version(name) for name in m['generation_environment']} != m['generation_environment']:
        raise ValueError('Generation environment changed since preparation')
    selected = [q for q in questions if (q['split'] in ('screen', 'calibration') if stage == 'screen'
                else q['split'] in SPLITS and q['index'] % cfg['shards'] == shard)]
    if stage != 'screen' and not read(out/'screen.json')['pass']:
        raise ValueError('Screen gate has not passed')
    pending_questions = [q for q in selected if not trajectory(out, q).exists()]
    if pending_questions:
        engine = LLM(model=str(MODEL), dtype='bfloat16', tensor_parallel_size=1,
            max_model_len=total_context, gpu_memory_utilization=.65,
            max_num_seqs=8, enforce_eager=True, seed=cfg['generation_seed'], generation_config='vllm')
    for q in selected:
        count = 1 if q['split'] == 'calibration' else cfg['screen_responses'] if q['split'] == 'screen' else cfg['responses']
        key = digest([m['questions_sha256'], q['id'], cfg['generation_seed'], count, cfg['max_tokens']])
        path = trajectory(out, q)
        if path.exists():
            if read(path)['key'] != key:
                raise ValueError('Stale generated batch')
            continue
        pending = out/'pending'/path.name
        if pending.exists():
            raw = read(pending)
            if raw['key'] != key:
                raise ValueError('Stale pending generation')
        else:
            params = SamplingParams(n=count, temperature=cfg['temperature'], top_p=1., top_k=-1,
                max_tokens=cfg['max_tokens'], seed=cfg['generation_seed']+q['index'])
            answers = engine.generate([{'prompt_token_ids': q['prompt_token_ids']}], params)[0].outputs
            raw = dict(key=key, question_id=q['id'], responses=[dict(response=r.text, token_ids=list(r.token_ids),
                finish_reason=r.finish_reason, stop_reason=r.stop_reason) for r in answers])
            write(pending, raw)
        if len(raw['responses']) != count:
            raise ValueError('Incomplete answer batch')
        scores = compute_score(['math_numeric']*count, [r['response'] for r in raw['responses']],
                               [q['ground_truth']]*count, exclude_errors=True, rule=cfg['verifier_rule'])
        rows = [dict(r, **score) for r, score in zip(raw['responses'], scores, strict=True)]
        for r in rows:
            if not r['token_ids']:
                r.update(score=None, verifier_status='empty_response')
        write(path, dict(raw, responses=rows))
        pending.unlink()
        print(f'Saved {q["id"]} ({q["split"]})', flush=True)
    if stage == 'screen':
        rows = [r for q in selected if q['split'] == 'screen' for r in read(trajectory(out, q))['responses']]
        statuses = Counter(r['verifier_status'] for r in rows)
        accuracy = statuses['correct']/len(rows)
        exclusion = sum(not frozen.usable_response(r) for r in rows)/len(rows)
        truncation = sum(r['finish_reason'] == 'length' for r in rows)/len(rows)
        gate = cfg['screen_gate']
        passed = cfg['smoke'] or (gate['min_accuracy'] <= accuracy <= gate['max_accuracy']
                                 and exclusion <= gate['max_exclusion'] and truncation <= gate['max_truncation'])
        write(out/'screen.json', dict(pass_checks=passed, **{'pass': passed}, accuracy=accuracy,
            exclusion_rate=exclusion, truncation_rate=truncation, statuses=dict(statuses),
            attempts=len(rows), smoke=cfg['smoke'], scope='Disjoint screening; does not certify reward correctness.'))
        review_html(out/'screen-responses.html', [dict(id=q['id']+f'/{i}', question=q['question'],
            reference=q['ground_truth'], reference_solution=q['reference_solution'], answer=r)
            for q in selected if q['split'] == 'screen' for i, r in enumerate(read(trajectory(out, q))['responses'])])
        if not passed:
            raise ValueError('Screen outside preregistered range. Inspect screen.json and saved answers before spending full budget.')


def lens(out, m, questions):
    import jlens
    import compare_jspace_probes as original
    from math_rl.jspace import load_readout_weights
    from math_rl.direct_jlens import readout_factor
    if not read(out/'screen.json')['pass']:
        raise ValueError('Screen did not pass')
    environment = original.installed_lens()
    if (out/'lens-metadata.json').exists():
        meta = read(out/'lens-metadata.json')
        if (meta['environment'] != environment or sha256(out/'lens.pt') != meta['sha256']
                or sha256(out/'readout-factor.pt') != meta['factor_sha256']):
            raise ValueError('Lens/environment changed')
        return
    hf, tok = original.load_model(); cfg = m['config']
    prompts = []
    for q in questions:
        if q['split'] != 'calibration':
            continue
        ids = q['prompt_token_ids'] + read(trajectory(out, q))['responses'][0]['token_ids']
        limit = min(len(ids), cfg['lens_max_tokens'])
        rng = np.random.default_rng(817+q['index'])
        length = int(rng.integers(min(limit, max(32, len(q['prompt_token_ids']))), limit+1))
        text = tok.decode(ids[:length], skip_special_tokens=False)
        # Official fitter tokenizes text. Record its exact token inputs, not assumed round-trip identity.
        prompts.append(text)
    write(out/'lens-inputs.json', dict(prompts=prompts,
        token_ids=[tok.encode(p, add_special_tokens=False) for p in prompts],
        recipe='First attempt irrespective of reward; uniformly truncated full causal context up to lens_max_tokens.'))
    adapter = jlens.from_hf(hf, tok, force_bos=False)
    jlens.configure_logging()
    fitted = jlens.fit(adapter, prompts=prompts, source_layers=m['layers'][1:],
        target_layer=m['layers'][0], dim_batch=cfg['lens_dim_batch'], max_seq_len=cfg['lens_max_tokens'],
        skip_first=16, checkpoint_path=str(out/'lens-checkpoint.pt'))
    if fitted.n_prompts != len(prompts) or any(not torch.isfinite(j).all() for j in fitted.jacobians.values()):
        raise ValueError('Incomplete/nonfinite official lens fit')
    fitted.save(str(out/'lens.tmp.pt'), dtype=torch.float32); (out/'lens.tmp.pt').replace(out/'lens.pt')
    del hf; torch.cuda.empty_cache()
    weights = load_readout_weights(MODEL)
    record = readout_factor(weights['lm_head.weight'].cuda(), weights['model.norm.weight'].cuda())
    frozen.save_tensor(out/'readout-factor.pt', record)
    write(out/'lens-metadata.json', dict(environment=environment, sha256=sha256(out/'lens.pt'),
        factor_sha256=sha256(out/'readout-factor.pt'), calibration_prompts=len(prompts),
        source_layers=m['layers'][1:], target=m['layers'][0],
        recipe='Official jlens.fit; final transport identity. Exact full-vocabulary Gram factor, no truncation/sparsity.'))


def extract(out, m, questions, shard):
    import compare_jspace_probes as original
    from math_rl.critic_layers import sampled_block_states
    selected = [q for q in questions if q['split'] in SPLITS and q['index'] % m['config']['shards'] == shard]
    hf, _ = original.load_model()
    layers = list(dict.fromkeys(m['layers']+[m['two_thirds']]))
    (out/'features').mkdir(exist_ok=True)
    checked = False
    for q in selected:
        source = trajectory(out, q); checksum = sha256(source); target = feature(out, q)
        if target.exists():
            old = torch.load(target, weights_only=True)
            if old['source_sha256'] != checksum:
                raise ValueError('Saved answers changed since feature extraction')
            continue
        rows = read(source)['responses']; blocks = {l: [] for l in layers}
        labels = []; positions = []; response_ids = []
        for i, row in enumerate(rows):
            if not frozen.usable_response(row):
                continue
            ids = q['prompt_token_ids']+row['token_ids']; n = len(q['prompt_token_ids']); t = len(row['token_ids'])
            tokens = torch.tensor([ids], device='cuda')
            pos = prefix_positions(t, q['index'], i)
            captured = sampled_block_states(hf.model, tokens, n, t, pos, layers)
            if not checked:
                # Change all future tokens while keeping shape fixed: verifies no lookahead.
                p = pos[1]; altered = tokens.clone(); altered[:, n+p:] = 0
                causal = sampled_block_states(hf.model, altered, n, t, [p], layers)
                errors = {}
                for l in layers:
                    torch.testing.assert_close(captured[l][1], causal[l][0], rtol=1e-4, atol=1e-4)
                    errors[str(l+1)] = float((captured[l][1]-causal[l][0]).abs().max())
                write(out/f'causality-{shard}.json', dict(question=q['id'], prefix=p, max_difference=errors))
                checked = True
            for l in layers:
                blocks[l].append(captured[l].float())
            labels.extend([float(row['score'])]*8); positions.extend(pos); response_ids.extend([i]*8)
        record = dict(source_sha256=checksum, question=q['index'],
            blocks={l: torch.cat(xs) if xs else torch.empty((0, m['dimension'])) for l, xs in blocks.items()},
            y=torch.tensor(labels), position=torch.tensor(positions, dtype=torch.long),
            response=torch.tensor(response_ids, dtype=torch.long))
        frozen.save_tensor(target, record)
        print(f'Cached {q["id"]}: {len(labels)} prefixes', flush=True)
    write(out/f'extraction-{shard}.json', dict(environment=original.installed_lens(),
        features={feature(out, q).name: sha256(feature(out, q)) for q in selected}))


def pack(out, m, questions, arm_index):
    import jlens
    import compare_jspace_probes as original
    from math_rl.direct_jlens import compact_readout, rms_coordinates
    from math_rl.jspace import load_readout_weights
    arm = ARMS[arm_index]; dest = out/'packed'/arm; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        return
    meta = read(out/'lens-metadata.json')
    if original.installed_lens() != meta['environment'] or sha256(out/'lens.pt') != meta['sha256']:
        raise ValueError('Lens/environment changed')
    if sha256(out/'readout-factor.pt') != meta['factor_sha256']:
        raise ValueError('Readout factor changed')
    hashes = {}
    for shard in range(m['config']['shards']):
        info = read(out/f'extraction-{shard}.json')
        if info['environment'] != meta['environment']:
            raise ValueError('Lens and extraction environments differ')
        hashes.update(info['features'])
    fitted = jlens.JacobianLens.load(str(out/'lens.pt'))
    factor = torch.load(out/'readout-factor.pt', weights_only=True)['factor'].float()
    weights = load_readout_weights(MODEL); gain = weights['model.norm.weight'].float()
    U = weights['lm_head.weight']; checks = []
    counts = {}; output_hashes = {}
    for split in SPLITS:
        qs = [q for q in questions if q['split'] == split]
        count = 0
        for q in qs:
            path = feature(out, q)
            if sha256(path) != hashes[path.name]:
                raise ValueError('Feature cache changed')
            count += len(torch.load(path, weights_only=True)['y'])
        if not count:
            raise ValueError(f'Empty retained {split} set')
        dimension = m['dimension']*(len(m['layers']) if arm.endswith('concat') else 1)
        x = np.lib.format.open_memmap(dest/f'{split}-x.tmp.npy', mode='w+', dtype='float32', shape=(count, dimension))
        metadata = {k: [] for k in ('y', 'position', 'response', 'question')}; cursor = 0
        for q in qs:
            data = torch.load(feature(out, q), weights_only=True); n = len(data['y'])
            if not n:
                continue
            blocks = data['blocks']; eps = m['rms_eps']
            if arm == 'hidden_two_thirds':
                values = blocks[m['two_thirds']]
            elif arm == 'final':
                values = rms_coordinates(blocks[m['layers'][0]], eps)*gain
            else:
                parts = []
                for l in m['layers']:
                    h = blocks[l]
                    if arm == 'direct_jlens_concat':
                        h = h if l == m['layers'][0] else fitted.transport(h, l)
                        value = compact_readout(h, factor, eps)
                        if not checks or len(checks) < len(m['layers']):
                            full = (rms_coordinates(h[:2].double(), eps)*gain.double())@U.double().T
                            compact = compact_readout(h[:2].double(), factor.double(), eps)
                            torch.testing.assert_close(compact@compact.T, full@full.T, rtol=2e-5, atol=1e-5)
                            checks.append(dict(block=l+1, gram_check=True))
                    elif arm == 'rms_hidden_concat':
                        value = rms_coordinates(h, eps)
                    else:
                        value = h
                    parts.append(value)
                values = torch.cat(parts, 1)
            if not torch.isfinite(values).all():
                raise ValueError('Nonfinite readout')
            x[cursor:cursor+n] = values.numpy(); cursor += n
            for k in ('y', 'position', 'response'):
                metadata[k].append(data[k])
            metadata['question'].append(torch.full((n,), q['index'], dtype=torch.long))
        x.flush(); del x
        (dest/f'{split}-x.tmp.npy').replace(dest/f'{split}-x.npy')
        frozen.save_tensor(dest/f'{split}-labels.pt', {k: torch.cat(v) for k, v in metadata.items()})
        counts[split] = count
        for suffix in ('x.npy', 'labels.pt'):
            path = dest/f'{split}-{suffix}'; output_hashes[path.name] = sha256(path)
    write(dest/'complete.json', dict(counts=counts, files=output_hashes, readout_checks=checks))


def dataset(out, arm, split):
    dest = out/'packed'/arm
    record = read(dest/'complete.json')
    for suffix in ('x.npy', 'labels.pt'):
        path = dest/f'{split}-{suffix}'
        if sha256(path) != record['files'][path.name]:
            raise ValueError('Packed data changed')
    data = torch.load(dest/f'{split}-labels.pt', weights_only=True)
    # Copy-on-write mmap: tensor indexing copies only minibatches, not the whole feature matrix.
    data['x'] = torch.from_numpy(np.load(dest/f'{split}-x.npy', mmap_mode='c'))
    return data


def fit(out, m, task):
    from math_rl.critic_probe import fit_trial
    import compare_jspace_probes as original
    if original.installed_lens() != read(out/'lens-metadata.json')['environment']:
        raise ValueError('Use the locked feature/fitting environment')
    arm, kind = [(a, h) for a in ARMS for h in HEADS][task]
    dest = out/'fits'/arm/kind; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'selection.json').exists():
        return
    cfg = m['config']; train = dataset(out, arm, 'train'); val = dataset(out, arm, 'val')
    if len(train['y'].unique()) != 2 or len(val['y'].unique()) != 2:
        raise ValueError('Both outcomes needed in train/validation')
    mean, scale = streaming_normalization(train['x'].numpy())
    selected = {}; summaries = []
    for seed in cfg['seeds']:
        trials = []
        for lr in cfg['lrs']:
            path = dest/f'seed{seed}-lr{lr}.pt'
            if path.exists():
                record = torch.load(path, weights_only=True)
            else:
                record = fit_trial(kind, train, val, mean.cuda(), scale.cuda(), torch.device('cuda'),
                    seed, lr, cfg['epochs'], width=cfg['width'], patience=cfg['patience'], min_epochs=cfg['min_epochs'])
                record.update(mean=mean, scale=scale)
                frozen.save_tensor(path, record)
            trials.append((record['validation_brier'], path, record))
            summaries.append({k: v for k, v in record.items() if k not in ('state', 'mean', 'scale')})
        _, path, best = min(trials, key=lambda t: t[0])
        selected[str(seed)] = dict(path=str(path.relative_to(out)), sha256=sha256(path),
            validation_brier=best['validation_brier'], best_at_budget_limit=best['best_at_budget_limit'])
    write(dest/'selection.json', dict(arm=arm, head=kind, selected=selected, trials=summaries,
        scope='BCE training; validation Brier selects learning rate/epoch separately per head seed; no test loading.'))


def linear_features(arm, blocks, m, fitted, factor, gain):
    from math_rl.direct_jlens import compact_readout, rms_coordinates
    eps = m['rms_eps']
    if arm == 'hidden_two_thirds':
        return blocks[m['two_thirds']]
    if arm == 'final':
        return rms_coordinates(blocks[m['layers'][0]], eps)*gain
    values = []
    for layer in m['layers']:
        h = blocks[layer]
        if arm == 'direct_jlens_concat':
            h = h if layer == m['layers'][0] else fitted.transport(h, layer)
            values.append(compact_readout(h, factor, eps))
        else:
            values.append(rms_coordinates(h, eps) if arm == 'rms_hidden_concat' else h)
    return torch.cat(values, 1)


def linear_stats(out, m, questions, shard):
    """One extra teacher-forced pass; stream all-token features into five matrix pairs."""
    import jlens
    import compare_jspace_probes as original
    from math_rl.critic_layers import sampled_block_states
    from math_rl.large_linear import statistics, accumulate
    if m['config']['linear_gamma'] != 1. or m['config']['linear_lambda'] != .99:
        raise ValueError('This locked comparison requires gamma=1 and LSTD lambda=.99')
    if shard >= m['config']['shards']:
        raise ValueError('Shard out of bounds')
    metadata = read(out/'lens-metadata.json')
    if original.installed_lens() != metadata['environment']:
        raise ValueError('Linear fitting environment changed')
    if (sha256(out/'lens.pt') != metadata['sha256']
            or sha256(out/'readout-factor.pt') != metadata['factor_sha256']):
        raise ValueError('Lens/readout factor changed')
    dest = out/'linear-statistics'; dest.mkdir(exist_ok=True)
    checkpoint = dest/f'shard-{shard}.pt'; done = dest/f'shard-{shard}.json'
    if done.exists():
        if sha256(checkpoint) != read(done)['sha256']:
            raise ValueError('Accumulated matrices changed')
        return
    normalization = {}
    for arm in ARMS:
        train = dataset(out, arm, 'train')
        normalization[arm] = streaming_normalization(train['x'].numpy())
        del train
    old = torch.load(checkpoint, weights_only=True) if checkpoint.exists() else None
    if old:
        for arm in ARMS:
            for actual, expected in zip(normalization[arm], old['normalization'][arm]):
                if not torch.equal(actual, expected):
                    raise ValueError('Fixed coordinates changed during accumulation')
    stats = {arm: {k: v.cuda() if torch.is_tensor(v) else v for k, v in old['stats'][arm].items()}
             if old else statistics(len(normalization[arm][0]), 'cuda') for arm in ARMS}
    processed = old['processed'] if old else {}; del old
    hf, _ = original.load_model()
    fitted = jlens.JacobianLens.load(str(out/'lens.pt'))
    factor = torch.load(out/'readout-factor.pt', weights_only=True)['factor'].float().cuda()
    gain = hf.model.norm.weight.detach()
    layers = list(dict.fromkeys(m['layers']+[m['two_thirds']]))
    selected = [q for q in questions if q['split'] == 'train' and q['index'] % m['config']['shards'] == shard]
    def save():
        frozen.save_tensor(checkpoint, dict(processed=processed, normalization=normalization,
            stats={arm: {k: v.cpu() if torch.is_tensor(v) else v for k, v in values.items()} for arm, values in stats.items()}))
    for i, q in enumerate(selected):
        source = trajectory(out, q); checksum = sha256(source)
        if q['id'] in processed:
            if processed[q['id']] != checksum:
                raise ValueError('Previously accumulated trajectory changed')
            continue
        for row in read(source)['responses']:
            if not frozen.usable_response(row):
                continue
            t = len(row['token_ids']); n = len(q['prompt_token_ids'])
            ids = torch.tensor([q['prompt_token_ids']+row['token_ids']], device='cuda')
            captured = sampled_block_states(hf.model, ids, n, t, list(range(t)), layers)
            blocks = {l: h.cuda() for l, h in captured.items()}
            for arm in ARMS:
                values = linear_features(arm, blocks, m, fitted, factor, gain)
                mean, scale = normalization[arm]
                accumulate(stats[arm], values, float(row['score']), mean, scale,
                           trace_lambda=m['config']['linear_lambda'])
            del blocks, captured, values
        processed[q['id']] = checksum
        if (i+1) % m['config']['linear_checkpoint_questions'] == 0:
            save()
        print(f'Accumulated all tokens: {q["id"]}', flush=True)
    save()
    write(done, dict(sha256=sha256(checkpoint), questions=len(processed),
        transitions=stats[ARMS[0]]['transitions'], gamma=1., trace_lambda=m['config']['linear_lambda']))


def linear_fit(out, m, arm_index):
    from math_rl.large_linear import solve
    from math_rl.lstd import predict as linear_predict
    from math_rl.critic_probe import metrics
    arm = ARMS[arm_index]; dest = out/'linear-fits'/arm; dest.mkdir(parents=True, exist_ok=True)
    if (dest/'selection.json').exists():
        return
    merged = None; normalization = None
    for shard in range(m['config']['shards']):
        path = out/'linear-statistics'/f'shard-{shard}.pt'
        if sha256(path) != read(path.with_suffix('.json'))['sha256']:
            raise ValueError('Statistics checkpoint changed')
        record = torch.load(path, weights_only=True); current = record['stats'][arm]
        if merged is None:
            merged = current; normalization = record['normalization'][arm]
        else:
            if any(not torch.equal(a, b) for a, b in zip(normalization, record['normalization'][arm])):
                raise ValueError('Shards used different coordinates')
            for k in merged:
                merged[k] += current[k]
        del record
    val = dataset(out, arm, 'val'); mean, scale = normalization; selections = {}
    for head in LINEAR_HEADS:
        weights, checks = solve(merged, head, m['config']['linear_epsilon'])
        values = linear_predict(weights, val, mean, scale)
        path = dest/f'{head}.pt'
        frozen.save_tensor(path, dict(weights=weights, mean=mean, scale=scale, checks=checks))
        selections[head] = dict(path=str(path.relative_to(out)), sha256=sha256(path), checks=checks,
            validation=metrics(val['y'].numpy(), values))
    write(dest/'selection.json', dict(selected=selections,
        scope='Fixed epsilon=.01, gamma=1, lambda=.99. No validation tuning; raw values, no clipping or sigmoid. Matched all-token training for Ridge/LSTD.'))


def report(out, m, questions):
    from math_rl.critic_probe import make_head, predict, metrics
    import compare_jspace_probes as original
    if original.installed_lens() != read(out/'lens-metadata.json')['environment']:
        raise ValueError('Use the locked feature/fitting environment')
    selections = {(arm, head): read(out/'fits'/arm/head/'selection.json') for arm in ARMS for head in HEADS}
    linear_selections = {arm: read(out/'linear-fits'/arm/'selection.json') for arm in ARMS}
    # Every selection must exist before ANY test outcome is evaluated here.
    write(out/'selection-lock.json', dict(neural={f'{a}/{h}': s['selected'] for (a, h), s in selections.items()},
                                        linear=linear_selections))
    predictions = {}; results = {}; reference = None
    for arm in ARMS:
        data = dataset(out, arm, 'test')
        labels = {k: data[k] for k in ('y', 'position', 'response', 'question')}
        if reference is None:
            reference = labels
        if any(not torch.equal(reference[k], labels[k]) for k in labels):
            raise ValueError('Arms have unmatched evaluation states')
        for head in HEADS:
            seed_results = []; values = []
            for seed, selected in selections[arm, head]['selected'].items():
                path = out/selected['path']
                if sha256(path) != selected['sha256']:
                    raise ValueError('Selected head changed')
                record = torch.load(path, weights_only=True)
                model = make_head(head, data['x'].shape[1], m['config']['width']).cuda()
                model.load_state_dict(record['state'])
                p = predict(model, head, data, record['mean'].cuda(), record['scale'].cuda(), torch.device('cuda'))
                np.save(out/'fits'/arm/head/f'test-seed{seed}.npy', p)
                values.append(p); seed_results.append(metrics(data['y'].numpy(), p))
            key = f'{arm}/{head}'; predictions[key] = np.stack(values)
            results[key] = dict(by_seed=seed_results, means={k: float(np.mean([r[k] for r in seed_results]))
                for k in ('accuracy', 'brier', 'cross_entropy', 'ece')})
            position = data['position'].numpy()
            results[key]['by_prefix'] = {}
            for label, mask in [('question_only', position == 0), ('partial_answer', position > 0)]:
                if mask.any():
                    per_seed = [metrics(data['y'].numpy()[mask], p[mask]) for p in values]
                    results[key]['by_prefix'][label] = dict(examples=int(mask.sum()),
                        accuracy=float(np.mean([r['accuracy'] for r in per_seed])),
                        brier=float(np.mean([r['brier'] for r in per_seed])))
        from math_rl.lstd import predict as linear_predict
        for head in LINEAR_HEADS:
            chosen = linear_selections[arm]['selected'][head]; path = out/chosen['path']
            if sha256(path) != chosen['sha256']:
                raise ValueError('Locked linear critic changed')
            record = torch.load(path, weights_only=True)
            p = linear_predict(record['weights'], data, record['mean'], record['scale'])
            np.save(out/'linear-fits'/arm/f'{head}-test.npy', p)
            score = metrics(data['y'].numpy(), p); key = f'{arm}/{head}'
            predictions[key] = p[None, :]
            results[key] = dict(by_seed=[score], means={k: score[k] for k in ('accuracy', 'brier', 'cross_entropy', 'ece')},
                note='Deterministic closed-form fit, not a training seed. Brier and calibration use raw values; cross-entropy clips to probability bounds. Inspect outside_probability_range.')
    y = reference['y'].numpy(); qids = reference['question'].numpy()
    comparisons = {}
    for head in HEADS+LINEAR_HEADS:
        for baseline in ARMS[1:]:
            a = predictions[f'{ARMS[0]}/{head}']; b = predictions[f'{baseline}/{head}']
            comparisons[f'{head}/direct_minus_{baseline}'] = dict(
                brier=paired_errors(y, a, b, qids), accuracy=paired_errors(y, a, b, qids, accuracy=True))
    for arm in ARMS:
        a, b = predictions[f'{arm}/lstd-0.99'], predictions[f'{arm}/ridge']
        comparisons[f'{arm}/lstd_minus_ridge'] = dict(brier=paired_errors(y, a, b, qids),
                                                     accuracy=paired_errors(y, a, b, qids, accuracy=True))
    train_labels = torch.load(out/'packed'/ARMS[0]/'train-labels.pt', weights_only=True)['y']
    prior = float(train_labels.mean())
    constant = metrics(y, np.full(len(y), prior))
    counts = {}; audit = []
    for split in SPLITS:
        rows = []
        for q in questions:
            if q['split'] != split:
                continue
            answers = read(trajectory(out, q))['responses']; rows.extend(answers)
            # Deterministic first attempt on a question-hash sample, independent of reward.
            if int(digest(q['id'])[:8], 16) % 20 == 0:
                audit.append(dict(question=q['question'], reference=q['ground_truth'],
                    reference_solution=q['reference_solution'], split=split, id=q['id'], answer=answers[0]))
        counts[split] = dict(attempts=len(rows), retained=sum(frozen.usable_response(r) for r in rows),
            statuses=dict(Counter(r['verifier_status'] for r in rows)),
            answer_accuracy_all_attempts=sum(r['verifier_status'] == 'correct' for r in rows)/len(rows),
            truncation_rate=sum(r['finish_reason'] == 'length' for r in rows)/len(rows),
            mean_tokens=float(np.mean([len(r['token_ids']) for r in rows])))
    with (out/'audit.jsonl').open('w') as stream:
        for row in audit:
            stream.write(json.dumps(row)+'\n')
    review_html(out/'responses.html', audit)
    scope = ('Frozen base, new Numina cn_k12 question families; pretraining/paraphrase overlap unknown. '
        'Critic accuracy predicts eventual verifier correctness, not answer accuracy. Retained-answer conditional metrics. '
        'Head seeds share data. Two primary Brier comparisons: direct vs hidden_two_thirds under Ridge and LSTD(.99); '
        'use their 97.5% intervals (Bonferroni family 95%). Other comparisons/accuracy intervals descriptive. '
        'Ridge/LSTD use matched all-token raw sums; neural heads use sampled prefixes. '
        'Review audit.jsonl before claims. Offline LSTD only; no PPO or lower-cost claim.')
    timings = [read(path) for path in sorted((out/'timings').glob('*.json'))]
    costs = dict(completed_and_failed_invocations=timings,
        gpu_hours=sum(r['seconds'] for r in timings if r['gpu'])/3600,
        scope='Recorded worker wall time times one allocated GPU; includes caught failures/retries, excludes queue time and this reporting invocation. Hard kills may lack timing records; reconcile with sacct.')
    write(out/'summary.json', dict(scope=scope, results=results, comparisons=comparisons,
        counts=counts, constant=constant, costs=costs,
        dimensions={a: m['dimension']*(len(m['layers']) if a.endswith('concat') else 1) for a in ARMS}))
    lines = [scope, 'representation/head                         accuracy    Brier',
             f'constant                                     {constant["accuracy"]:.4f}    {constant["brier"]:.5f}']
    lines += [f'{key:44} {r["means"]["accuracy"]:.4f}    {r["means"]["brier"]:.5f}' for key, r in results.items()]
    lines += [f'{key}: {value}' for key, value in comparisons.items()]
    (out/'report.txt').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'repair-context', 'repair-verifier', 'screen', 'generate', 'lens', 'extract', 'pack', 'fit', 'linear-stats', 'linear-fit', 'report'), required=True)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(); args.out = args.out.resolve()
    if args.task < 0:
        parser.error('task must be nonnegative')
    started = time.perf_counter(); status = args.out/'status'/f'{args.stage}-{args.task}.json'
    invocation = f'{args.stage}-{args.task}-{time.time_ns()}'
    def finish(state, **extra):
        record = dict(state=state, stage=args.stage, task=args.task, seconds=time.perf_counter()-started,
            gpu=args.stage not in ('prepare', 'repair-context', 'repair-verifier', 'pack', 'linear-fit'), job=os.environ.get('SLURM_JOB_ID'), **extra)
        write(status, record)
        write(args.out/'timings'/f'{invocation}.json', record)
    write(status, dict(state='running', job=os.environ.get('SLURM_JOB_ID')))
    try:
        if args.stage == 'prepare':
            prepare(args)
        elif args.stage == 'repair-context':
            repair_context(args.out)
        elif args.stage == 'repair-verifier':
            repair_verifier(args.out)
        else:
            m, questions = validate(args.out)
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if args.stage in ('screen', 'generate'):
                if args.task >= m['config']['shards']:
                    raise ValueError('Shard out of bounds')
                generate(args.out, m, questions, args.stage, args.task)
            elif args.stage == 'lens':
                lens(args.out, m, questions)
            elif args.stage == 'extract':
                if args.task >= m['config']['shards']:
                    raise ValueError('Shard out of bounds')
                extract(args.out, m, questions, args.task)
            elif args.stage == 'pack':
                pack(args.out, m, questions, args.task)
            elif args.stage == 'fit':
                fit(args.out, m, args.task)
            elif args.stage == 'linear-stats':
                linear_stats(args.out, m, questions, args.task)
            elif args.stage == 'linear-fit':
                linear_fit(args.out, m, args.task)
            else:
                report(args.out, m, questions)
        finish('complete', environment={n: version(n) for n in ('torch', 'transformers', 'numpy')})
    except Exception as exc:
        finish('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
