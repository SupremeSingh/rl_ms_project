"""GPU adapters for value_audit. Generation and prediction use separate processes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .provenance import sha256, write_json
from .value_audit import (ARMS, ROOT, audit_questions, boundary_positions, checkpoint_paths,
                          read, stable_seed)


def tokenizer_identity(tokenizer):
    payload = dict(vocab=tokenizer.get_vocab(), eos=tokenizer.eos_token_id,
                   bos=tokenizer.bos_token_id, pad=tokenizer.pad_token_id)
    # VERL sets pad=eos when the base tokenizer has no pad token. That does not
    # change input token IDs; compare the effective tokenization contract.
    payload['pad'] = payload['eos'] if payload['pad'] is None else payload['pad']
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_tokenizer(actor):
    from transformers import AutoTokenizer
    base = AutoTokenizer.from_pretrained(ROOT/'models/qwen-math', local_files_only=True, trust_remote_code=False)
    fitted = AutoTokenizer.from_pretrained(actor, local_files_only=True, trust_remote_code=False)
    if tokenizer_identity(base) != tokenizer_identity(fitted):
        raise ValueError('Actor and original action-token vocabularies differ')
    return base


class Generator:
    def __init__(self, actor, m):
        from vllm import LLM
        self.tokenizer = load_tokenizer(actor)
        self.model = LLM(model=str(actor), tokenizer=str(ROOT/'models/qwen-math'),
                         dtype='bfloat16', tensor_parallel_size=1, gpu_memory_utilization=.8,
                         enforce_eager=True, max_model_len=m['max_model_len'], max_num_seqs=16,
                         enable_prefix_caching=True, seed=m['reference_seed'], trust_remote_code=False)
        self.m = m

    def generate(self, prompt, count, remaining, seed):
        from vllm import SamplingParams
        if not prompt or remaining < 1 or remaining > self.m['response_cap'] or len(prompt)+remaining > self.m['max_model_len']:
            raise ValueError('Invalid continuation budget')
        params = SamplingParams(n=count, temperature=1., top_p=1., top_k=-1,
                                max_tokens=remaining, seed=seed, detokenize=False,
                                stop_token_ids=[self.tokenizer.eos_token_id], ignore_eos=False)
        generated = self.model.generate([dict(prompt_token_ids=list(prompt))], params, use_tqdm=False)
        outputs = generated[0].outputs
        if len(outputs) != count:
            raise ValueError('Missing continuations')
        rows=[]
        for output in outputs:
            ids=list(output.token_ids)
            # vLLM versions may omit the stopping EOS from token_ids. Restore
            # only an actual EOS stop, never a length-truncated final action.
            if (output.finish_reason == 'stop' and output.stop_reason in (None, self.tokenizer.eos_token_id)
                    and (not ids or ids[-1] != self.tokenizer.eos_token_id)):
                ids.append(self.tokenizer.eos_token_id)
            if not ids or len(ids)>remaining:
                raise ValueError('Empty/out-of-budget generation')
            rows.append(dict(token_ids=ids, capped=output.finish_reason=='length'))
        return rows


def bank(out, m, phase, arm):
    folder=out/phase;target=folder/f'bank-{arm}.json'
    if target.exists():
        data=read(target)
        if data['manifest_sha256']!=sha256(out/'manifest.json'):raise ValueError('Stale bank shard')
        return
    _,actor,_,_=checkpoint_paths(out,m,arm,phase)
    questions=[q for q in audit_questions(out,m,phase) if q['bank_arm']==arm]
    tokenizer=load_tokenizer(actor)
    records=[];started=time.perf_counter()
    if questions:
        from slice_critic.config import completion_prompt
        from slice_critic.chunks import token_chunks
        generator=Generator(actor,m)
        for q in questions:
            prompt=tokenizer.encode(completion_prompt(q['question']),add_special_tokens=False)
            if not 0<len(prompt)<=512:raise ValueError('Question prompt exceeds training budget')
            answer=generator.generate(prompt,1,m['response_cap'],stable_seed(m['reference_seed'],phase,'bank',q['id']))[0]
            plan=token_chunks(tokenizer,answer['token_ids'],'math',32)
            positions=boundary_positions(plan.starts,m['budgets'][phase]['prefixes'])
            records.append(dict(id=q['id'],ground_truth=q['ground_truth'],bank_arm=arm,prompt_ids=prompt,
                                answer_ids=answer['token_ids'],capped=answer['capped'],positions=positions,
                                fallback_reason=plan.fallback_reason))
    write_json(target,dict(manifest_sha256=sha256(out/'manifest.json'),tokenizer_sha256=tokenizer_identity(tokenizer),
                           questions=records,seconds=time.perf_counter()-started))


def output_record(out, phase, rows, seconds, **extra):
    return dict(manifest_sha256=sha256(out/'manifest.json'),bank_sha256=sha256(out/phase/'bank.json'),
                states=rows,seconds=seconds,**extra)


@torch.inference_mode()
def last_prefix_value(model, prompt_ids, prefix_ids):
    """A token-classification critic's last available token values the NEXT action."""
    if not prompt_ids or model.config.num_labels != 1:
        raise ValueError('Expected nonempty prompt and scalar token critic')
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + prefix_ids], device=device)
    return float(model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       use_cache=False, return_dict=True).logits[0, -1, 0])


@torch.inference_mode()
def predict(out,m,phase,arm):
    if arm=='grpo':return  # Computed from independent prompt-only groups in MC process.
    folder=out/phase;target=folder/f'predictions-{arm}.json'
    if target.exists():
        data=read(target)
        if data['bank_sha256']!=sha256(folder/'bank.json'):raise ValueError('Stale predictions')
        return
    bank_data=read(folder/'bank.json');states=bank_data['states']
    _,actor,critic,head=checkpoint_paths(out,m,arm,phase)
    if tokenizer_identity(load_tokenizer(actor))!=bank_data['tokenizer_sha256']:
        raise ValueError('Bank token contract mismatch')
    started=time.perf_counter();rows=[]
    if arm=='ppo':
        from transformers import AutoModelForTokenClassification
        model, info=AutoModelForTokenClassification.from_pretrained(critic,local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
        if info['missing_keys'] or info['unexpected_keys'] or info.get('mismatched_keys'):
            raise ValueError(f'Incomplete PPO critic restore: {info}')
        if model.config.num_labels!=1:raise ValueError('Expected scalar token value head')
        model=model.to('cuda').eval().requires_grad_(False)
        for state in states:
            value=last_prefix_value(model,state['prompt_ids'],state['prefix_ids'])
            rows.append(dict(state_id=state['state_id'],prediction=value))
    else:
        from transformers import AutoModel
        from slice_critic.online import frozen_prefixes,bottleneck_prefixes,load_filter
        from .value_features import normalize_readout
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        model=AutoModel.from_pretrained(ROOT/'models/qwen-math',local_files_only=True,trust_remote_code=False,
            torch_dtype=torch.float32,attn_implementation='eager').to('cuda').eval().requires_grad_(False)
        weights=torch.load(head,map_location='cpu',weights_only=True)['heads'][0]['weights'].double()
        fitted=None
        if ARMS[arm]['layer']=='bottleneck':
            fitted=load_filter(out/'filters/sparse.pt',m['filter_sha256'],layers=m['layers'],eps=m['rms_eps'],
                               model_path=ROOT/'models/qwen-math')
        eos=load_tokenizer(actor).eos_token_id
        for state in states:
            # The existing causal feature API uses a dummy future action. Its
            # hidden state is never read and cannot affect the preceding prefix.
            ids=torch.tensor([state['prompt_ids']+state['prefix_ids']+[eos]])
            mask=torch.ones_like(ids)
            if fitted is not None:
                x=bottleneck_prefixes(model,ids,mask,1,fitted,m['layers'],m['rms_eps'])[0]
            else:
                x=frozen_prefixes(model,ids,mask,1,layer='two_thirds')[0]
                x=normalize_readout(x,'rms',model.config.rms_norm_eps)
            design=torch.cat([torch.as_tensor(x).double(),torch.ones(1,1,dtype=torch.float64)],1)
            value=float((design@weights).item())
            rows.append(dict(state_id=state['state_id'],prediction=value))
    if not all(np.isfinite(r['prediction']) for r in rows):raise ValueError('Nonfinite values')
    write_json(target,output_record(out,phase,rows,time.perf_counter()-started))


def grade(generator, questions, generated, prefix):
    from .ppo_reward import compute_score
    text=[generator.tokenizer.decode(prefix+r['token_ids'],skip_special_tokens=True) for r in generated]
    scores=compute_score(['numina_numeric']*len(text),text,[questions]*len(text))
    # Keep every outcome under the training reward rule. Parse/verification
    # timeouts count as zero; infrastructure errors raise and stop the stage.
    for item,score,response in zip(generated,scores,text,strict=True):
        item.update(score=score['score'],verifier_status=score['verifier_status'],response=response)
    return generated


def cached_samples(path,expected,generate):
    if path.exists():
        data=read(path)
        if data['identity']!=expected:raise ValueError('Cached sample identity changed')
    else:
        started=time.perf_counter()
        data=dict(identity=expected,samples=generate(),seconds=time.perf_counter()-started)
        write_json(path,data)
    if len(data['samples'])!=expected['count'] or any(r['score'] not in (0,1) for r in data['samples']):
        raise ValueError('Incomplete/invalid cached samples')
    return data


def monte_carlo(out,m,phase,arm,shard=0):
    folder=out/phase;target=folder/f'reference-{arm}-{shard}.json'
    if target.exists():
        record=read(target)
        if record['bank_sha256']!=sha256(folder/'bank.json'):raise ValueError('Stale reference')
        if arm!='grpo' or shard!=0 or (folder/'predictions-grpo.json').exists():return
    started=time.perf_counter();bank_data=read(folder/'bank.json')
    _,actor,_,_=checkpoint_paths(out,m,arm,phase)
    generator=Generator(actor,m)
    if tokenizer_identity(generator.tokenizer)!=bank_data['tokenizer_sha256']:
        raise ValueError('Bank tokenizer changed')
    cache=folder/'continuations'/arm;cache.mkdir(parents=True,exist_ok=True)
    header=dict(manifest_sha256=sha256(out/'manifest.json'),bank_sha256=sha256(folder/'bank.json'),arm=arm,phase=phase)
    baseline={};baseline_records=[];cached_seconds=0.
    if arm=='grpo' and shard==0:
        for q in bank_data['questions']:
            state=next(s for s in bank_data['states'] if s['question_id']==q['id'])
            seed=stable_seed(m['reference_seed'],phase,arm,'independent-group',q['id'])
            identity=dict(header,question_id=q['id'],count=m['grpo_group_size'],seed=seed,kind='grpo-prompt-baseline')
            sample=cached_samples(cache/f'group-{stable_seed(q["id"])}.json',identity,
                lambda:grade(generator,q['ground_truth'],generator.generate(state['prompt_ids'],m['grpo_group_size'],m['response_cap'],seed),[]))
            baseline[q['id']]=sum(v['score'] for v in sample['samples'])/m['grpo_group_size']
            baseline_records.append(sample);cached_seconds+=sample['seconds']
        predictions=[dict(state_id=s['state_id'],prediction=baseline[s['question_id']]) for s in bank_data['states']]
        write_json(folder/'predictions-grpo.json',output_record(out,phase,predictions,sum(s['seconds'] for s in baseline_records),
            groups=baseline,scope='Independent four-answer prompt group, not a prefix-conditioned value network.'))
    records=[];statuses=Counter();tokens=0;capped=0
    assigned=bank_data['states'][shard::m['budgets'][phase]['shards']]
    for state in assigned:
        samples=[];key=stable_seed(state['state_id'])
        total=m['budgets'][phase]['samples']
        for offset in range(0,total,m['reference_batch']):
            count=min(m['reference_batch'],total-offset)
            seed=stable_seed(m['reference_seed'],phase,arm,'reference',state['state_id'],offset)
            identity=dict(header,state_id=state['state_id'],offset=offset,count=count,seed=seed,kind='prefix-reference')
            sample=cached_samples(cache/f'state-{key}-{offset}.json',identity,
                lambda:grade(generator,state['ground_truth'],generator.generate(state['prompt_ids']+state['prefix_ids'],count,
                                   state['remaining_tokens'],seed),state['prefix_ids']))
            samples.extend(sample['samples']);cached_seconds+=sample['seconds']
        statuses.update(s['verifier_status'] for s in samples)
        tokens+=sum(len(s['token_ids']) for s in samples)
        capped+=sum(s['capped'] for s in samples)
        records.append(dict(state_id=state['state_id'],samples=len(samples),successes=sum(s['score'] for s in samples)))
        print(f'{arm}, shard {shard}: {len(records)}/{len(assigned)} states complete',flush=True)
    write_json(target,output_record(out,phase,records,time.perf_counter()-started,response_tokens=tokens,
        label_statuses=dict(statuses),capped_continuations=capped,
        baseline_response_tokens=sum(len(r['token_ids']) for group in baseline_records for r in group['samples']),
        cached_generation_and_verification_seconds=cached_seconds,
        scope='Wall seconds cover this invocation; cache timers retain work across restarts. '
              'Independent generation blocks; count includes capped and inconclusive-verifier zero rewards.'))
