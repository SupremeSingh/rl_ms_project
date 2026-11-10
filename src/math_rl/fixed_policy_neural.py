"""Two-GPU FSDP full-transformer regression on fixed-policy Monte Carlo returns.

FP32 master weights/Adam states are sharded; forwards use BF16. No actor updates.
The head is zero-initialized like our PPO baseline, but labels are shared MC
outcomes rather than GAE returns. Validation alone selects an epoch (including 0).
"""
from functools import partial
import os

import numpy as np
import torch

from .fixed_policy import ROOT, items, answer, features, read, sha256, write_json
from .value_audit import boundary_positions


def preaction_values(logits, prompt, response):
    if prompt<1 or response<1 or logits.shape[0]!=prompt+response:
        raise ValueError('Expected causal logits for prompt plus sampled actions')
    return logits[prompt-1:prompt+response-1]


def return_loss(logits,prompt,response,outcome,total_tokens,world_size=1):
    if outcome not in (0,1) or total_tokens<response:raise ValueError('Invalid shared return target')
    values=preaction_values(logits,prompt,response).float()
    # FSDP averages rank gradients; this recovers pooled-token MSE for the
    # complete global window, even when answer lengths differ across ranks.
    return (values-outcome).square().sum()*world_size/total_tokens


def equal_question_mse(rows):
    groups={}
    for q,error in rows:groups.setdefault(q,[]).append(error)
    if not groups or not all(np.isfinite(v).all() for v in groups.values()):
        raise ValueError('Invalid/empty validation values')
    return float(np.mean([np.mean(v) for v in groups.values()]))


def train(out,m):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp import StateDictType,FullStateDictConfig
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoConfig,AutoModelForTokenClassification
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from slice_critic.storage import model_identity
    from .ppo_baseline import zero_value_head,prepare_value_export
    if int(os.environ.get('WORLD_SIZE','0'))!=2:
        raise ValueError('Run transformer fit with torchrun --nproc_per_node=2')
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    # No automatic resume: partial optimizer trajectories must not be relabeled.
    if rank==0:
        (out/'.neural-fit-started').mkdir(exist_ok=False)
    dist.barrier()
    torch.manual_seed(m['neural']['seed'])
    config=AutoConfig.from_pretrained(ROOT/'models/qwen-math',local_files_only=True)
    config.num_labels=1;config.classifier_dropout=0.;config.attention_dropout=0.
    raw=AutoModelForTokenClassification.from_pretrained(ROOT/'models/qwen-math',config=config,
        local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='sdpa')
    zero_value_head(raw);prepare_value_export(raw)
    raw.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model=FSDP(raw,device_id=rank,use_orig_params=False,
        auto_wrap_policy=partial(transformer_auto_wrap_policy,transformer_layer_cls={Qwen2DecoderLayer}),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,buffer_dtype=torch.float32))
    optimizer=torch.optim.AdamW(model.parameters(),lr=m['neural']['lr'],weight_decay=0.)
    training=[(row,answer(out,row)) for row in items(out,m) if row['split']=='train']
    validation=[(row,answer(out,row),features(out,row)['starts']) for row in items(out,m) if row['split']=='val']
    if len(training)%m['neural']['answer_batch'] or m['neural']['answer_batch']%2:
        raise ValueError('Global answer windows must divide evenly across ranks')

    def forward(data):
        ids=torch.tensor([data['prompt_ids']+data['token_ids']],device=rank)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True).logits[0,:,0]

    @torch.no_grad()
    def evaluate():
        model.eval();rows=[]
        # Both ranks must perform the same number of FSDP forwards.
        if len(validation)%2:raise ValueError('Uneven validation forwards')
        for row,data,starts in validation[rank::2]:
            values=preaction_values(forward(data),len(data['prompt_ids']),len(data['token_ids']))
            positions=boundary_positions(starts,m['prefixes'])
            rows.append((row['id'],float((values[positions].float()-data['score']).square().mean())))
        gathered=[None]*2;dist.all_gather_object(gathered,rows)
        return equal_question_mse([r for part in gathered for r in part])

    def save():
        with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True,rank0_only=True)):
            state=model.state_dict()
        if rank==0:
            model._fsdp_wrapped_module.save_pretrained(out/'transformer',state_dict=state,safe_serialization=True)
        dist.barrier()

    best=evaluate();history=[dict(epoch=0,validation_brier=best)];best_epoch=0;save()
    for epoch in range(1,m['neural']['epochs']+1):
        model.train();order=np.random.default_rng(m['neural']['seed']+epoch).permutation(len(training))
        total_error=0.;seen=0
        batch=m['neural']['answer_batch']
        for start in range(0,len(order),batch):
            window=[training[i] for i in order[start:start+batch]]
            total_tokens=sum(len(data['token_ids']) for _,data in window)
            optimizer.zero_grad(set_to_none=True)
            for row,data in window[rank::2]:
                loss=return_loss(forward(data),len(data['prompt_ids']),len(data['token_ids']),
                                 data['score'],total_tokens,world_size=2)
                if not torch.isfinite(loss):raise ValueError('Nonfinite transformer loss')
                loss.backward();total_error+=float(loss.detach())*total_tokens/2;seen+=len(data['token_ids'])
            norm=model.clip_grad_norm_(1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite transformer gradient')
            optimizer.step()
            if rank==0:print(f'Transformer epoch {epoch}, answers {start+batch}/{len(order)}',flush=True)
        totals=torch.tensor([total_error,seen],dtype=torch.float64,device=rank)
        dist.all_reduce(totals)
        score=evaluate();history.append(dict(epoch=epoch,validation_brier=score,
            training_pooled_token_brier=float(totals[0]/totals[1])))
        if score<best:
            best,best_epoch=score,epoch;save()
    if rank==0:
        write_json(out/'neural.json',dict(manifest_sha256=sha256(out/'manifest.json'),
            model_identity=model_identity(out/'transformer'),best_epoch=best_epoch,history=history,
            targets='Same fixed-policy terminal returns as Ridge; pooled-token squared error.',
            scope='Full transformer fit; frozen actor; validation-selected epoch including epoch zero. Not a PPO run.'))
    dist.barrier();dist.destroy_process_group()
