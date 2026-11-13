"""Four-rank memory, negative-pool and gradient gate for the pinned official loss."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
import compare as c


def check_embedding_gradients(actual,reference,world):
    # Every rank evaluates the same global mean loss. Official AllGather.backward
    # sums the contributions from those ranks; leaf embedding grads are W times
    # the single-loss reference. DDP averages shared parameter grads afterward.
    torch.testing.assert_close(actual,reference*world,atol=1e-5,rtol=1e-4)


def check_ddp_parameters(fn,similarity,allq,alld,rank,world,device):
    from torch.nn.parallel import DistributedDataParallel
    torch.manual_seed(812)
    layer=torch.nn.Linear(allq.shape[1],12,bias=False).to(device)
    reference=torch.nn.Linear(allq.shape[1],12,bias=False).to(device)
    reference.load_state_dict(layer.state_dict())
    ddp=DistributedDataParallel(layer,device_ids=[device.index])
    local=torch.cat([allq[rank*8:(rank+1)*8],alld[rank*8:(rank+1)*8]])
    q,d=ddp(local).chunk(2)
    value=fn(q,d);value.backward()
    expected=torch.nn.functional.cross_entropy(similarity(reference(allq),reference(alld))*20,
                                               torch.arange(8*world,device=device))
    expected.backward()
    torch.testing.assert_close(value,expected,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(layer.weight.grad,reference.weight.grad,atol=1e-5,rtol=1e-4)
    return float((layer.weight.grad-reference.weight.grad).abs().max())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();m=c.validate(out)
    from official_train import native_amp
    native_amp()
    dist.init_process_group('nccl');rank=dist.get_rank();world=dist.get_world_size()
    if world!=4:raise ValueError('Exactly four ranks required')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    # Load loss from the sealed official checkout, not an unpinned downloaded copy.
    sys.path.insert(0,m['official'])
    from llm2vec.loss.HardNegativeNLLLoss import HardNegativeNLLLoss
    fn=HardNegativeNLLLoss()
    torch.manual_seed(99)
    allq=torch.randn(32,16,device=device);alld=torch.randn(32,16,device=device)
    q=allq[rank*8:(rank+1)*8].clone().requires_grad_();d=alld[rank*8:(rank+1)*8].clone().requires_grad_()
    shapes=[];original=fn.similarity_fct
    def similarity(x,y):shapes.append([len(x),len(y)]);return original(x,y)
    fn.similarity_fct=similarity
    actual=fn(q,d);actual.backward()
    rq=allq.clone().requires_grad_();rd=alld.clone().requires_grad_()
    ref=torch.nn.functional.cross_entropy(original(rq,rd)*20,torch.arange(32,device=device));ref.backward()
    torch.testing.assert_close(actual,ref,atol=1e-5,rtol=1e-5)
    check_embedding_gradients(q.grad,rq.grad[rank*8:(rank+1)*8],world)
    check_embedding_gradients(d.grad,rd.grad[rank*8:(rank+1)*8],world)
    if shapes!=[[32,32]]:raise ValueError('Official loss did not use 32 global candidates')
    ddp_error=check_ddp_parameters(fn,original,allq,alld,rank,world,device)
    if rank==0:print('Global loss, gather-gradient scaling and DDP parameter-gradient checks passed',flush=True)
    from llm2vec import LLM2Vec
    from peft import LoraConfig,get_peft_model
    context=c.checked(out,'profile.json')['context']
    wrapper=LLM2Vec.from_pretrained(m['model'],enable_bidirectional=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',pooling_mode='mean',max_length=context,skip_instruction=False,attention_dropout=.3)
    model=wrapper.model.to(device).eval().requires_grad_(False)
    # Check padding at a real prefix, comparing all token states, as well as pooling.
    tok=wrapper.tokenizer;ids=tok.encode('Question: calculate nine divided by three. Solution: divide nine by three.',add_special_tokens=False)
    with torch.no_grad():
        x=torch.tensor([ids],device=device);single=model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False).last_hidden_state
        pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        padded=torch.tensor([[pad]*16+ids],device=device);mask=torch.tensor([[0]*16+[1]*len(ids)],device=device)
        batch=model(input_ids=padded,attention_mask=mask,use_cache=False).last_hidden_state[:,-len(ids):]
        torch.testing.assert_close(single.float(),batch.float(),atol=.02,rtol=.02)
    # Actual two-view BF16/LoRA/checkpointed forwards and backward at the selected cap.
    model.requires_grad_(False)
    wrapper.model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    wrapper.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    wrapper.train();wrapper.model.config.use_cache=False
    vocab=model.config.vocab_size
    torch.manual_seed(100+rank)
    tokens=torch.randint(10,vocab-10,(8,context),device=device);mask=torch.ones_like(tokens)
    def forward():return wrapper(dict(input_ids=tokens,attention_mask=mask,embed_mask=mask))
    loss=fn(forward(),forward());loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in wrapper.parameters()):raise ValueError('Nonfinite distributed training gradients')
    memory=torch.tensor(torch.cuda.max_memory_allocated(),device=device,dtype=torch.float64);dist.all_reduce(memory,op=dist.ReduceOp.MAX)
    dist.barrier()
    if rank==0:c.seal(out,'distributed-check.json',dict(passed=True,world_size=world,local_batch=8,contrastive_batch=32,loss_shapes=shapes,embedding_gradient_scale=world,gradient_reference_match=True,ddp_parameter_gradient_reference_match=True,ddp_gradient_max_abs_error=ddp_error,padding_passed=True,context=context,peak_allocated_bytes=float(memory),loss=float(loss)))
    dist.destroy_process_group()

if __name__=='__main__':main()
