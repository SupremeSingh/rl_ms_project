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
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);p.add_argument('--study',type=Path);p.add_argument('--report',type=Path);a=p.parse_args()
    if a.study:
        if a.report is None:p.error('--study needs a fresh --report outside the study')
        from diagnose_padding import diagnostic_inputs
        out=a.study.resolve();m,path=diagnostic_inputs(out);m=dict(m,model=str(path),official=str(c.ROOT/'third_party/llm2vec'))
        if a.report.exists() or a.report.resolve()==out or out in a.report.resolve().parents:raise ValueError('Use a fresh report outside study')
        if c.subprocess.check_output(['git','-C',m['official'],'rev-parse','HEAD'],text=True).strip()!=c.study.OFFICIAL or c.subprocess.check_output(['git','-C',m['official'],'status','--porcelain'],text=True).strip():raise ValueError('Clean pinned official checkout required')
    else:
        if a.out is None:p.error('--out or --study required')
        out=a.out.resolve();m=c.validate(out)
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
    from llm2vec import LLM2Vec as OfficialLLM2Vec
    from unpadded_adapter import make_unpadded
    LLM2Vec=make_unpadded(OfficialLLM2Vec)
    from peft import LoraConfig,get_peft_model
    context=c.checked(out,'profile.json')['context']
    wrapper=LLM2Vec.from_pretrained(m['model'],enable_bidirectional=True,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',pooling_mode='mean',max_length=context,skip_instruction=False,attention_dropout=.3)
    model=wrapper.model.to(device).eval().requires_grad_(False)
    # Exact execution path: eight logical examples, eight singleton forwards.
    tok=wrapper.tokenizer
    texts=['Question: nine divided by three? Solution: '+('divide nine by three. '*i) for i in range(1,9)]
    def device_batch(batch):
        return {'sequences':[{k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in seq.items()} for seq in batch['sequences']]}
    with torch.no_grad():
        grouped=wrapper(device_batch(wrapper.tokenize(texts)))
        single=torch.cat([wrapper(device_batch(wrapper.tokenize([text]))) for text in texts])
        torch.testing.assert_close(single.float(),grouped.float(),atol=.02,rtol=.02)
    if rank==0:print('Unpadded multi-example embeddings match individual prefix encodings',flush=True)
    # Actual two-view BF16/LoRA/checkpointed forwards and backward at the selected cap.
    model.requires_grad_(False)
    wrapper.model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    wrapper.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    wrapper.train();wrapper.model.config.use_cache=False
    vocab=model.config.vocab_size
    torch.manual_seed(100+rank)
    tokens=torch.randint(10,vocab-10,(8,context),device=device);mask=torch.ones_like(tokens)
    def forward():return wrapper({'sequences':[dict(input_ids=tokens[i:i+1],attention_mask=mask[i:i+1],embed_mask=mask[i:i+1]) for i in range(8)]})
    loss=fn(forward(),forward());loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in wrapper.parameters()):raise ValueError('Nonfinite distributed training gradients')
    memory=torch.tensor(torch.cuda.max_memory_allocated(),device=device,dtype=torch.float64);dist.all_reduce(memory,op=dist.ReduceOp.MAX)
    dist.barrier()
    if rank==0:
        report=dict(passed=True,world_size=world,local_batch=8,contrastive_batch=32,loss_shapes=shapes,embedding_gradient_scale=world,gradient_reference_match=True,ddp_parameter_gradient_reference_match=True,ddp_gradient_max_abs_error=ddp_error,execution='singleton-no-padding-joint-contrastive',unpadded_execution_passed=True,logical_batch_per_rank=8,encoder_forward_batch=1,context=context,peak_allocated_bytes=float(memory),loss=float(loss))
        if a.study:
            report['scope']='Standalone unpadded execution gate, not full study verification'
            c.write_json(a.report,report)
        else:c.seal(out,'distributed-check.json',report)
    dist.destroy_process_group()

if __name__=='__main__':main()
