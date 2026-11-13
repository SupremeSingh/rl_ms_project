"""Read-only precision/backend controls for the padding discrepancy; no training."""
import argparse
import gc
import json
import os
from pathlib import Path

import torch
import compare as c
from diagnose_padding import diagnostic_inputs,inputs,errors


def bidirectional_mask(mask,dtype):
    """Explicit additive all-to-all attention over real tokens, masking pad keys."""
    n=mask.shape[1]
    result=torch.zeros(mask.shape[0],1,n,n,device=mask.device,dtype=dtype)
    return result.masked_fill(mask[:,None,None,:]==0,torch.finfo(dtype).min)


@torch.inference_mode()
def run_case(path,dtype,backend):
    from llm2vec import LLM2Vec
    wrapper=LLM2Vec.from_pretrained(str(path),enable_bidirectional=True,torch_dtype=dtype,
        attn_implementation=backend,pooling_mode='mean',skip_instruction=False)
    model=wrapper.model.cuda().eval().requires_grad_(False)
    tokenizer=wrapper.tokenizer
    ids=tokenizer.encode('Question: calculate nine divided by three. Solution: divide nine by three.',add_special_tokens=False)
    dense=backend=='eager'
    outputs={};details={}
    variants=[('base',0,'left',True,0),('shift16',0,'left',False,16),
              ('left16_aligned',16,'left',True,0),('left64_aligned',64,'left',True,0),
              ('right16_aligned',16,'right',True,0)]
    def forward(args):
        args=dict(args)
        if dense:args['attention_mask']=bidirectional_mask(args['attention_mask'],dtype)
        return model(**args).last_hidden_state.float().cpu()
    for name,n,side,aligned,shift in variants:
        args,selection=inputs(ids,tokenizer.pad_token_id,n,side,aligned,shift)
        captured={};hooks=[]
        for key in ('q_proj','k_proj','v_proj'):
            def capture(module,args,value,key=key):captured[key]=value[0,selection].float().cpu()
            hooks.append(getattr(model.layers[0].self_attn,key).register_forward_hook(capture))
        try:output=forward(args)[0,selection]
        finally:
            for h in hooks:h.remove()
        outputs[name]=(output,captured)
        if name!='base':
            original,projections=outputs['base']
            details[name]=dict(hidden=errors(original,output),pooled=errors(original.mean(0),output.mean(0)),
                first_layer_projections={k:errors(projections[k],captured[k]) for k in captured})
    args,_=inputs(ids,tokenizer.pad_token_id,aligned=True)
    mutated=dict(args);mutated['input_ids']=args['input_ids'].clone()
    mutated['input_ids'][0,-1]=(int(mutated['input_ids'][0,-1])+7)%model.config.vocab_size
    sensitivity=float((forward(args)[0,0]-forward(mutated)[0,0]).abs().max())
    if sensitivity<=1e-5:raise ValueError('Precision control failed to demonstrate actual bidirectional attention')
    details['bidirectional_sensitivity']=sensitivity
    base=outputs['base'][0]
    del wrapper,model,outputs
    gc.collect();torch.cuda.empty_cache()
    return details,base


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();src=a.study.resolve()
    if out.exists() or out==src or src in out.parents:raise ValueError('Use a fresh report outside the study')
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='1':raise ValueError('Unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE for FP32 reference')
    from official_train import native_amp
    native_amp();_,path=diagnostic_inputs(src)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    result=dict(manifest_sha256=c.sha256(src/'manifest.json'),model_path=str(path),cases={},
        scope='Diagnostic precision controls; no training, no package patches, no gate approval')
    reference=None
    for name,dtype,backend in [('fp32-eager',torch.float32,'eager'),('bf16-eager',torch.bfloat16,'eager'),
                                ('bf16-flash',torch.bfloat16,'flash_attention_2'),('fp16-flash',torch.float16,'flash_attention_2')]:
        try:
            details,base=run_case(path,dtype,backend)
            if name=='fp32-eager':reference=base
            elif reference is not None:details['base_vs_fp32']=errors(reference,base)
            else:details['fp32_reference_unavailable']=True
            result['cases'][name]=details
            print(name,json.dumps(details),flush=True)
        except Exception as exc:
            result['cases'][name]=dict(error=repr(exc));print(name,repr(exc),flush=True)
            gc.collect();torch.cuda.empty_cache()
        out.parent.mkdir(parents=True,exist_ok=True)
        # Preserve partial diagnostics if a later control fails or runs out of time.
        c.write_json(out,result)
    print('Report saved:',out)


if __name__=='__main__':main()
