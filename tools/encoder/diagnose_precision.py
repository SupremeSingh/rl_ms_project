"""Read-only precision/backend controls for the padding discrepancy; no training."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch
import compare as c
from diagnose_padding import diagnostic_inputs,inputs,errors

CASES={'fp32-eager':(torch.float32,'eager'),'bf16-eager':(torch.bfloat16,'eager'),
       'bf16-flash':(torch.bfloat16,'flash_attention_2'),'fp16-flash':(torch.float16,'flash_attention_2')}


def validate_indices(args,vocab_size,dense):
    ids=args['input_ids'];positions=args.get('position_ids')
    if int(ids.min())<0 or int(ids.max())>=vocab_size:raise ValueError('Token ID outside embedding vocabulary')
    if positions is not None:
        if int(positions.min())<0:raise ValueError('Negative position ID')
        if dense and int(positions.max())>=ids.shape[1]:
            raise ValueError('Eager rotary cache is sliced to sequence length; shifted indices exceed that slice')


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
    variants=[('base',0,'left',True,0),('left16_default',16,'left',False,0),
              ('left16_aligned',16,'left',True,0),('left64_aligned',64,'left',True,0),
              ('right16_aligned',16,'right',True,0)]
    if not dense:variants.insert(1,('shift16',0,'left',False,16))
    else:details['shift16']=dict(skipped='Unpadded shifted positions exceed eager rotary-cache slice; use left16_default versus left16_aligned')
    def forward(args):
        args=dict(args)
        if dense:args['attention_mask']=bidirectional_mask(args['attention_mask'],dtype)
        return model(**args).last_hidden_state.float().cpu()
    for name,n,side,aligned,shift in variants:
        args,selection=inputs(ids,tokenizer.pad_token_id,n,side,aligned,shift,device='cpu')
        validate_indices(args,model.config.vocab_size,dense)
        print('Running',str(dtype),backend,name,flush=True)
        args={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in args.items()}
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


def run_workers(src,out,reference):
    result={}
    for name in CASES:
        child=out.with_name(out.stem+'-'+name+'.json')
        cmd=[sys.executable,str(Path(__file__).resolve()),'--study',str(src),'--out',str(child),
             '--case',name,'--reference',str(reference)]
        status=subprocess.run(cmd,env=dict(os.environ,CUDA_LAUNCH_BLOCKING='1'),check=False)
        result[name]=c.read(child) if child.exists() else dict(error='Worker exited before writing report',exit_code=status.returncode)
        c.write_json(out,dict(cases=result,scope='Isolated diagnostic workers; not training approval'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--case',choices=list(CASES));p.add_argument('--reference',type=Path);a=p.parse_args()
    out=a.out.resolve();src=a.study.resolve()
    if out.exists() or out==src or src in out.parents:raise ValueError('Use a fresh report outside the study')
    if a.case and a.reference is None:p.error('--case requires --reference')
    out.parent.mkdir(parents=True,exist_ok=True)
    if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='1':raise ValueError('Unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE for FP32 reference')
    _,path=diagnostic_inputs(src)
    result=dict(manifest_sha256=c.sha256(src/'manifest.json'),model_path=str(path),cases={},
        scope='Diagnostic precision controls; no training, no package patches, no gate approval')
    if a.case:
        try:
            from official_train import native_amp
            native_amp()
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            torch.set_float32_matmul_precision('highest')
            name=a.case;dtype,backend=CASES[name]
            details,base=run_case(path,dtype,backend)
            if name=='fp32-eager':
                if a.reference.exists():raise FileExistsError(a.reference)
                torch.save(base,a.reference)
            elif a.reference.exists():details['base_vs_fp32']=errors(torch.load(a.reference,map_location='cpu',weights_only=True),base)
            else:details['fp32_reference_unavailable']=True
            c.write_json(out,details)
            print(name,json.dumps(details),flush=True)
        except Exception as exc:
            c.write_json(out,dict(error=repr(exc),traceback=traceback.format_exc()))
            # A device assertion poisons this process; never call CUDA cleanup or
            # attempt another case here. The parent starts a fresh worker.
            raise
        return
    reference=out.with_suffix('.reference.pt')
    if reference.exists():raise FileExistsError(reference)
    result['cases']=run_workers(src,out,reference)
    c.write_json(out,result)
    print('Report saved:',out)


if __name__=='__main__':main()
