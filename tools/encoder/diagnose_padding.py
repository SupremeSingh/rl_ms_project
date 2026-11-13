"""One-GPU read-only padding/position/kernel diagnosis; no training or package edits."""
import argparse
import inspect
import json
from pathlib import Path

import torch
import compare as c


def diagnostic_inputs(out):
    # A read-only model diagnostic needs the sealed manifest and base snapshot,
    # not training-question/actor paths that may have node-specific mount aliases.
    m=c.checked(out,'manifest.json')
    if m['protocol'] not in (c.PROTOCOL,'encoder-pooled-v2'):raise ValueError('Unexpected study protocol')
    recorded=Path(m['model']);resolved=recorded
    if not resolved.is_dir():
        if recorded.parts[-3:]!=(c.ROOT.name,'models','qwen-math'):
            raise ValueError('Cannot relocate an unrecognized base-model path')
        resolved=c.ROOT/'models/qwen-math'
    if c.model_identity(resolved)!=m['base_identity']:
        raise ValueError('Diagnostic base model differs from sealed study')
    return m,resolved


def errors(a,b):
    a=a.float();b=b.float()
    if a.shape!=b.shape:raise ValueError('Mismatched diagnostic shapes')
    close=torch.isclose(a,b,atol=.02,rtol=.02)
    return dict(max_abs=float((a-b).abs().max()),fraction_outside_tolerance=float((~close).float().mean()),
                relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),passed=bool(close.all()))


def inputs(ids,pad_id,padding=0,side='left',aligned=False,shift=0,device='cuda'):
    tokens=([pad_id]*padding+ids) if side=='left' else (ids+[pad_id]*padding)
    mask=([0]*padding+[1]*len(ids)) if side=='left' else ([1]*len(ids)+[0]*padding)
    args=dict(input_ids=torch.tensor([tokens],device=device),attention_mask=torch.tensor([mask],device=device),use_cache=False)
    if aligned:args['position_ids']=(args['attention_mask'].cumsum(-1)-1).clamp_min(0)+shift
    elif shift:args['position_ids']=torch.arange(len(tokens),device=device)[None,:]+shift
    selection=slice(padding,padding+len(ids)) if side=='left' else slice(0,len(ids))
    return args,selection


@torch.inference_mode()
def kernel_probe(length):
    from flash_attn import flash_attn_func,flash_attn_varlen_func
    torch.manual_seed(7821)
    q=torch.randn(1,length,12,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn_like(q);v=torch.randn_like(q)
    cu=torch.tensor([0,length],device='cuda',dtype=torch.int32)
    dense=flash_attn_func(q,k,v,dropout_p=0.,causal=False)
    variable=flash_attn_varlen_func(q[0],k[0],v[0],cu,cu,length,length,dropout_p=0.,causal=False)
    return errors(dense[0],variable)


@torch.inference_mode()
def diagnose(out):
    from official_train import native_amp
    native_amp();m,model_path=diagnostic_inputs(out)
    from llm2vec import LLM2Vec
    from importlib.metadata import version
    print('Verified diagnostic model:',model_path,flush=True)
    wrapper=LLM2Vec.from_pretrained(str(model_path),enable_bidirectional=True,torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',pooling_mode='mean',skip_instruction=False)
    model=wrapper.model.cuda().eval().requires_grad_(False)
    ids=wrapper.tokenizer.encode('Question: calculate nine divided by three. Solution: divide nine by three.',add_special_tokens=False)
    report=dict(versions={p:version(p) for p in ('llm2vec','transformers','flash-attn','torch')},
        model_config_attention=model.config._attn_implementation,
        attention_classes=[type(l.self_attn).__name__ for l in model.layers],
        is_causal=[l.self_attn.is_causal for l in model.layers],
        mask_source=inspect.getfile(type(model)._update_causal_mask),
        manifest_sha256=c.sha256(out/'manifest.json'),recorded_model_path=m['model'],resolved_model_path=str(model_path),
        validation_scope='Sealed manifest and identical frozen base model; not training-data/runtime approval',
        variants={},scope='Diagnostic only; does not approve padding or modify existing study')
    pad=wrapper.tokenizer.pad_token_id
    results={}
    variants=[('unpadded_default',0,'left',False,0),('unpadded_aligned',0,'left',True,0),
              ('unpadded_shift16',0,'left',False,16),('left16_default',16,'left',False,0),
              ('left16_aligned',16,'left',True,0),('left64_aligned',64,'left',True,0),
              ('right16_aligned',16,'right',True,0)]
    for name,n,side,aligned,shift in variants:
        args,selection=inputs(ids,pad,n,side,aligned,shift)
        captured={};hooks=[];first={}
        def pre(module,args,kwargs):
            for key in ('position_ids','attention_mask'):
                value=kwargs.get(key)
                first[key]=None if value is None else dict(shape=list(value.shape),values=value.detach().cpu().tolist())
        hooks.append(model.layers[0].self_attn.register_forward_pre_hook(pre,with_kwargs=True))
        for i,layer in enumerate(model.layers):
            def hook(module,args,output,i=i):
                h=output[0] if isinstance(output,tuple) else output
                captured[i]=h[0,selection].float().cpu()
            hooks.append(layer.register_forward_hook(hook))
        try:
            hidden=model(**args).last_hidden_state[0,selection].float().cpu()
            results[name]=(hidden,captured)
            item=dict(first_attention_inputs=first)
            if name!='unpadded_default':
                base,blocks=results['unpadded_default']
                item.update(final=errors(base,hidden),pooled=errors(base.mean(0),hidden.mean(0)),
                            by_block={str(i+1):errors(blocks[i],captured[i]) for i in captured})
            report['variants'][name]=item
            print(name,json.dumps(item.get('final',{})),flush=True)
        except Exception as exc:report['variants'][name]=dict(error=repr(exc))
        finally:
            for h in hooks:h.remove()
    try:report['flash_dense_vs_varlen']=kernel_probe(len(ids))
    except Exception as exc:report['flash_dense_vs_varlen']=dict(error=repr(exc))
    print('Flash dense versus varlen:',json.dumps(report['flash_dense_vs_varlen']),flush=True)
    print('Attention configuration:',report['model_config_attention'],report['attention_classes'][0],
          'is_causal:',report['is_causal'][0],flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    src=a.study.resolve();dest=a.out.resolve()
    if src==dest or src in dest.parents:raise ValueError('Write diagnostic outside original study')
    if dest.exists():raise FileExistsError(dest)
    report=diagnose(src)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print('Diagnostic saved:',dest)


if __name__=='__main__':main()
