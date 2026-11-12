"""Run inside the isolated LLM2Vec environment; no training or PPO changes."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'packages/slice-critic/src')]
import torch
from math_rl.encoder_features import load_encoder, prefix_inputs, block_features, attention_probe, require_bidirectional


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default=str(ROOT/'models/qwen-math'))
    p.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    p.add_argument('--attention',default='eager',choices=['eager','sdpa','flash_attention_2'])
    p.add_argument('--dtype',default='fp32',choices=['fp32','bf16'])
    p.add_argument('--batch-size',type=int,default=1,
                   help='Default unpadded prefixes; >1 requires batched invariance to pass')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(17)
    report=dict(passed=False,scope='Integration only; random compressor, no MNTP/SimCSE adaptation, no accuracy claim.',
        versions={n:metadata.version(n) for n in ['llm2vec','transformers','peft','torch']})
    try:
        started=time.perf_counter()
        dtype=torch.float32 if args.dtype=='fp32' else torch.bfloat16
        if args.attention=='flash_attention_2' and (args.device!='cuda' or dtype!=torch.bfloat16):
            raise ValueError('FlashAttention sanity requires CUDA and bf16')
        model,tokenizer=load_encoder(args.model,args.device,dtype,attention=args.attention)
        report.update(attention=args.attention,dtype=args.dtype,
                      feature_batch_size=args.batch_size,
                      execution='unpadded-prefix' if args.batch_size==1 else 'padded-batch')
        prompt=tokenizer.encode('Question: What is nine divided by three?\nAnswer:',add_special_tokens=False)
        answer=tokenizer.encode(' First identify the numbers. Then divide nine by three. The answer is three.',add_special_tokens=False)
        starts=[0,len(answer)//3,2*len(answer)//3]
        sequences=prefix_inputs(prompt,answer,starts)
        layers=[27,26,24,20,12] if len(model.layers)==28 else list(range(min(3,len(model.layers))))
        x=block_features(model,sequences,layers,tokenizer.pad_token_id,batch_size=args.batch_size)
        solo=block_features(model,sequences,layers,tokenizer.pad_token_id,batch_size=1)
        tolerance=2e-4 if dtype==torch.float32 else .02
        torch.testing.assert_close(x,solo,rtol=tolerance,atol=tolerance)
        # Optional diagnostic does not certify unsupported padded execution when
        # the selected execution contract is strictly unpadded batch_size=1.
        batched=block_features(model,sequences,layers,tokenizer.pad_token_id,batch_size=4)
        batch_difference=float((batched-solo).abs().max())
        try:
            torch.testing.assert_close(batched,solo,rtol=tolerance,atol=tolerance)
            batching_passed=True
        except AssertionError:
            batching_passed=False
        report.update(padded_batch_diagnostic=dict(passed=batching_passed,
            max_abs_difference=batch_difference,rtol=tolerance,atol=tolerance),
            padded_batching_approved=batching_passed)
        # Only unseen suffix changes; fixed supplied boundaries are not reselected.
        altered=answer[:starts[-1]]+[tokenizer.eos_token_id]*5
        y=block_features(model,prefix_inputs(prompt,altered,starts),layers,tokenizer.pad_token_id,
                         batch_size=args.batch_size)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        sensitivity=attention_probe(model,sequences[-1][:16])
        report.update(feature_shape=list(x.shape),layers_zero_based=layers,
            future_suffix_invariance=True,selected_execution_repeatability=True,
            actual_bidirectional_sensitivity=sensitivity,encoder_parameters=sum(v.numel() for v in model.parameters()))
        del model
        if args.device=='cuda':torch.cuda.empty_cache()
        causal,_=load_encoder(args.model,args.device,dtype,bidirectional=False,attention=args.attention)
        causal_sensitivity=attention_probe(causal,sequences[-1][:16])
        report['causal_control_sensitivity']=causal_sensitivity
        require_bidirectional(sensitivity,causal_sensitivity)
        del causal
        from slice_critic.bottleneck import Bottleneck
        from slice_critic.cumulative import CumulativeCritic
        compressor=Bottleneck(x.shape[1],768).eval().requires_grad_(False)
        z=compressor.features(x)
        critic=CumulativeCritic(['sanity'],method='ridge',folds=1,epsilon=.01)
        rewards=torch.tensor([[0.,0.,1.]])
        for step in (1,2):
            values,heads,metrics=critic.update([z],rewards,torch.ones_like(rewards),['sanity'],step)
        assert critic.stats[0]['answers']==2 and critic.stats[0]['transitions']==6
        assert len(heads)==1 and torch.isfinite(values).all()
        report.update(passed=True,random_compressor_shape=list(z.shape),matrix_dimension=769,
            cumulative_answers=2,cumulative_states=6,seconds=time.perf_counter()-started)
    except Exception as exc:
        report['error']=str(exc)
        raise
    finally:
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
