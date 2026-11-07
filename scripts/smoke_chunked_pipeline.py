"""Six real v3 two-update checks; validation only, no accuracy ranking."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from math_rl import runner
from math_rl.provenance import write_json
from math_rl.ppo_baseline import audit_health
from smoke_slice_ppo import run as slice_smoke, verify_run


def run(args):
    if not 0 <= args.gae_lambda <= 1:
        raise ValueError('GAE lambda must be in [0,1]')
    out=args.out.absolute()
    # First run the published artifact through Ridge and LSTD PPO, two updates each.
    slice_smoke(args)
    checks=json.loads((out/'verification.json').read_text())['methods']
    try:
        for name,method in [('ridge-two_thirds','ridge'),('lstd-two_thirds','lstd'),('ppo','ppo'),('grpo','grpo')]:
            case=out/name
            write_json(out/'status.json',dict(state='running',method=name))
            cmd=runner.command(out,method,42,2,feature_mode='frozen',
                ppo_profile='higher_lr' if method=='ppo' else 'not_applicable',
                evaluation_split='val',case_name=name)
            cmd+=['experiment.chunking=math','experiment.critic_folds=1',f'algorithm.lam={args.gae_lambda}']
            cmd+=[f'experiment.chunk_fallback_tokens={args.chunk_fallback_tokens}']
            if method in ('ridge','lstd'):
                cmd+=['experiment.critic_statistics=cumulative','experiment.cumulative_epsilon=0.01',
                    'experiment.critic_lambda=0.9','actor_rollout_ref.model.critic_feature_layer=two_thirds',
                    'actor_rollout_ref.model.critic_encoder_precision=fp32']
            env=dict(os.environ,RAY_ADDRESS='local',MATH_RL_SEED='42')
            env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',None)
            with tempfile.TemporaryDirectory(prefix='ch-',dir='/tmp') as ray_tmp:
                env['RAY_TMPDIR']=ray_tmp
                with (out/f'{name}.log').open('w') as log:
                    subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=json.loads((case/'result.json').read_text())
            if result['updates']!=2 or result['answers']!=128 or result['evaluation_split']!='val':
                raise ValueError('Incomplete smoke actor updates')
            record=dict(passed=True,updates=2,answers=128)
            if method=='ppo':
                health=audit_health(case/'ppo-critic-audit',2)
                if not all(health['gates'][k] for k in ('complete','finite','nonzero_updates','zero_initial_values')):
                    raise ValueError('Transformer critic functional checks failed')
                record['health']=health
            if method in ('ridge','lstd'):
                import torch
                stats=torch.load(case/'linear-critic/statistics.pt',map_location='cpu',weights_only=True)
                if stats['folds']!=1 or len(stats['stats'])!=1 or stats['stats'][0]['answers']!=128:
                    raise ValueError('Expected one cumulative head retaining all answers')
                if not all(torch.isfinite(stats['stats'][0][k]).all() for k in ('a','b')):
                    raise ValueError('Nonfinite statistics')
            if method!='grpo':
                lines=[json.loads(v) for v in (case/'chunk-metrics.jsonl').read_text().splitlines()]
                if len(lines)!=2 or any(v['chunks']>v['response_tokens'] or v['chunks']<64 for v in lines):
                    raise ValueError('Invalid reasoning chunk accounting')
                record['chunk_counts']=[v['chunks'] for v in lines]
                if method in ('ridge','lstd'):
                    fit=[json.loads(v) for v in (case/'linear-critic/metrics.jsonl').read_text().splitlines()]
                    if [v['transitions'] for v in fit]!=record['chunk_counts']:
                        raise ValueError('Matrices did not count exactly one transition per chunk')
            checks[name]=record
        write_json(out/'pipeline-verification.json',dict(passed=True,chunking='math',critic_folds=1,
            chunk_fallback_tokens=args.chunk_fallback_tokens,
            epsilon=.01,gae_lambda=args.gae_lambda,actor_advantage='chunk GAE broadcast to tokens',methods=checks,
            scope='Real two-update GPU integration; not a ranking or verifier accuracy audit'))
        write_json(out/'status.json',dict(state='complete'))
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',error=str(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifact',type=Path)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gae-lambda',type=float,default=.95)
    parser.add_argument('--chunk-fallback-tokens',type=int,default=64)
    run(parser.parse_args())
