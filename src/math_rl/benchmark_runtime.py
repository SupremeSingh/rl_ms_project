"""Opt-in fixed-action replay for timing ONLY; never used for scientific training."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from . import ppo_runtime  # retain original deterministic worker initialization


def replay_outputs(inputs,n,workload):
    output=[]
    for item in inputs:
        key=','.join(map(str,item['prompt_token_ids']))
        answers=workload[key]
        if n!=4 or len(answers)!=4: raise ValueError('Benchmark requires exactly four fixed actions per prompt')
        output.append(SimpleNamespace(outputs=[SimpleNamespace(token_ids=list(a),logprobs=None) for a in answers]))
    return output


def install():
    path=os.environ.get('MATH_RL_BENCH_WORKLOAD')
    if not path: raise RuntimeError('Benchmark runtime cannot run without explicit workload')
    workload=json.loads(Path(path).read_text())['answers']
    # Preserve dispatch metadata while making each worker phase finish GPU work
    # before returning its timing to the pinned driver loop.
    from functools import wraps
    from verl.workers.fsdp_workers import ActorRolloutRefWorker,CriticWorker
    def synchronized(function):
        @wraps(function)
        def call(self,*args,**kwargs):
            torch.cuda.synchronize()
            result=function(self,*args,**kwargs)
            torch.cuda.synchronize()
            return result
        return call
    for cls,names in [(ActorRolloutRefWorker,['compute_log_prob','update_actor']),
                      (CriticWorker,['compute_values','update_critic'])]:
        for name in names:setattr(cls,name,synchronized(getattr(cls,name)))
    from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout
    original=vLLMRollout.generate_sequences
    def replay(self,prompts,**kwargs):
        if prompts.meta_info.get('validate') or not prompts.meta_info.get('do_sample',True): raise ValueError('No validation/generation in fixed-workload timing')
        engine=self.inference_engine;generate=engine.generate
        def fixed(*args,**kw):
            if args: raise ValueError('Unexpected pinned rollout interface')
            return replay_outputs(kw['prompts'],kw['sampling_params'].n,workload)
        engine.generate=fixed
        try:return original(self,prompts,**kwargs)
        finally:engine.generate=generate
    vLLMRollout.generate_sequences=replay
    from slice_critic import CumulativeCritic
    original_init=CumulativeCritic.__init__
    def init(self,*args,**kwargs):
        original_init(self,*args,**kwargs)
        late=os.environ.get('MATH_RL_BENCH_STATISTICS')
        if late:
            saved=torch.load(late,map_location='cpu',weights_only=True)
            if saved['method']!='ridge' or saved['folds']!=1 or saved['epsilon']!=self.epsilon:raise ValueError('Incompatible late statistics')
            self.stats=saved['stats'];self.epoch=0
    CumulativeCritic.__init__=init


install()
