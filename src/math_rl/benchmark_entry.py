"""Isolated benchmark driver: original update loop, disabled scientific evaluation."""
import importlib
import ray

def no_evaluation(self,label):
    return dict(accuracy=0,questions=0,seconds=0,scope='Disabled in fixed-workload timing; not an accuracy result')

module=importlib.import_module('math_rl.main')
original_run=getattr(module.run,'_function',None)
if original_run is None:raise RuntimeError('Unsupported pinned Ray RemoteFunction interface')

@ray.remote(num_cpus=1)
def benchmark_run(config):
    from . import benchmark_runtime
    from .trainer import MathTrainer
    MathTrainer.evaluate_test=no_evaluation
    return original_run(config)

module.run=benchmark_run
if __name__=='__main__':module.main()
