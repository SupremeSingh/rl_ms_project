"""Isolated benchmark driver: original update loop, disabled scientific evaluation."""
import importlib
import ray
import hydra
from .runner import ROOT

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

# Calling the imported decorated main directly makes Hydra interpret its
# relative config_path as a package named "configs". Anchor this launcher to
# the real repository directory, then call the original undecorated body.
@hydra.main(version_base=None,config_path=str(ROOT/'configs'),config_name='online')
def main(config):
    return module.main.__wrapped__(config)

if __name__=='__main__':main()
