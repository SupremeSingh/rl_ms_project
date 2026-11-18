"""Original training entry point with opt-in driver timing hooks."""
import importlib
import hydra
import ray
from .runner import ROOT
module=importlib.import_module('math_rl.main')
original_run=getattr(module.run,'_function',None)
if original_run is None:raise RuntimeError('Unsupported pinned Ray interface')

@ray.remote(num_cpus=1)
def curve_run(config):
    from .curve_runtime import install
    install()
    return original_run(config)

module.run=curve_run
@hydra.main(version_base=None,config_path=str(ROOT/'configs'),config_name='online')
def main(config):return module.main.__wrapped__(config)
if __name__=='__main__':main()
