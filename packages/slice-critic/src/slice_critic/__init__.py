"""Reusable frozen features and linear value prediction, independent of VERL."""
__version__ = "0.1.1"


def __getattr__(name):
    # Keep the verifier subprocess usable without importing Torch or Transformers.
    if name in ('Critic', 'State'):
        from .inference import Critic, State
        return {'Critic': Critic, 'State': State}[name]
    if name == 'TrainingConfig':
        from .config import TrainingConfig
        return TrainingConfig
    if name == 'train_critic':
        from .training import train_critic
        return train_critic
    if name == 'CumulativeCritic':
        from .cumulative import CumulativeLSTD
        return CumulativeLSTD
    if name == 'inspect_artifact':
        from .artifacts import inspect_artifact
        return inspect_artifact
    if name == 'export_ppo_features':
        from .legacy import export_filter
        return export_filter
    if name in ('load_ppo_features', 'encode_rollout_features'):
        from .online import load_filter, bottleneck_prefixes
        return {'load_ppo_features': load_filter, 'encode_rollout_features': bottleneck_prefixes}[name]
    raise AttributeError(name)


__all__ = ['Critic', 'State', 'TrainingConfig', 'train_critic', 'CumulativeCritic',
           'inspect_artifact', 'export_ppo_features', 'load_ppo_features', 'encode_rollout_features']
