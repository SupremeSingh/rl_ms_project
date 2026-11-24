"""Reusable frozen features and linear value prediction, independent of VERL."""
__version__ = "0.2.0"


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
        from .cumulative import CumulativeCritic
        return CumulativeCritic
    if name == 'encode_hidden_rollout_features':
        from .online import frozen_prefixes
        return frozen_prefixes
    if name == 'inspect_artifact':
        from .artifacts import inspect_artifact
        return inspect_artifact
    if name == 'export_ppo_features':
        from .bridge import export_filter
        return export_filter
    if name in ('load_ppo_features', 'encode_rollout_features'):
        from .online import load_filter, bottleneck_prefixes
        return {'load_ppo_features': load_filter, 'encode_rollout_features': bottleneck_prefixes}[name]
    raise AttributeError(name)


__all__ = ['Critic', 'State', 'TrainingConfig', 'train_critic', 'CumulativeCritic',
           'encode_hidden_rollout_features', 'inspect_artifact', 'export_ppo_features', 'load_ppo_features', 'encode_rollout_features']
