"""Reusable frozen features and linear value prediction, independent of VERL."""
__version__ = "0.1.0"


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
    raise AttributeError(name)


__all__ = ['Critic', 'State', 'TrainingConfig', 'train_critic', 'CumulativeCritic']
