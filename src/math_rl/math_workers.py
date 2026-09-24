"""Small extension of the pinned VERL actor: reuse its old-log-probability forward."""
import numpy as np

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from math_rl.online_critic import capture_prefixes


class FeatureActorWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data):
        if (self.config.rollout.log_prob_micro_batch_size_per_gpu != 1
                or self.config.rollout.log_prob_use_dynamic_bsz
                or self.config.actor.ulysses_sequence_parallel_size != 1
                or not self.config.model.use_remove_padding):
            raise ValueError('Feature capture requires fixed microbatch=1, SP=1, remove_padding=True')
        width = data.batch['responses'].shape[1]
        attention = data.batch['attention_mask'].cpu()
        lengths = list(zip(attention[:, :-width].sum(-1).tolist(), attention[:, -width:].sum(-1).tolist()))
        with capture_prefixes(self.actor.actor_module, lengths) as features:
            result = super().compute_log_prob(data)
        # Ragged arrays avoid transferring padding states. VERL concatenates these
        # in the same DP order as old_log_probs, including after batch balancing.
        packed = np.empty(len(features), dtype=object)
        for i, states in enumerate(features):
            packed[i] = states
        result.non_tensor_batch['pretoken_features'] = packed
        return result
