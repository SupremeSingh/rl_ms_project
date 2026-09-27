"""Pinned VERL features: reuse actor forward or explicitly run a frozen encoder."""
import numpy as np
import torch

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from math_rl.online_critic import capture_prefixes, frozen_prefixes


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
        if self.config.model.get('critic_feature_mode', 'actor') == 'frozen':
            # Copy inputs before VERL moves/mutates its DataProto.
            ids = data.batch['input_ids'].detach().cpu().clone()
            masks = data.batch['attention_mask'].detach().cpu().clone()
            result = super().compute_log_prob(data)
            if not hasattr(self, '_critic_encoder'):
                from transformers import AutoModel
                self._critic_encoder = AutoModel.from_pretrained(
                    self.config.model.path, torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa', trust_remote_code=False).eval().requires_grad_(False)
            try:
                self._critic_encoder.to(torch.cuda.current_device())
                features = frozen_prefixes(self._critic_encoder, ids, masks, width)
            finally:
                self._critic_encoder.cpu()
                torch.cuda.empty_cache()
        else:
            with capture_prefixes(self.actor.actor_module, lengths) as features:
                result = super().compute_log_prob(data)
        # Ragged arrays avoid transferring padding states. VERL concatenates these
        # in the same DP order as old_log_probs, including after batch balancing.
        packed = np.empty(len(features), dtype=object)
        for i, states in enumerate(features):
            packed[i] = states
        result.non_tensor_batch['pretoken_features'] = packed
        return result
