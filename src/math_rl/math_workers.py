"""Pinned VERL features: reuse actor forward or explicitly run a frozen encoder."""
import time
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
            from math_rl.cumulative_lstd import encoder_epoch
            step = int(data.meta_info.get('critic_update_step', 1))
            interval = int(data.meta_info.get('critic_refresh_interval', 0))
            epoch = encoder_epoch(step, interval)
            refresh_seconds = 0.
            if epoch != getattr(self, '_critic_encoder_epoch', 0):
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType
                from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
                from math_rl.critic_layers import copy_backbone_state
                started = time.perf_counter()
                module = self.actor.actor_module
                if not isinstance(module, FSDP):
                    raise ValueError('Encoder refresh requires pinned FSDP1')
                # All DP ranks participate; each CPU encoder receives the same full state.
                load_fsdp_model_to_gpu(module)
                try:
                    with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT,
                            FullStateDictConfig(offload_to_cpu=True, rank0_only=False)):
                        state = module.state_dict()
                    copy_backbone_state(self._critic_encoder, state)
                    del state
                finally:
                    offload_fsdp_model_to_cpu(module)
                refresh_seconds = time.perf_counter() - started
            self._critic_encoder_epoch = epoch
            try:
                self._critic_encoder.to(torch.cuda.current_device())
                features = frozen_prefixes(self._critic_encoder, ids, masks, width,
                    layer=self.config.model.get('critic_feature_layer', 'final'))
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
        if self.config.model.get('critic_feature_mode', 'actor') == 'frozen':
            result.non_tensor_batch['critic_encoder_epoch'] = np.full(len(features), epoch, dtype=np.int64)
            result.meta_info.setdefault('metrics', {})['critic/encoder_refresh_seconds'] = refresh_seconds
        return result
