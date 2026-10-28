"""Pinned VERL features: fixed frozen encoders and detached pre-action features."""
import numpy as np
import torch

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from slice_critic import encode_hidden_rollout_features as frozen_prefixes
from slice_critic import load_ppo_features as load_filter, encode_rollout_features as bottleneck_prefixes


class FeatureActorWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data):
        if (self.config.rollout.log_prob_micro_batch_size_per_gpu != 1
                or self.config.rollout.log_prob_use_dynamic_bsz
                or self.config.actor.ulysses_sequence_parallel_size != 1
                or not self.config.model.use_remove_padding):
            raise ValueError('Feature capture requires fixed microbatch=1, SP=1, remove_padding=True')
        width = data.batch['responses'].shape[1]
        if self.config.model.critic_feature_mode != 'frozen':
            raise ValueError('Linear critics require the frozen encoder')
        # Copy inputs before VERL moves/mutates its DataProto.
        ids = data.batch['input_ids'].detach().cpu().clone()
        masks = data.batch['attention_mask'].detach().cpu().clone()
        result = super().compute_log_prob(data)
        if not hasattr(self, '_critic_encoder'):
            from transformers import AutoModel
            fp32 = self.config.model.get('critic_encoder_precision', 'bf16') == 'fp32'
            if fp32:
                import os
                if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
                    raise RuntimeError('Unset TF32 override for fixed FP32 features')
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.set_float32_matmul_precision('highest')
            self._critic_encoder = AutoModel.from_pretrained(
                self.config.model.path, torch_dtype=torch.float32 if fp32 else torch.bfloat16,
                attn_implementation='eager' if fp32 else 'sdpa', trust_remote_code=False).eval().requires_grad_(False)
        epoch = 0
        refresh_seconds = 0.
        self._critic_encoder_epoch = epoch
        try:
            self._critic_encoder.to(torch.cuda.current_device())
            layer = self.config.model.get('critic_feature_layer', 'final')
            if layer == 'bottleneck':
                if not hasattr(self, '_critic_filter'):
                    self._critic_filter = load_filter(self.config.model.critic_bottleneck_path,
                        self.config.model.critic_bottleneck_sha256,
                        layers=list(self.config.model.critic_bottleneck_layers),
                        eps=self.config.model.critic_bottleneck_rms_eps,
                        model_path=self.config.model.path)
                features = bottleneck_prefixes(self._critic_encoder, ids, masks, width,
                    self._critic_filter, list(self.config.model.critic_bottleneck_layers),
                    self.config.model.critic_bottleneck_rms_eps)
            else:
                features = frozen_prefixes(self._critic_encoder, ids, masks, width, layer=layer)
        finally:
            self._critic_encoder.cpu()
            torch.cuda.empty_cache()
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
