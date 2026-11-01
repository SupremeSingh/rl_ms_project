"""Pinned VERL features: fixed frozen encoders and detached pre-action features."""
import numpy as np
import torch

from verl.single_controller.base.decorator import Dispatch, Execute, register
from verl.workers.fsdp_workers import ActorRolloutRefWorker

from slice_critic import encode_hidden_rollout_features as frozen_prefixes
from slice_critic import load_ppo_features as load_filter, encode_rollout_features as bottleneck_prefixes


class FeatureActorWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ALL_TO_ALL,execute_mode=Execute.RANK_ZERO)
    def neural_predict(self,features,mask,step,settings):
        if self.rank!=0 or not torch.cuda.is_available():
            raise RuntimeError('Neural critic must run on rank-zero GPU')
        if not hasattr(self,'_neural_service'):
            from math_rl.neural_critic import NeuralCriticService
            # compute_log_prob already verified/loaded this immutable filter and
            # offloaded the actor and frozen encoder before this blocking RPC.
            if not hasattr(self,'_critic_filter'): raise RuntimeError('Extract features before neural values')
            self._neural_service=NeuralCriticService(self._critic_filter,
                device=torch.device('cuda',torch.cuda.current_device()),**settings)
            self._neural_settings=dict(settings)
        if self._neural_settings!=dict(settings): raise ValueError('Neural settings changed during training')
        return self._neural_service.predict(features,mask,step)

    @register(dispatch_mode=Dispatch.ALL_TO_ALL,execute_mode=Execute.RANK_ZERO)
    def neural_fit(self,returns,mask,step,checkpoint=False):
        if self.rank!=0 or not hasattr(self,'_neural_service'):
            raise RuntimeError('Neural values must be predicted by rank zero before fitting')
        return self._neural_service.fit(returns,mask,step,checkpoint)

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
                if self.config.model.get('critic_feature_view','compressed')=='raw_concat':
                    from slice_critic.layers import sampled_block_states
                    from slice_critic.features import rms_coordinates
                    features=[]
                    layers=list(self.config.model.critic_bottleneck_layers)
                    for row,mask in zip(ids,masks,strict=True):
                        prompt=int(mask[:-width].sum());response=int(mask[-width:].sum())
                        tokens=row[mask.bool()].unsqueeze(0).to(torch.cuda.current_device())
                        blocks=sampled_block_states(self._critic_encoder,tokens,prompt,response,list(range(response)),layers)
                        x=torch.cat([rms_coordinates(blocks[l].float(),self.config.model.critic_bottleneck_rms_eps)
                                     for l in layers],1)
                        features.append(x.numpy().copy())
                else:
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
