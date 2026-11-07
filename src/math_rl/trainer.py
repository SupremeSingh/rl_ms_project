"""PPO/GRPO comparison using the pinned VERL loop, losses, GAE and checkpoints.

For linear critics only, the scoped advantage hook supplies learned values
where VERL normally reads transformer-critic values. No transformer critic is
allocated or updated in those runs. The pinned loop remains otherwise unchanged.
"""
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from verl.trainer.ppo import ray_trainer

from math_rl.completion_dataset import CompletionDataset
from math_rl.provenance import write_json


class MathTrainer(ray_trainer.RayPPOTrainer):
    def _validate_config(self):
        # The parent infers use_critic from GAE. Linear PPO supplies values itself;
        # disable the transformer before either config checks or worker creation.
        if self.config.experiment.method in ('lstd', 'ridge'):
            self.use_critic = False
        super()._validate_config()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = self.config.experiment.method
        self.linear = self.method in ('lstd', 'ridge')
        self.output = Path(self.config.trainer.default_local_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.completed_updates = 0

    def fit(self):
        original = ray_trainer.compute_advantage
        model_config = getattr(getattr(self.config, 'actor_rollout_ref', None), 'model', None)
        feature_mode = getattr(model_config, 'critic_feature_mode', 'actor')
        cumulative = None
        if self.linear:
            from datasets import Dataset
            from slice_critic import CumulativeCritic
            rows = Dataset.from_parquet(str(self.config.data.train_files))
            questions = [str(row['extra_info']['prompt_id']) for row in rows]
            cumulative = CumulativeCritic(questions, self.config.data.seed,
                self.config.experiment.cumulative_epsilon, self.config.experiment.critic_lambda,
                method=self.config.experiment.method, folds=self.config.experiment.get("critic_folds", 2))

        chunk_mode = self.config.experiment.get('chunking', 'token')
        fallback_tokens = self.config.experiment.get('chunk_fallback_tokens', 32)

        def advantages(data, *args, **kwargs):
            mask = data.batch['response_mask']
            plans = None
            if chunk_mode != 'token' and self.method != 'grpo':
                from math_rl.chunked import plans_for_batch, compress_rewards, chunk_credit
                plans = plans_for_batch(self.tokenizer, data.batch['responses'], mask, chunk_mode, fallback_tokens)
            if self.linear:
                device = data.batch['token_level_rewards'].device
                features = data.non_tensor_batch.pop('pretoken_features')
                questions = [str(info['prompt_id']) for info in data.non_tensor_batch['extra_info']]
                fit_rewards, fit_mask = data.batch['token_level_rewards'], mask
                if plans is not None:
                    if any(len(f) != len(p.starts) for f, p in zip(features, plans, strict=True)):
                        raise ValueError('Expected one frozen feature vector per chunk start')
                    fit_rewards, fit_mask = compress_rewards(fit_rewards, plans)
                if cumulative is not None:
                    epoch = 0
                    versions = data.non_tensor_batch.pop('critic_encoder_epoch')
                    if any(int(v) != epoch for v in versions):
                        raise RuntimeError('Workers returned features from a different encoder epoch')
                    values, heads, metrics = cumulative.update(features,
                        fit_rewards, fit_mask,
                        questions, self.global_steps, epoch)
                metrics['feature_mode'] = feature_mode
                if plans is None:
                    data.batch['values'] = values.to(device)
                else:
                    expanded = data.batch['token_level_rewards'].new_zeros(mask.shape)
                    for i, plan in enumerate(plans):
                        expanded[i, :plan.tokens] = plan.expand(values[i, :len(plan.starts)]).to(device)
                    data.batch['values'] = expanded
                metrics.update(chunking=chunk_mode, response_tokens=int(mask.sum()),
                               state_unit='chunk' if plans is not None else 'token')
                folder = self.output / 'linear-critic'
                folder.mkdir(exist_ok=True)
                torch.save(dict(heads=heads, step=self.global_steps, method=self.method,
                    feature_policy=feature_mode,
                    fitting_mode='cumulative_raw_sum'),
                    folder / f'{self.global_steps}.pt')
                if cumulative is not None:
                    temporary = folder / 'statistics.tmp'
                    torch.save(cumulative.state_dict(), temporary)
                    temporary.replace(folder / 'statistics.pt')
                with (folder / 'metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(metrics, step=self.global_steps)) + '\n')
            if plans is not None:
                from verl.utils.torch_functional import masked_whiten
                adv, returns, boundaries, values = chunk_credit(data.batch['values'],
                    data.batch['token_level_rewards'], mask, plans, gae_lambda=self.config.algorithm.lam)
                data.batch['advantages'] = masked_whiten(adv, mask) * mask
                data.batch['returns'] = returns
                data.batch['values'] = values
                data.batch['chunk_start_mask'] = boundaries
                result = data
                with (self.output / 'chunk-metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(step=self.global_steps, chunking=chunk_mode,
                        chunk_fallback_tokens=fallback_tokens,
                        chunks=sum(len(p.starts) for p in plans), response_tokens=int(mask.sum()),
                        segmentation_fallback_answers=sum(p.fallback_reason is not None for p in plans),
                        segmentation_fallback_reasons={reason:sum(p.fallback_reason == reason for p in plans)
                            for reason in sorted({p.fallback_reason for p in plans if p.fallback_reason})},
                        advantage='chunk_gae', gae_lambda=float(self.config.algorithm.lam))) + '\n')
            else:
                result = original(data, *args, **kwargs)
            if not all(torch.isfinite(result.batch[k]).all() for k in ('advantages', 'returns')):
                raise RuntimeError('Nonfinite advantages or returns')
            return result

        # A single trainer runs in its own Ray driver process. Restore the module
        # function even on failure; chunked credit bypasses token GAE; PPO policy loss is unchanged.
        ray_trainer.compute_advantage = advantages
        try:
            super().fit()
        finally:
            ray_trainer.compute_advantage = original
        self.completed_updates = self.global_steps - 1
        if self.completed_updates != self.total_training_steps:
            raise RuntimeError('Training ended before the requested update budget')

    def evaluate_test(self, label):
        """Fixed greedy test; scores never enter optimization or model selection."""
        from verl.utils.dataset.rl_dataset import collate_fn
        dataset = CompletionDataset(self.config.experiment.test_file, self.tokenizer, self.config.data)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        old_loader, old_reward = self.val_dataloader, self.val_reward_fn
        old_dir = self.config.trainer.validation_data_dir
        rows = []

        def record(batch, return_dict=False):
            result = old_reward(batch, return_dict=True)
            scores = result['reward_tensor'].sum(-1).cpu().tolist()
            extra = result.get('reward_extra_info', {})
            width = batch.batch['responses'].shape[1]
            counts = batch.batch['attention_mask'][:, -width:].sum(-1).long().cpu().tolist()
            for i, (score, count) in enumerate(zip(scores, counts, strict=True)):
                info = batch.non_tensor_batch['extra_info'][i]
                tokens = batch.batch['responses'][i, :count].cpu().tolist()
                rows.append(dict(id=info['prompt_id'], level=int(info['level']),
                    ground_truth=batch.non_tensor_batch['reward_model'][i]['ground_truth'],
                    response=self.tokenizer.decode(tokens, skip_special_tokens=True), score=float(score),
                    response_tokens=count,
                    capped=count == self.config.data.max_response_length and bool(tokens)
                        and tokens[-1] != self.tokenizer.eos_token_id,
                    verifier_status=str(extra['verifier_status'][i]), extracted=str(extra['extracted'][i])))
            return result if return_dict else result['reward_tensor']

        started = time.perf_counter()
        self.val_dataloader, self.val_reward_fn = loader, record
        self.config.trainer.validation_data_dir = None
        self.global_steps = self.completed_updates
        try:
            self._validate()
        finally:
            self.val_dataloader, self.val_reward_fn = old_loader, old_reward
            self.config.trainer.validation_data_dir = old_dir
        expected = [r['extra_info']['prompt_id'] for r in dataset.rows]
        if [r['id'] for r in rows] != expected:
            raise RuntimeError('Test output order/count differs from the frozen question list')
        with (self.output / f'{label}-test.jsonl').open('x') as handle:
            for row in rows:
                handle.write(json.dumps(row) + '\n')
        result = dict(accuracy=sum(r['score'] for r in rows) / len(rows), questions=len(rows),
                      truncation_rate=sum(r['capped'] for r in rows) / len(rows),
                      seconds=time.perf_counter() - started)
        write_json(self.output / f'{label}-test.json', result)
        return result
