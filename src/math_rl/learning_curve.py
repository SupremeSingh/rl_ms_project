"""Optional observation hook; preserves the pinned PPO/GAE and critic fitting order."""
import json
import time
from math_rl.trainer import MathTrainer
from math_rl.provenance import write_json


class LearningCurveTrainer(MathTrainer):
    def fit(self):
        self._curve_active=True;self._curve_started=time.perf_counter()
        try:return super().fit()
        finally:self._curve_active=False

    def _validate(self):
        metrics=super()._validate()
        if not getattr(self,'_curve_active',False) or self.global_steps<=0:return metrics
        step=self.global_steps
        if step%self.config.trainer.test_freq and step!=self.total_training_steps:return metrics
        old_completed=self.completed_updates
        self._curve_active=False;self.completed_updates=step
        try:
            # evaluate_test calls _validate: disabling this hook prevents recursion.
            result=self.evaluate_test(f'curve-{step}')
        finally:
            self.global_steps=step;self.completed_updates=old_completed;self._curve_active=True
        rows=[json.loads(v) for v in (self.output/'linear-critic/metrics.jsonl').read_text().splitlines()]
        if [r['step'] for r in rows]!=list(range(1,step+1)):raise RuntimeError('Missing training token/critic accounting')
        record=dict(step=step,answers=step*64,response_tokens=sum(r['transitions'] for r in rows),
            online_seconds=time.perf_counter()-self._curve_started,accuracy=result['accuracy'],
            questions=result['questions'],truncation_rate=result['truncation_rate'],
            evaluation_seconds=result['seconds'],critic_fit_seconds=sum(r['fit_seconds'] for r in rows),
            evaluation_split=self.config.experiment.evaluation_split,
            scope='After actor update; includes validation/test monitoring overhead; excludes initialization/base evaluation')
        write_json(self.output/f'curve-{step}.json',record)
        return metrics
