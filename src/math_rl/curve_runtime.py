"""Driver-only measurement hooks; learning objectives and worker classes unchanged."""
import time
from .provenance import write_json


def install():
    from .trainer import MathTrainer
    original_fit=MathTrainer.fit
    original_validate=MathTrainer._validate
    original_save=MathTrainer._save_checkpoint
    original_evaluate=MathTrainer.evaluate_test

    def evaluate(self,label):
        self._curve_label=label
        try:return original_evaluate(self,label)
        finally:self._curve_label='periodic'

    def validate(self,*args,**kwargs):
        start=time.perf_counter();result=original_validate(self,*args,**kwargs);elapsed=time.perf_counter()-start
        key='val-core/numina_numeric/reward/mean@1'
        if key not in result:raise ValueError('Missing Numina monitoring accuracy')
        started=getattr(self,'_curve_started',None)
        active=started is not None and not getattr(self,'_curve_finished',False)
        train_seconds=(time.perf_counter()-started-getattr(self,'_curve_eval_seconds',0)-getattr(self,'_curve_save_seconds',0)-elapsed) if active else 0.
        row=dict(step=int(self.global_steps),label=getattr(self,'_curve_label','periodic'),accuracy=float(result[key]),
            evaluation_seconds=elapsed,training_seconds=max(0.,train_seconds),
            training_gpu_hours=max(0.,train_seconds)*2/3600)
        if active:self._curve_eval_seconds+=elapsed
        with (self.output/'learning-curve.jsonl').open('a') as f:
            import json
            f.write(json.dumps(row,allow_nan=False)+'\n')
        return result

    def save(self,*args,**kwargs):
        start=time.perf_counter();result=original_save(self,*args,**kwargs)
        self._curve_save_seconds=getattr(self,'_curve_save_seconds',0)+time.perf_counter()-start
        return result

    def fit(self):
        self._curve_started=time.perf_counter();self._curve_finished=False
        self._curve_eval_seconds=0.;self._curve_save_seconds=0.
        result=original_fit(self)
        elapsed=time.perf_counter()-self._curve_started;self._curve_finished=True
        write_json(self.output/'training-clock.json',dict(fit_wall_seconds=elapsed,evaluation_seconds=self._curve_eval_seconds,
            checkpoint_seconds=self._curve_save_seconds,training_seconds=elapsed-self._curve_eval_seconds-self._curve_save_seconds,
            scope='Driver elapsed time excludes measured periodic validation and checkpoint intervals; includes rollout generation and original training operations. Initialization and outer base/final evaluations are separate.'))
        return result

    MathTrainer.evaluate_test=evaluate;MathTrainer._validate=validate
    MathTrainer._save_checkpoint=save;MathTrainer.fit=fit
