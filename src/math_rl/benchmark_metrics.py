"""Durable benchmark timing records, independent of Ray console delivery."""
import json
import math
from pathlib import Path


def append_metrics(path,step,data,workload_sha256):
    if 'timing_s/step' not in data:return
    values={}
    for key,value in data.items():
        if key.startswith(('timing_s/','perf/','response_length/','global_seqlen/')):
            try:number=float(value)
            except (TypeError,ValueError):continue
            if not math.isfinite(number):raise ValueError('Nonfinite benchmark metric')
            values[key]=number
    if type(step) is not int or step<1:raise ValueError('Invalid benchmark step')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(dict(step=step,metrics=values,workload_sha256=workload_sha256),allow_nan=False)+'\n')
        f.flush()


def read_steps(path,total,workload_sha256):
    records=[json.loads(line) for line in Path(path).read_text().splitlines()]
    if [r['step'] for r in records]!=list(range(1,total+1)):raise ValueError('Missing, duplicated or unordered benchmark timing steps')
    for row in records:
        if row['workload_sha256']!=workload_sha256:raise ValueError('Timing workload identity mismatch')
        if not math.isfinite(row['metrics']['timing_s/step']) or row['metrics']['timing_s/step']<0:raise ValueError('Invalid step timing')
    return {r['step']:r['metrics'] for r in records}
