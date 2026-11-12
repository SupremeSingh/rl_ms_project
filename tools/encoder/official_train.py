"""Run unchanged official scripts with native PyTorch AMP, not container Apex.

The container exposes an Apex distribution without apex.amp. Transformers 4.44.2
imports it merely on discovery, even for BF16 jobs. This process-local optional
dependency override does not edit either upstream package or install anything.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys


def native_amp():
    import transformers
    from transformers.utils import import_utils
    if transformers.__version__!='4.44.2':
        raise ValueError('Native AMP launcher is scoped to pinned Transformers 4.44.2')
    import_utils._apex_available=False
    from transformers import Trainer
    assert not import_utils.is_apex_available()
    return Trainer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('script',nargs='?',type=Path)
    p.add_argument('config',nargs='?',type=Path)
    a=p.parse_args()
    native_amp()
    print('Encoder Trainer import passed: native PyTorch AMP; optional Apex disabled for this process.',flush=True)
    if a.check:return
    if a.script is None or a.config is None:p.error('script and config required')
    cfg=json.loads(a.config.read_text())
    if not cfg.get('bf16') or cfg.get('fp16',False) or cfg.get('half_precision_backend','auto')=='apex':
        raise ValueError('Official encoder launcher requires native BF16, not Apex/FP16')
    sys.argv=[str(a.script),str(a.config)]
    runpy.run_path(str(a.script),run_name='__main__')


if __name__=='__main__':main()
