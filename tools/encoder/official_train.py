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
import os


def native_amp():
    # Encoder training is exclusively PyTorch. Avoid probing unused TF/Flax stacks.
    os.environ['USE_TF']='0'
    os.environ['USE_FLAX']='0'
    print('[encoder startup] Importing Transformers',file=sys.stderr,flush=True)
    import transformers
    from transformers.utils import import_utils
    if transformers.__version__!='4.44.2':
        raise ValueError('Native AMP launcher is scoped to pinned Transformers 4.44.2')
    import_utils._apex_available=False
    print('[encoder startup] Importing Trainer (Apex disabled)',file=sys.stderr,flush=True)
    from transformers import Trainer
    assert not import_utils.is_apex_available()
    return Trainer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check',action='store_true')
    p.add_argument('--unpadded-simcse',action='store_true',help='Singleton encoder execution; preserve the official joint contrastive loss')
    p.add_argument('--traceback-after',type=int,default=0,
                   help='Dump Python stack after this many seconds during startup')
    p.add_argument('script',nargs='?',type=Path)
    p.add_argument('config',nargs='?',type=Path)
    a=p.parse_args()
    if a.traceback_after:
        import faulthandler
        faulthandler.dump_traceback_later(a.traceback_after,repeat=False)
    native_amp()
    if a.traceback_after:
        faulthandler.cancel_dump_traceback_later()
    print('Encoder Trainer import passed: native PyTorch AMP; optional Apex disabled for this process.',flush=True)
    if a.check:return
    if a.script is None or a.config is None:p.error('script and config required')
    if a.unpadded_simcse:
        if a.script.name!='run_simcse.py':p.error('Unpadded adapter is only for SimCSE')
        import llm2vec
        from unpadded_adapter import make_unpadded
        llm2vec.LLM2Vec=make_unpadded(llm2vec.LLM2Vec)
        print('Unpadded singleton encoding enabled; joint contrastive batch retained',flush=True)
    cfg=json.loads(a.config.read_text())
    if not cfg.get('bf16') or cfg.get('fp16',False) or cfg.get('half_precision_backend','auto')=='apex':
        raise ValueError('Official encoder launcher requires native BF16, not Apex/FP16')
    sys.argv=[str(a.script),str(a.config)]
    runpy.run_path(str(a.script),run_name='__main__')


if __name__=='__main__':main()
