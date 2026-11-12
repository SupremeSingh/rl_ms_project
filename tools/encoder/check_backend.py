"""CPU-only actual-attention probe on tiny randomly initialized official Qwen2 models."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import torch
from transformers import Qwen2Config,Qwen2Model
from llm2vec.models import Qwen2BiModel
from math_rl.encoder_features import attention_probe

torch.set_num_threads(4)
results={}
for backend in ('eager','sdpa'):
    results[backend]={}
    for cls in (Qwen2Model,Qwen2BiModel):
        torch.manual_seed(17)
        config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,
                          num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)
        config._attn_implementation=backend
        model=cls(config).eval().requires_grad_(False)
        results[backend][cls.__name__]=attention_probe(model,[1,2,3,4,5])
print(json.dumps(results,indent=2))
print('Nonzero Qwen2BiModel first-token sensitivity is required for bidirectionality.')
print('This diagnostic reports backend behavior; it is not the encoder release gate.')
