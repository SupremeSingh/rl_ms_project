"""Persistent frozen Transformers backend. Optional dependencies load on demand."""
import os
from pathlib import Path
import torch
from .config import completion_prompt, immutable_revision
from .features import rms_coordinates
from .layers import sampled_block_states
from .storage import model_identity


class TransformersBackend:
    def __init__(self, path, *, device='cuda'):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='1':
            raise ValueError('Unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE for reproducible frozen features')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.path=Path(path).resolve()
        self.model_files=model_identity(self.path)
        self.tokenizer=AutoTokenizer.from_pretrained(self.path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(self.path,local_files_only=True,
            trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='eager').to(device).eval().requires_grad_(False)
        if self.model.config.model_type!='qwen2': raise ValueError('Expected a Qwen2 backbone')
        self.device=torch.device(device)
        self.num_layers=len(self.model.model.layers)
        self.hidden_size=self.model.config.hidden_size
        self.max_context=self.model.config.max_position_embeddings
        self.rms_eps=self.model.config.rms_norm_eps
        if (self.num_layers,self.hidden_size)!=(28,1536):
            raise ValueError('Version 1 requires the 28-block, 1536-wide Qwen2.5-Math-1.5B encoder')

    def prompt_ids(self, question):
        return self.tokenizer.encode(completion_prompt(question),add_special_tokens=False)

    def prefix_ids(self, text):
        return self.tokenizer.encode(text,add_special_tokens=False)

    def check_tokens(self, tokens):
        if not tokens or len(tokens)>self.max_context:
            raise ValueError('Empty input or native context exceeded; inputs are never silently truncated')
        if any(type(t) is not int or t<0 or t>=self.model.config.vocab_size for t in tokens):
            raise ValueError('Invalid token ID')

    @torch.inference_mode()
    def state_features(self, tokens, layers):
        self.check_tokens(tokens)
        ids=torch.tensor([tokens],device=self.device)
        # A dummy final token gives the same pre-action indexing used in training;
        # causal attention prevents it from influencing the selected prefix state.
        if len(tokens)==self.max_context:
            captured={}
            hooks=[]
            def capture(layer):
                def hook(module, inputs, output):
                    hidden=output[0] if isinstance(output,tuple) else output
                    captured[layer]=hidden[0,-1:].detach().float().cpu()
                return hook
            try:
                for layer in layers: hooks.append(self.model.model.layers[layer].register_forward_hook(capture(layer)))
                self.model.model(input_ids=ids,use_cache=False,return_dict=True)
            finally:
                for hook in hooks: hook.remove()
            blocks=captured
        else:
            dummy=torch.cat([ids,ids[:,-1:]],1)
            blocks=sampled_block_states(self.model.model,dummy,len(tokens),1,[0],layers)
        return torch.cat([rms_coordinates(blocks[layer].float(),self.rms_eps) for layer in layers],1)

    @torch.inference_mode()
    def trajectory_features(self, prompt, answer, positions, layers):
        self.check_tokens(prompt+answer)
        ids=torch.tensor([prompt+answer],device=self.device)
        blocks=sampled_block_states(self.model.model,ids,len(prompt),len(answer),positions,layers)
        return torch.cat([rms_coordinates(blocks[layer].float(),self.rms_eps) for layer in layers],1)

    @torch.inference_mode()
    def generate(self, prompt, count, max_tokens, seed):
        if len(prompt)+max_tokens>self.max_context:
            raise ValueError('Prompt and generation budget exceed native context')
        self.check_tokens(prompt)
        torch.manual_seed(seed)
        if self.device.type=='cuda': torch.cuda.manual_seed_all(seed)
        ids=torch.tensor([prompt],device=self.device)
        result=self.model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),
            do_sample=True,temperature=1.,top_p=1.,top_k=0,num_return_sequences=count,
            max_new_tokens=max_tokens,pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,use_cache=True)
        rows=[]
        for sequence in result[:,len(prompt):].cpu().tolist():
            if self.tokenizer.eos_token_id in sequence:
                sequence=sequence[:sequence.index(self.tokenizer.eos_token_id)+1]
            rows.append(dict(token_ids=sequence,response=self.tokenizer.decode(sequence,skip_special_tokens=True),
                             capped=len(sequence)==max_tokens and sequence[-1]!=self.tokenizer.eos_token_id))
        return rows


def resolve_model(config):
    if config.model_path: return Path(config.model_path).resolve(), config.model_revision
    from huggingface_hub import HfApi, snapshot_download
    revision=HfApi().model_info(config.model_id,revision=config.model_revision).sha
    if not immutable_revision(revision): raise ValueError('Model revision did not resolve to an immutable commit SHA')
    return Path(snapshot_download(config.model_id,revision=revision)), revision
