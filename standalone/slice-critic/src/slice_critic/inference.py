"""Immutable artifact loading and persistent local batch scoring."""
from dataclasses import dataclass
from pathlib import Path
import torch
from .bottleneck import Bottleneck
from .config import INSTRUCTION
from .storage import read_artifact, model_identity, sha256
from .artifacts import load_validated_artifact, validate_payload


@dataclass(frozen=True)
class State:
    question: str
    prefix: str = ''
    prefix_token_ids: list[int] | None = None


class Critic:
    def __init__(self, manifest, checkpoint, calibration, backend):
        validate_payload(manifest, checkpoint, calibration)
        if manifest.get('prompt_instruction') != INSTRUCTION:
            raise ValueError('Artifact prompt differs from the supported completion contract')
        self.manifest=manifest; self.calibration=calibration; self.backend=backend
        width=manifest['features']['width']; self.layers=manifest['features']['layers_zero_based']
        self.mean=checkpoint['mean'].float().cpu(); self.scale=checkpoint['scale'].float().cpu()
        self.weights=checkpoint['weights'].double().cpu()
        if (self.mean.shape!=self.scale.shape or self.mean.ndim!=1 or self.weights.shape!=(width+1,)
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all()
                or (self.scale<=0).any() or not torch.isfinite(self.weights).all()):
            raise ValueError('Invalid feature normalization or linear head')
        self.compressor=Bottleneck(len(self.mean),width)
        self.compressor.load_state_dict(checkpoint['compressor'],strict=True)
        if any(not torch.isfinite(p).all() for p in self.compressor.parameters()):
            raise ValueError('Nonfinite compressor')
        self.compressor.eval().requires_grad_(False)
        if len(self.mean)!=len(self.layers)*backend.hidden_size or len(set(self.layers))!=len(self.layers):
            raise ValueError('Model and saved feature dimensions differ')
        if any(i<0 or i>=backend.num_layers for i in self.layers): raise ValueError('Invalid saved layers')
        if backend.rms_eps!=manifest['features']['rms_eps']: raise ValueError('Model RMS normalization differs')

    @classmethod
    def load(cls, path, device='cuda', *, model_path=None):
        manifest,checkpoint,calibration=load_validated_artifact(path)
        saved=manifest['model']
        resolved=Path(model_path or saved['local_path'])
        if not resolved.is_dir():
            from huggingface_hub import snapshot_download
            if not saved.get('revision'): raise ValueError('Provide model_path for this local-only model artifact')
            resolved=Path(snapshot_download(saved['id'],revision=saved['revision']))
        if model_identity(resolved)!=saved['files']:
            raise ValueError('Encoder/tokenizer differs from the artifact; no automatic substitution')
        from .backend import TransformersBackend
        backend=TransformersBackend(resolved,device=device)
        return cls(manifest,checkpoint,calibration,backend)

    @torch.inference_mode()
    def encode_batch(self, states):
        """Return detached CPU features, preserving order; encoder loaded once."""
        features=[]
        for state in states:
            if isinstance(state,dict): state=State(**state)
            if not isinstance(state,State) or not isinstance(state.question,str) or not state.question.strip():
                raise ValueError('Each state needs a nonempty question')
            if state.prefix_token_ids is not None and state.prefix:
                raise ValueError('Supply prefix text OR exact token IDs, not both')
            if not isinstance(state.prefix,str): raise ValueError('prefix must be text')
            if state.prefix_token_ids is not None and (not isinstance(state.prefix_token_ids, list)
                    or any(type(t) is not int or t < 0 for t in state.prefix_token_ids)):
                raise ValueError('prefix_token_ids must be a list of nonnegative integers')
            prefix=state.prefix_token_ids if state.prefix_token_ids is not None else self.backend.prefix_ids(state.prefix)
            ids=self.backend.prompt_ids(state.question)+prefix
            x=self.backend.state_features(ids,self.layers).float().cpu()
            if x.shape != (1, self.mean.numel()) or not torch.isfinite(x).all():
                raise ValueError('Encoder returned invalid state features')
            z=self.compressor.features((x-self.mean)/self.scale)
            if not torch.isfinite(z).all(): raise ValueError('Nonfinite compressed features')
            features.append(z)
        return torch.cat(features) if features else torch.empty((0,len(self.weights)-1))

    @torch.inference_mode()
    def score_batch(self, states):
        features=self.encode_batch(states)
        x=torch.cat([features.double(),torch.ones(len(features),1)],1)
        raw=x@self.weights
        if not torch.isfinite(raw).all(): raise ValueError('Nonfinite critic values')
        probability=None
        if self.calibration.get('method')=='platt':
            probability=torch.sigmoid(raw*self.calibration['slope']+self.calibration['intercept'])
        return [dict(value=float(v),probability=float(probability[i]) if probability is not None else None,
                     critic_version=self.manifest['version'],model_id=self.manifest['model']['id'])
                for i,v in enumerate(raw)]


def load_filter(path, checksum, *, layers=None, eps=None, model_path=None):
    """Compatibility loader for existing PPO bottleneck checkpoints (768 wide)."""
    if not checksum or sha256(path)!=checksum: raise ValueError('Bottleneck checkpoint checksum mismatch')
    record=torch.load(Path(path),map_location='cpu',weights_only=True)
    if layers is not None and 'layers' in record and list(layers)!=record['layers']:
        raise ValueError('Configured layers differ from the published feature basis')
    if eps is not None and 'rms_eps' in record and eps!=record['rms_eps']:
        raise ValueError('Configured RMS epsilon differs from the published feature basis')
    if model_path is not None and 'model' in record and model_identity(model_path)!=record['model']['files']:
        raise ValueError('Configured encoder differs from the published feature basis')
    model=Bottleneck(record['mean'].numel(),record['width'])
    model.load_state_dict(record['state'],strict=True)
    model.eval().requires_grad_(False)
    if (record['width']!=768 or record['mean'].ndim!=1 or record['mean'].shape!=record['scale'].shape
            or any(not torch.isfinite(p).all() for p in model.parameters())
            or not torch.isfinite(record['mean']).all()
            or not torch.isfinite(record['scale']).all() or (record['scale']<=0).any()):
        raise ValueError('Expected a finite frozen 768-dimensional filter')
    return model,record['mean'].float(),record['scale'].float()
