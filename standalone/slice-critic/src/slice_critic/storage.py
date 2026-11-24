"""Atomic stage files and immutable, checksummed artifacts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch

SCHEMA = 'slice-critic-v1'


def sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, record, *, sort_keys=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd, name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as handle:
            json.dump(record,handle,indent=2,allow_nan=False,sort_keys=sort_keys); handle.write('\n')
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


def save_tensor(path, record):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    os.close(fd)
    try:
        torch.save(record,name); os.replace(name,path)
        write_json(path.with_suffix('.sha256.json'),{'sha256':sha256(path)})
    finally:
        if os.path.exists(name): os.unlink(name)


def load_tensor(path):
    path=Path(path)
    if sha256(path)!=read_json(path.with_suffix('.sha256.json'))['sha256']:
        raise ValueError(f'Checkpoint checksum mismatch: {path}')
    return torch.load(path,map_location='cpu',weights_only=True)


def model_identity(path):
    path=Path(path).resolve()
    files={p.name:sha256(p) for p in sorted(path.iterdir())
           if p.is_file() and (p.suffix in ('.json','.safetensors','.model','.txt') or p.name.endswith('.bin'))}
    if 'config.json' not in files or not any(n.endswith(('.safetensors','.bin')) for n in files):
        raise ValueError('Expected a complete local Hugging Face model snapshot')
    return files


def publish(out, manifest, checkpoint, calibration, evaluation):
    """Publish only after all components are complete; never overwrite a version."""
    out=Path(out)
    if out.exists(): raise ValueError(f'Artifact version already exists: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.artifact-',dir=out.parent) as temp:
        folder=Path(temp)
        torch.save(checkpoint,folder/'checkpoint.pt')
        write_json(folder/'calibration.json',calibration)
        write_json(folder/'evaluation.json',evaluation)
        record=dict(manifest,schema=SCHEMA,files={p.name:sha256(p) for p in folder.iterdir()})
        write_json(folder/'manifest.json',record)
        # A competing publisher must fail rather than replace an existing version.
        if out.exists(): raise ValueError('Artifact was published concurrently')
        folder.rename(out)
    return out


def read_artifact(path):
    path=Path(path)
    manifest=read_json(path/'manifest.json')
    if manifest.get('schema')!=SCHEMA: raise ValueError('Unsupported critic artifact schema')
    required={'checkpoint.pt','calibration.json','evaluation.json'}
    if set(manifest.get('files',{}))!=required: raise ValueError('Incomplete critic artifact')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or sha256(path/name)!=digest:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    return manifest,torch.load(path/'checkpoint.pt',map_location='cpu',weights_only=True),read_json(path/'calibration.json')


def implementation_identity():
    """Pin all package sources that can affect resumed features or training."""
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
