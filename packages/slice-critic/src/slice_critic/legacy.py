"""Export a feature-only checkpoint for the existing PPO configuration contract."""
from pathlib import Path
from .storage import read_artifact, save_tensor, sha256


def export_filter(artifact, output):
    manifest, checkpoint, _ = read_artifact(artifact)
    features = manifest['features']
    if features['width'] != 768:
        raise ValueError('The existing PPO adapter requires width 768')
    output = Path(output).resolve()
    if output.exists() or output.with_suffix('.sha256.json').exists():
        raise ValueError('Use a new filter path; never replace a published feature basis')
    record = dict(state=checkpoint['compressor'], mean=checkpoint['mean'],
                  scale=checkpoint['scale'], width=features['width'],
                  kind=features['compression'], layers=features['layers_zero_based'],
                  rms_eps=features['rms_eps'], model=manifest['model'],
                  source_artifact_manifest_sha256=sha256(Path(artifact)/'manifest.json'),
                  representation_training_families=manifest['representation_training_families'],
                  selection_families=manifest['selection_families'])
    save_tensor(output, record)
    return dict(path=str(output), sha256=sha256(output),
                layers=features['layers_zero_based'], rms_eps=features['rms_eps'],
                scope='Features only. PPO fits fresh cross-fitted value heads; exclude these training/selection families.')
