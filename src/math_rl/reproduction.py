"""Lock final experiment inputs; never copy trained actors or silently bless changed data."""
import json
from pathlib import Path
import shutil

from slice_critic import export_ppo_features
from slice_critic.artifacts import load_validated_artifact
from slice_critic.storage import model_identity
from slice_critic.text import family_key
from .experiment import METHODS, PRIMARY
from .provenance import sha256, snapshot, write_json
from .ppo_baseline import PPO_PROFILES

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def validate_questions(questions, blocked=()):
    ids, families = set(), set()
    for q in questions:
        family = family_key(q['question'])
        if q['id'] in ids or family in families or family in blocked or q['split'] not in ('train','val','test'):
            raise ValueError('Overlapping or invalid online question families/IDs')
        ids.add(q['id']); families.add(family)
    if any(sum(q['split']==split for q in questions) < n for split,n in [('train',16),('val',1),('test',1)]):
        raise ValueError('Insufficient online question splits')


def base_manifest(out, model_hashes, layers, eps, **kwargs):
    return dict(protocol='slice-numina-online-v1', conditions=METHODS, method_subset='six_method',
        seeds=[301,302,303,304,305], steps=120, calibration_steps=60, answers_per_update=64,
        critic_lambda=.9, actor_lambda=.95, epsilon=.01, primary_comparisons=list(PRIMARY),
        layers=layers, rms_eps=eps, model_hashes=model_hashes,
        model_assets=read(ROOT/'configs/assets.json') if (ROOT/'configs/assets.json').exists() else None,
        data_hashes={p.name:sha256(p) for p in (out/'data').iterdir()}, provenance=snapshot(ROOT), **kwargs)


def prepare_from_artifact(artifact, data, out):
    artifact, data, out = map(lambda p: Path(p).resolve(), (artifact,data,out))
    if any(out == p or p in out.parents for p in (artifact, data)):
        raise ValueError('Output must be separate from source inputs')
    manifest, checkpoint, _ = load_validated_artifact(artifact)
    features=manifest['features']
    if (features['width'] != 768 or features['compression'] != 'sparse'
            or features['layers_zero_based'] != [27,26,24,20,12]
            or checkpoint['mean'].numel() != 7680):
        raise ValueError('The final comparison requires the five-block sparse-768 representation')
    prepared = read(data/'manifest.json')
    if prepared['protocol'] != 'slice-numina-data-v1':
        raise ValueError('Expected prepared Numina data')
    for name, digest in prepared['hashes'].items():
        if sha256(data/name) != digest:
            raise ValueError(f'Prepared data changed: {name}')
    blocked = set(manifest['representation_training_families']) | set(manifest['selection_families'])
    questions = read(data/'online/questions.json')['questions']
    validate_questions(questions, blocked)
    if model_identity(ROOT/'models/qwen-math') != manifest['model']['files']:
        raise ValueError('Artifact and actor model identities differ')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(data/'online', out/'data')
    exported = export_ppo_features(artifact, out/'filters/sparse.pt')
    m = base_manifest(out, manifest['model']['files'], exported['layers'], exported['rms_eps'],
        filters={'sparse':exported}, revisions=prepared['revisions'],
        representation_artifact=dict(path=str(artifact),sha256=sha256(artifact/'manifest.json')),
        filter_pretraining=dict(scope='Offline representation cost is additional; see artifact training report'),
        scope='New SLICE artifact and family-disjoint Numina data. New experiment, not an exact historical filter reconstruction.')
    write_json(out/'manifest.json',m)


def import_final(source, out):
    """One-time input capture from the final six-method run, independent of old launchers."""
    source, out = Path(source).resolve(), Path(out).resolve()
    if out == source or source in out.parents:
        raise ValueError('Output must be separate from the historical study')
    old = read(source/'manifest.json')
    if (old.get('protocol') != 'numina-online-v1' or old['conditions'] != METHODS or old['steps'] != 120 or old['critic_lambda'] != .9
            or old['seeds'] != [301,302,303,304,305]):
        raise ValueError('Expected the completed final six-method lambda-0.9 study')
    for name,digest in old['data_hashes'].items():
        if sha256(source/'data'/name) != digest:
            raise ValueError(f'Historical data changed: {name}')
    for name,digest in old['model_hashes'].items():
        if sha256(ROOT/'models/qwen-math'/name) != digest:
            raise ValueError('Historical base model differs')
    selected=read(source/'selection.json')
    if selected['profile'] not in PPO_PROFILES:
        raise ValueError('Unknown selected PPO profile')
    # Reproduce the profile actually used by the actor runs, not just a label in
    # a selection file (early smoke code once ignored this setting).
    profiles = set()
    for name in METHODS:
        for seed in old['seeds']:
            result = read(source/f'{name}-seed{seed}/result.json')
            if result['updates'] != 120 or result['answers'] != 7680 or result['seed'] != seed:
                raise ValueError('Historical runs do not share the final budget')
            if name == 'ppo': profiles.add(result['ppo_profile'])
            elif METHODS[name]['method'] in ('ridge','lstd'):
                rows=[json.loads(line) for line in (source/f'{name}-seed{seed}/linear-critic/metrics.jsonl').read_text().splitlines()]
                if ([row['step'] for row in rows] != list(range(1,121))
                        or any(row['fitting_mode'] != 'cumulative_raw_sum' or row['critic_lambda'] != .9
                               or row['epsilon'] != .01 or row['feature_mode'] != 'frozen' for row in rows)):
                    raise ValueError('Historical critic settings differ from the final cumulative protocol')
    if len(profiles) != 1 or not profiles.issubset(PPO_PROFILES):
        raise ValueError('Historical PPO profiles differ across seeds')
    actual_profile = profiles.pop()
    original=old['filters']['sparse']
    checkpoint=source/'filters'/Path(original['path']).name
    if not checkpoint.exists(): checkpoint=Path(original['path'])
    if sha256(checkpoint) != original['sha256']:
        raise ValueError('Historical sparse filter changed')
    questions=read(source/'data/questions.json')['questions']
    validate_questions(questions)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(source/'data',out/'data')
    (out/'filters').mkdir()
    shutil.copyfile(checkpoint,out/'filters/sparse.pt')
    for name in ('manifest.json','selection.json','summary.json'):
        shutil.copyfile(source/name,out/f'historical-{name}')
    # Old data/filter identities are checked above; code provenance is deliberately new.
    m=base_manifest(out,old['model_hashes'],old['layers'],old['rms_eps'],
        filters={'sparse':dict(original,path='sparse.pt')},revisions=old['revisions'],
        historical_source=dict(path=str(source),manifest_sha256=sha256(source/'manifest.json')),
        filter_pretraining=old.get('filter_pretraining'),
        scope='Fresh actors from the final locked data/filter/profile. No old actors or test-based selection reused; hardware nondeterminism remains.')
    write_json(out/'manifest.json',m)
    write_json(out/'selection.json',dict(selected,profile=actual_profile,reused_from=str(source),
        declared_profile=selected['profile'],
        selection_data='Replay actual historical training profile; original selection preserved separately. No test-based reselection.'))
    for name in ('assets.json','container.sha256','environment.txt','ppo-verifier-environment.txt'):
        path=ROOT/'configs'/name
        if path.exists():
            (out/'runtime').mkdir(exist_ok=True)
            shutil.copyfile(path,out/'runtime'/name)
