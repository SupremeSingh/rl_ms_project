"""Bounded local training and inference; preview never downloads models or data."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('preview', 'train'):
        child = commands.add_parser(name)
        child.add_argument('--config', type=Path, required=True)
        if name == 'train':
            child.add_argument('--out', type=Path, required=True)
            child.add_argument('--device', default='cuda')
    child = commands.add_parser('inspect')
    child.add_argument('artifact', type=Path)
    child = commands.add_parser('score')
    child.add_argument('artifact', type=Path)
    child.add_argument('--states', type=Path, required=True, help='JSONL question/prefix states')
    child.add_argument('--model-path', type=Path)
    child.add_argument('--device', default='cuda')
    child.add_argument('--batch-size', type=int, default=16)
    child = commands.add_parser('export-filter')
    child.add_argument('artifact', type=Path)
    child.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ('preview', 'train'):
        from .config import TrainingConfig
        config = TrainingConfig(**json.loads(args.config.read_text())).validate()
        if args.command == 'preview':
            result = config.preview()
        else:
            from .training import train_critic
            result = dict(artifact=str(train_critic(config, args.out, device=args.device)))
    elif args.command == 'inspect':
        from .storage import read_artifact, read_json
        manifest, _, calibration = read_artifact(args.artifact)
        result = dict(manifest=manifest, calibration=calibration,
                      evaluation=read_json(args.artifact/'evaluation.json'))
    elif args.command == 'export-filter':
        from .legacy import export_filter
        result = export_filter(args.artifact, args.out)
    else:
        if args.batch_size < 1:
            parser.error('--batch-size must be positive')
        from .inference import Critic
        critic = Critic.load(args.artifact, device=args.device, model_path=args.model_path)
        batch = []
        with args.states.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                batch.append(json.loads(line))
                if len(batch) == args.batch_size:
                    for row in critic.score_batch(batch):
                        print(json.dumps(row, allow_nan=False))
                    batch.clear()
        for row in critic.score_batch(batch):
            print(json.dumps(row, allow_nan=False))
        return
    print(json.dumps(result, indent=2, allow_nan=False))
