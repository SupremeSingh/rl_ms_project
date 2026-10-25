"""Small real-model package train -> artifact -> score -> PPO-filter round trip."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch

from slice_critic import Critic, State, TrainingConfig, train_critic
from slice_critic.legacy import export_filter
from slice_critic.inference import load_filter
from slice_critic.storage import write_json, read_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Prepared Numina data/questions.json')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('models/qwen-math'))
    parser.add_argument('--verifier-python', type=Path, default=Path('.venv-ppo-verifier/bin/python'))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    raw = json.loads(args.source.read_text())
    questions = [q for q in raw['questions'] if q['split'] == 'train'][:40]
    if len(questions) != 40:
        raise ValueError('Require 40 saved training questions; no validation/test reuse')
    data = out/'questions.jsonl'
    data.write_text('\n'.join(json.dumps(dict(problem=q['question'], answer=q['ground_truth']))
                              for q in questions)+'\n')
    config = TrainingConfig(dataset=str(data), model_path=str(args.model.resolve()),
        verifier_python=str(args.verifier_python.absolute()), max_questions=40,
        responses_per_question=2, max_new_tokens=512, epochs=2, compression_width=768)
    write_json(out/'config.json', asdict(config))
    artifact = train_critic(config, out/'training')
    manifest, checkpoint, calibration = read_artifact(artifact)
    # Release generation/extraction memory before loading the persistent scorer.
    torch.cuda.empty_cache()
    critic = Critic.load(artifact, model_path=args.model.resolve())
    state = State(question=questions[0]['question'])
    features = critic.encode_batch([state])
    result = critic.score_batch([state])[0]
    manual = torch.cat([features.double(), torch.ones(1,1)],1)@checkpoint['weights']
    assert abs(result['value']-float(manual[0])) < 1e-8
    exported = export_filter(artifact, out/'ppo-filter.pt')
    model, mean, scale = load_filter(exported['path'], exported['sha256'])
    x = critic.backend.state_features(critic.backend.prompt_ids(state.question), critic.layers)
    torch.testing.assert_close(model.features((x-mean)/scale), features)
    # The frozen basis, rather than the static fitted value head, is reused in PPO.
    write_json(out/'verification.json', dict(passed=True, artifact=str(artifact),
        blocks_one_based=manifest['features']['blocks_one_based'], feature_width=features.shape[1],
        probe_result=result, ppo_filter=exported,
        scope='40-question integration smoke; not an accuracy benchmark or an actor-training run'))
    print(json.dumps(dict(passed=True, artifact=str(artifact)), indent=2))


if __name__ == '__main__':
    main()
