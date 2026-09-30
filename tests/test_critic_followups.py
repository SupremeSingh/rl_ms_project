import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from critic_followups import command, conditions
from compare_critic_layers import positions


def test_cumulative_and_reset_controls():
    variants=conditions('cumulative',10)
    assert variants['frozen_cumulative']==dict(reset=0,refresh=0)
    assert variants['frozen_batch']==dict(reset=1,refresh=0)
    refresh=conditions('refresh',10)
    assert refresh['frozen_reset']['reset']==refresh['refreshed_reset']['reset']==10
    args=command(Path('/tmp/study'),'case',42,60,.01,.99,refresh['refreshed_reset'])
    assert 'experiment.encoder_refresh_interval=10' in args
    assert 'experiment.statistics_reset_interval=10' in args
    assert 'actor_rollout_ref.model.critic_feature_mode=frozen' in args
    assert 'experiment.replay_capacity=0' in args


def test_prefix_sampling_same_for_all_layers():
    p=positions(10,3,100)
    assert p==positions(10,3,100) and p[0]==0 and len(p)==8
    assert all(1<=i<100 for i in p[1:])
    assert positions(0,0,1)==[0]*8


def test_layer_report_requires_identical_prefixes(tmp_path):
    import json
    import numpy as np
    import pytest
    import torch
    from compare_critic_layers import summarize, LAYERS, HEADS
    for name in LAYERS:
        folder=tmp_path/name;folder.mkdir();(folder/'heads').mkdir()
        metrics=dict(brier=.25,accuracy=.5)
        (folder/'summary.json').write_text(json.dumps(dict(ridge=dict(test=metrics),
            heads=[dict(kind=k,test=metrics) for k in HEADS])))
        data=dict(y=torch.tensor([0.,1.]),question=torch.tensor([0,1]),
                  response=torch.tensor([0,0]),position=torch.tensor([0,1]))
        torch.save(data,folder/'test.pt')
        for kind in list(HEADS)+['ridge_value']:
            file='ridge_value-predictions.npy' if kind=='ridge_value' else f'{kind}-42-predictions.npy'
            np.save(folder/'heads'/file,np.array([.5,.5]))
    summarize(tmp_path,dict(config=dict(seeds=[42])))
    report=json.loads((tmp_path/'summary.json').read_text())
    assert all(r['question_mean_brier_difference']==0 for r in report['paired'].values())
    data['position'][0]=2;torch.save(data,tmp_path/'third/test.pt')
    with pytest.raises(ValueError,match='Unpaired'):
        summarize(tmp_path,dict(config=dict(seeds=[42])))


def test_seven_way_layer_study_budget_and_locked_ppo():
    variants=conditions('layers',10)
    assert len(variants)==7
    for name,setting in variants.items():
        args=command(Path('/tmp/study'),name,42,60,.01,.99,setting)
        assert 'trainer.total_training_steps=60' in args
        if name=='ppo':
            assert 'experiment.ppo_profile=more_fitting' in args
            assert 'experiment.critic_statistics=cumulative' not in args
        else:
            assert 'experiment.critic_statistics=cumulative' in args
            assert 'experiment.encoder_refresh_interval=0' in args
            assert 'experiment.statistics_reset_interval=0' in args
            assert f'actor_rollout_ref.model.critic_feature_layer={setting["layer"]}' in args
            assert f'experiment.method={setting["method"]}' in args
            assert f'experiment.critic_lambda={setting["trace"]}' in args
